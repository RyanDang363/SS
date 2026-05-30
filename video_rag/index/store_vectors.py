"""Stage 12: persist embedding records in a searchable local vector index.

Reads Stage 11 embedding JSONL and writes vectors plus retrieval metadata into
a Chroma collection. This stage does NOT create embeddings, rerank, answer
questions, or run evaluation.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

from video_rag.io_utils import read_jsonl, write_json
from video_rag.schemas import Chunk, EmbeddingRecord, VectorStoreManifest

PathLike = str | Path

VARIANT_TO_FIELD = {
    "transcript_only": "combined_text_transcript_only",
    "transcript_ocr": "combined_text_transcript_ocr",
    "transcript_vlm": "combined_text_transcript_vlm",
    "transcript_ocr_vlm": "combined_text_all",
}


def _format_chunk_seconds(chunk_seconds: int | float) -> str:
    value = float(chunk_seconds)
    if value <= 0:
        raise ValueError("chunk_seconds must be greater than 0")
    return f"{value:g}"


def _index_dir_name(video_id: str, chunk_seconds: int | float, variant: str) -> str:
    label = _format_chunk_seconds(chunk_seconds)
    return f"{video_id}_{label}s_{variant}"


def _select_document_text(chunk: Chunk, variant: str) -> str | None:
    field = VARIANT_TO_FIELD.get(variant)
    if field is None:
        return None
    value = getattr(chunk, field, None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _load_chunk_documents(
    chunks_path: Path,
    variant: str,
) -> dict[str, str]:
    if not chunks_path.exists():
        return {}
    documents: dict[str, str] = {}
    for chunk in read_jsonl(chunks_path, Chunk):
        text = _select_document_text(chunk, variant)
        if text is not None:
            documents[chunk.chunk_id] = text
    return documents


def _validate_embeddings(
    embeddings: list[EmbeddingRecord],
    *,
    video_id: str,
    variant: str,
    embeddings_path: Path,
) -> int:
    if not embeddings:
        raise ValueError(f"embedding file is empty: {embeddings_path}")

    vector_dims: set[int] = set()
    for embedding in embeddings:
        label = f"{embeddings_path}:{embedding.chunk_id}"
        if embedding.video_id != video_id:
            raise ValueError(
                f"{label}: video_id {embedding.video_id!r} does not match {video_id!r}"
            )
        if embedding.embedding_variant != variant:
            raise ValueError(
                f"{label}: embedding_variant {embedding.embedding_variant!r} "
                f"does not match {variant!r}"
            )
        actual_dim = len(embedding.vector)
        vector_dims.add(actual_dim)
        if embedding.vector_dim is not None and embedding.vector_dim != actual_dim:
            raise ValueError(
                f"{label}: vector_dim {embedding.vector_dim} does not match "
                f"len(vector) {actual_dim}"
            )

    if len(vector_dims) != 1:
        raise ValueError(
            f"{embeddings_path}: inconsistent vector dimensions {sorted(vector_dims)}"
        )
    return next(iter(vector_dims))


def _embedding_metadata(embedding: EmbeddingRecord) -> dict[str, str | float]:
    return {
        "chunk_id": embedding.chunk_id,
        "video_id": embedding.video_id,
        "start_time": embedding.start_time,
        "end_time": embedding.end_time,
        "embedding_variant": embedding.embedding_variant,
        "embedding_model": embedding.embedding_model,
    }


def _create_chroma_client(index_dir: Path) -> Any:
    try:
        import chromadb
    except ImportError as e:
        raise RuntimeError(
            "Chroma is required to run Stage 12 vector storage. "
            'Install with: pip install -e ".[vectorstore]"'
        ) from e
    return chromadb.PersistentClient(path=str(index_dir))


def _store_in_chroma(
    index_dir: Path,
    collection_name: str,
    embeddings: list[EmbeddingRecord],
    documents_by_chunk_id: dict[str, str],
) -> None:
    client = _create_chroma_client(index_dir)
    collection = client.get_or_create_collection(name=collection_name)

    ids = [embedding.chunk_id for embedding in embeddings]
    vectors = [embedding.vector for embedding in embeddings]
    metadatas = [_embedding_metadata(embedding) for embedding in embeddings]
    documents = [
        documents_by_chunk_id.get(embedding.chunk_id, "") for embedding in embeddings
    ]

    collection.add(
        ids=ids,
        embeddings=vectors,
        metadatas=metadatas,
        documents=documents,
    )


def _reset_chroma_index(index_dir: Path, collection_name: str) -> None:
    client = _create_chroma_client(index_dir)
    try:
        client.delete_collection(collection_name)
    except ValueError:
        pass
    del client
    gc.collect()


def _index_exists(index_dir: Path) -> bool:
    return index_dir.exists() and any(index_dir.iterdir())


def store_vectors(
    video_id: str,
    data_dir: PathLike = "data",
    chunk_seconds: int | float = 30,
    variant: str = "transcript_ocr_vlm",
    backend: str = "chroma",
    overwrite: bool = False,
) -> VectorStoreManifest:
    """Read embedding records and persist them in a local vector index."""
    if backend != "chroma":
        raise ValueError(f"unsupported vector store backend: {backend!r}")

    data_root = Path(data_dir)
    chunk_seconds_value = float(chunk_seconds)
    index_name = _index_dir_name(video_id, chunk_seconds_value, variant)
    embeddings_path = data_root / "embeddings" / f"{index_name}.jsonl"
    chunks_path = data_root / "chunks" / f"{video_id}_{_format_chunk_seconds(chunk_seconds_value)}s_enriched.jsonl"
    index_dir = data_root / "indexes" / index_name
    manifest_path = index_dir / "vector_store_manifest.json"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"missing embedding file: {embeddings_path}. Run Stage 11 first."
        )

    embeddings = list(read_jsonl(embeddings_path, EmbeddingRecord))
    vector_dim = _validate_embeddings(
        embeddings,
        video_id=video_id,
        variant=variant,
        embeddings_path=embeddings_path,
    )

    if _index_exists(index_dir) and not overwrite:
        raise FileExistsError(
            f"vector index already exists: {index_dir} (pass overwrite=True)"
        )

    index_dir.mkdir(parents=True, exist_ok=True)
    if overwrite and _index_exists(index_dir):
        _reset_chroma_index(index_dir, index_name)

    documents_by_chunk_id = _load_chunk_documents(chunks_path, variant)
    _store_in_chroma(index_dir, index_name, embeddings, documents_by_chunk_id)

    manifest = VectorStoreManifest(
        video_id=video_id,
        chunk_seconds=chunk_seconds_value,
        embedding_variant=variant,
        backend=backend,
        index_path=index_dir.as_posix(),
        num_vectors=len(embeddings),
        vector_dim=vector_dim,
    )
    write_json(manifest_path, manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_rag.index.store_vectors",
        description="Store embedding records in a local vector index (Stage 12).",
    )
    p.add_argument("--video-id", required=True, help="Registered video_id.")
    p.add_argument(
        "--data-dir",
        dest="data_dir",
        type=Path,
        default=Path("data"),
        help="Artifact root (default: data).",
    )
    p.add_argument(
        "--chunk-seconds",
        type=float,
        default=30,
        help="Chunk window size used by Stage 9 (default: 30).",
    )
    p.add_argument(
        "--variant",
        default="transcript_ocr_vlm",
        help="Embedding/search-text variant (default: transcript_ocr_vlm).",
    )
    p.add_argument(
        "--backend",
        default="chroma",
        help="Vector store backend (default: chroma).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing vector index.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = store_vectors(
            video_id=args.video_id,
            data_dir=args.data_dir,
            chunk_seconds=args.chunk_seconds,
            variant=args.variant,
            backend=args.backend,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError) as e:
        print(f"FAIL  {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1

    print(
        f"Wrote vector index: {manifest.index_path} "
        f"({manifest.num_vectors} vectors, dim={manifest.vector_dim})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
