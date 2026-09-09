"""Stage 15: retrieve timestamped video evidence from a vector index.

Embeds a user query with the same embedding provider family used at indexing
time, queries the Stage 12 Chroma index, and joins hits back to Stage 10
enriched chunks when available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from video_rag.index.embedding_providers import (
    PROVIDERS,
    EmbeddingProvider,
    get_provider,
)
from video_rag.io_utils import read_json, read_jsonl
from video_rag.schemas import Chunk, RetrievalResult, VectorStoreManifest

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


def _index_name(video_id: str, chunk_seconds: int | float, variant: str) -> str:
    return f"{video_id}_{_format_chunk_seconds(chunk_seconds)}s_{variant}"


def _load_chunks(path: Path) -> dict[str, Chunk]:
    if not path.exists():
        return {}
    return {chunk.chunk_id: chunk for chunk in read_jsonl(path, Chunk)}


def _resolve_provider(
    provider: EmbeddingProvider | str, *, model: str
) -> EmbeddingProvider:
    if isinstance(provider, str):
        return get_provider(provider, model=model)
    return provider


def _create_chroma_client(index_dir: Path) -> Any:
    try:
        import chromadb
    except ImportError as e:
        raise RuntimeError(
            "Chroma is required to run retrieval. "
            'Install with: pip install -e ".[vectorstore]"'
        ) from e
    return chromadb.PersistentClient(path=str(index_dir))


def _as_float(value: Any, *, field: str, chunk_id: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{chunk_id}: metadata field {field!r} is not numeric") from e


def _selected_text(chunk: Chunk, variant: str) -> str | None:
    field = VARIANT_TO_FIELD.get(variant)
    if field is None:
        return None
    value = getattr(chunk, field, None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _build_result(
    chunk_id: str,
    score: float,
    metadata: dict[str, Any],
    document: str | None,
    chunk: Chunk | None,
    variant: str,
) -> RetrievalResult:
    video_id = str(metadata.get("video_id") or (chunk.video_id if chunk else ""))
    start_time = _as_float(
        metadata.get("start_time", chunk.start_time if chunk else None),
        field="start_time",
        chunk_id=chunk_id,
    )
    end_time = _as_float(
        metadata.get("end_time", chunk.end_time if chunk else None),
        field="end_time",
        chunk_id=chunk_id,
    )

    combined_text = _selected_text(chunk, variant) if chunk is not None else None
    if combined_text is None and document:
        combined_text = document

    return RetrievalResult(
        chunk_id=chunk_id,
        video_id=video_id,
        score=score,
        start_time=start_time,
        end_time=end_time,
        transcript_text=chunk.transcript_text if chunk is not None else None,
        ocr_text=chunk.ocr_text if chunk is not None else None,
        vlm_caption=chunk.vlm_caption if chunk is not None else None,
        combined_text=combined_text,
        frame_paths=chunk.frame_paths if chunk is not None else [],
    )


def retrieve(
    video_id: str,
    question: str,
    *,
    data_dir: PathLike = "data",
    chunk_seconds: int | float = 30,
    variant: str = "transcript_ocr_vlm",
    provider: EmbeddingProvider | str = "openai",
    model: str = "text-embedding-3-small",
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Retrieve top-k evidence chunks for a question."""
    if not video_id or not video_id.strip():
        raise ValueError("video_id must be non-empty")
    if not question or not question.strip():
        raise ValueError("question must be non-empty")
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if variant not in VARIANT_TO_FIELD:
        valid = ", ".join(sorted(VARIANT_TO_FIELD))
        raise ValueError(f"unknown variant {variant!r} (valid: {valid})")

    data_root = Path(data_dir)
    index_name = _index_name(video_id, chunk_seconds, variant)
    index_dir = data_root / "indexes" / index_name
    manifest_path = index_dir / "vector_store_manifest.json"
    chunks_path = data_root / "chunks" / f"{video_id}_{_format_chunk_seconds(chunk_seconds)}s_enriched.jsonl"

    if not index_dir.exists() or not index_dir.is_dir():
        raise FileNotFoundError(f"missing vector index directory: {index_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing vector store manifest: {manifest_path}")

    manifest = read_json(manifest_path, VectorStoreManifest)
    if manifest.video_id != video_id:
        raise ValueError(
            f"{manifest_path}: video_id {manifest.video_id!r} does not match {video_id!r}"
        )
    if abs(float(manifest.chunk_seconds) - float(chunk_seconds)) > 1e-6:
        raise ValueError(
            f"{manifest_path}: chunk_seconds {manifest.chunk_seconds} does not match "
            f"{float(chunk_seconds)}"
        )
    if manifest.embedding_variant != variant:
        raise ValueError(
            f"{manifest_path}: embedding_variant {manifest.embedding_variant!r} "
            f"does not match {variant!r}"
        )
    if manifest.backend not in (None, "chroma"):
        raise ValueError(f"unsupported vector store backend: {manifest.backend!r}")

    resolved_provider = _resolve_provider(provider, model=model)
    query_vectors = resolved_provider.embed_texts([question.strip()])
    if len(query_vectors) != 1:
        raise RuntimeError(
            f"provider returned {len(query_vectors)} query vectors for one question"
        )

    client = _create_chroma_client(index_dir)
    collection = client.get_collection(name=index_name)
    raw = collection.query(
        query_embeddings=query_vectors,
        n_results=min(top_k, manifest.num_vectors),
        include=["metadatas", "documents", "distances"],
    )

    ids = raw.get("ids", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    documents = raw.get("documents", [[]])[0]
    distances = raw.get("distances", [[]])[0]
    chunks = _load_chunks(chunks_path)

    results: list[RetrievalResult] = []
    for chunk_id, metadata, document, distance in zip(
        ids, metadatas, documents, distances
    ):
        if not isinstance(metadata, dict):
            metadata = {}
        results.append(
            _build_result(
                chunk_id=str(chunk_id),
                score=float(distance),
                metadata=metadata,
                document=document if isinstance(document, str) else None,
                chunk=chunks.get(str(chunk_id)),
                variant=variant,
            )
        )
    return results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_rag.search.retrieve",
        description="Retrieve timestamped video evidence from a vector index.",
    )
    p.add_argument("--video-id", required=True, help="Registered video_id.")
    p.add_argument("--question", required=True, help="Question to retrieve for.")
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
        help="Chunk window size used by the index (default: 30).",
    )
    p.add_argument(
        "--variant",
        default="transcript_ocr_vlm",
        choices=sorted(VARIANT_TO_FIELD.keys()),
        help="Embedding/search-text variant (default: transcript_ocr_vlm).",
    )
    p.add_argument(
        "--provider",
        required=True,
        choices=sorted(PROVIDERS.keys()),
        help="Embedding provider used for the query.",
    )
    p.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="Embedding model for the query provider.",
    )
    p.add_argument("--top-k", type=int, default=5, help="Number of hits to return.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        results = retrieve(
            video_id=args.video_id,
            question=args.question,
            data_dir=args.data_dir,
            chunk_seconds=args.chunk_seconds,
            variant=args.variant,
            provider=args.provider,
            model=args.model,
            top_k=args.top_k,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
        print(f"FAIL  {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1

    print(json.dumps([r.model_dump(mode="json") for r in results], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
