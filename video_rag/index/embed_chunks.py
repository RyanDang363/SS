"""Stage 11: embed enriched chunks into vectors.

Reads the Stage 10 enriched chunks from
``data/chunks/{video_id}_{chunk_seconds}s_enriched.jsonl``, selects one
search-text variant, embeds that text through an :class:`EmbeddingProvider`,
and writes ``EmbeddingRecord`` JSONL to
``data/embeddings/{video_id}_{chunk_seconds}s_{variant}.jsonl``.

This stage does not build searchable text (Stage 10), store vectors in a vector
DB / Chroma (Stage 12), retrieve, rerank, answer questions, or evaluate. It only
turns enriched chunk text into embedding records.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from video_rag.index.embedding_providers import (
    PROVIDERS,
    EmbeddingProvider,
    get_provider,
)
from video_rag.io_utils import read_jsonl, write_jsonl
from video_rag.schemas import EmbeddingRecord

logger = logging.getLogger(__name__)

PathLike = str | Path

# Maps a variant to the enriched-chunk field holding its search text.
VARIANT_FIELD: dict[str, str] = {
    "transcript_only": "combined_text_transcript_only",
    "transcript_ocr": "combined_text_transcript_ocr",
    "transcript_vlm": "combined_text_transcript_vlm",
    "transcript_ocr_vlm": "combined_text_all",
}


class _EnrichedChunkView(BaseModel):
    """Read-only view of a Stage 10 enriched chunk.

    Only the fields Stage 11 needs are declared. ``extra="ignore"`` tolerates the
    many other fields a real enriched chunk carries. The ``combined_text_*``
    fields default to ``None`` (absent) so the stage can distinguish a field that
    Stage 10 never produced (→ hard error) from one that is present but empty
    (→ skipped).
    """

    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    combined_text_transcript_only: str | None = None
    combined_text_transcript_ocr: str | None = None
    combined_text_transcript_vlm: str | None = None
    combined_text_all: str | None = None


def _fmt(value: float) -> str:
    """Format a seconds value for filenames, dropping a trailing ``.0``."""
    if value == int(value):
        return str(int(value))
    return str(value)


def _input_path(data_root: Path, video_id: str, chunk_seconds: float) -> Path:
    return data_root / "chunks" / f"{video_id}_{_fmt(chunk_seconds)}s_enriched.jsonl"


def _output_path(
    data_root: Path, video_id: str, chunk_seconds: float, variant: str
) -> Path:
    name = f"{video_id}_{_fmt(chunk_seconds)}s_{variant}"
    return data_root / "embeddings" / f"{name}.jsonl"


def _resolve_provider(
    provider: EmbeddingProvider | str, *, model: str
) -> EmbeddingProvider:
    if isinstance(provider, str):
        return get_provider(provider, model=model)
    return provider


def _atomic_write_jsonl(path: Path, records: list[EmbeddingRecord]) -> None:
    tmp = path.parent / (path.name + ".tmp")
    try:
        write_jsonl(tmp, records)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def embed_chunks(
    video_id: str,
    data_dir: PathLike = "data",
    chunk_seconds: float = 30,
    variant: str = "transcript_ocr_vlm",
    provider: EmbeddingProvider | str = "openai",
    overwrite: bool = False,
    *,
    model: str = "text-embedding-3-small",
    batch_size: int = 64,
) -> list[EmbeddingRecord]:
    """Embed one search-text variant of a video's enriched chunks.

    Returns the persisted ``EmbeddingRecord`` list (one per non-empty chunk, in
    input order). Chunks whose selected text is empty/whitespace are skipped with
    a warning.
    """
    if not video_id or not video_id.strip():
        raise ValueError("video_id must be non-empty")
    if variant not in VARIANT_FIELD:
        valid = ", ".join(sorted(VARIANT_FIELD))
        raise ValueError(f"unknown variant {variant!r} (valid: {valid})")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    chunk_seconds = float(chunk_seconds)
    data_root = Path(data_dir)
    field = VARIANT_FIELD[variant]
    input_path = _input_path(data_root, video_id, chunk_seconds)
    output_path = _output_path(data_root, video_id, chunk_seconds, variant)

    if not input_path.exists():
        raise FileNotFoundError(
            f"missing enriched chunks: {input_path}. Run Stage 10 first."
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"embedding output already exists: {output_path} (pass overwrite=True)"
        )

    chunks = list(read_jsonl(input_path, _EnrichedChunkView))
    if not chunks:
        raise ValueError(f"no enriched chunks found in {input_path}")

    # Select text per chunk, failing if the variant field is absent and skipping
    # chunks whose text is empty after whitespace cleanup.
    selected: list[tuple[_EnrichedChunkView, str]] = []
    for chunk in chunks:
        value = getattr(chunk, field)
        if value is None:
            raise ValueError(
                f"enriched chunks are missing field {field!r} required for "
                f"variant {variant!r}: {input_path}"
            )
        text = value.strip()
        if not text:
            logger.warning(
                "skipping chunk %s: empty %s text for variant %r",
                chunk.chunk_id,
                field,
                variant,
            )
            continue
        selected.append((chunk, text))

    resolved = _resolve_provider(provider, model=model)

    # Embed in batches, preserving input order.
    vectors: list[list[float]] = []
    texts = [text for _, text in selected]
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(resolved.embed_texts(batch))

    if len(vectors) != len(selected):
        raise RuntimeError(
            f"provider returned {len(vectors)} vectors for {len(selected)} texts"
        )

    records: list[EmbeddingRecord] = []
    for (chunk, _), vector in zip(selected, vectors):
        records.append(
            EmbeddingRecord(
                chunk_id=chunk.chunk_id,
                video_id=chunk.video_id,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                embedding_model=resolved.model,
                embedding_provider=resolved.name,
                embedding_variant=variant,
                vector=list(vector),
                vector_dim=len(vector),
            )
        )

    _atomic_write_jsonl(output_path, records)
    return records


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_rag.index.embed_chunks",
        description="Embed enriched chunks into vectors (Stage 11).",
    )
    p.add_argument("--video-id", dest="video_id", required=True, help="Registered video_id.")
    p.add_argument(
        "--variant",
        required=True,
        choices=sorted(VARIANT_FIELD.keys()),
        help="Which enriched search-text variant to embed.",
    )
    p.add_argument(
        "--provider",
        required=True,
        choices=sorted(PROVIDERS.keys()),
        help="Embedding provider (mock is for testing only).",
    )
    p.add_argument(
        "--chunk-seconds",
        type=float,
        default=30,
        help="Chunk window length used by Stage 9/10 (default: 30).",
    )
    p.add_argument(
        "--data-dir",
        dest="data_dir",
        type=Path,
        default=Path("data"),
        help="Artifact root (default: data).",
    )
    p.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="Embedding model for the openai provider (default: text-embedding-3-small).",
    )
    p.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=64,
        help="Number of texts per provider request (default: 64).",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Replace existing embedding file."
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        records = embed_chunks(
            video_id=args.video_id,
            data_dir=args.data_dir,
            chunk_seconds=args.chunk_seconds,
            variant=args.variant,
            provider=args.provider,
            overwrite=args.overwrite,
            model=args.model,
            batch_size=args.batch_size,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError) as e:
        print(f"FAIL  {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1

    output_path = _output_path(
        Path(args.data_dir), args.video_id, float(args.chunk_seconds), args.variant
    )
    print(f"Wrote embeddings: {output_path.as_posix()}")
    print(f"  video_id: {args.video_id}")
    print(f"  variant:  {args.variant}")
    print(f"  provider: {args.provider}")
    print(f"  records:  {len(records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
