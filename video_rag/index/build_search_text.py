"""Stage 10: build searchable text variants from Stage 9 chunks.

Reads timestamped chunk records and writes enriched chunk records with
retrieval-oriented text fields. This stage does NOT realign timestamps, embed,
store vectors, retrieve, rerank, or answer questions.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from video_rag.io_utils import read_jsonl, write_jsonl
from video_rag.schemas import Chunk

PathLike = str | Path


def _format_chunk_seconds(chunk_seconds: int | float) -> str:
    value = float(chunk_seconds)
    if value <= 0:
        raise ValueError("chunk_seconds must be greater than 0")
    return f"{value:g}"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _format_sections(sections: list[tuple[str, str]]) -> str:
    non_empty = [(label, text) for label, text in sections if text]
    return "\n\n".join(f"{label}:\n{text}" for label, text in non_empty)


def _enrich_chunk(chunk: Chunk) -> Chunk:
    transcript = _clean_text(chunk.transcript_text)
    ocr = _clean_text(chunk.ocr_text)
    vlm = _clean_text(chunk.vlm_caption)

    transcript_only = _format_sections([("Transcript", transcript)])
    transcript_ocr = _format_sections(
        [("Transcript", transcript), ("On-screen text", ocr)]
    )
    transcript_vlm = _format_sections(
        [("Transcript", transcript), ("Visual caption", vlm)]
    )
    all_text = _format_sections(
        [
            ("Transcript", transcript),
            ("On-screen text", ocr),
            ("Visual caption", vlm),
        ]
    )

    data = chunk.model_dump(mode="json")
    data.update(
        {
            "combined_text_transcript_only": transcript_only,
            "combined_text_transcript_ocr": transcript_ocr,
            "combined_text_transcript_vlm": transcript_vlm,
            "combined_text_all": all_text,
            "combined_text": all_text,
        }
    )
    return Chunk.model_validate(data)


def build_search_text(
    video_id: str,
    data_dir: PathLike = "data",
    chunk_seconds: int | float = 30,
    overwrite: bool = False,
) -> list[Chunk]:
    """Add searchable text variants to Stage 9 chunk records."""
    chunk_seconds_label = _format_chunk_seconds(chunk_seconds)
    data_root = Path(data_dir)
    chunks_dir = data_root / "chunks"
    input_path = chunks_dir / f"{video_id}_{chunk_seconds_label}s.jsonl"
    output_path = chunks_dir / f"{video_id}_{chunk_seconds_label}s_enriched.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(
            f"missing Stage 9 chunk file: {input_path}. Run Stage 9 first."
        )

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"enriched chunk output already exists: {output_path} "
            "(pass overwrite=True)"
        )

    chunks_dir.mkdir(parents=True, exist_ok=True)
    enriched = [_enrich_chunk(chunk) for chunk in read_jsonl(input_path, Chunk)]
    write_jsonl(output_path, enriched)
    return enriched


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_rag.index.build_search_text",
        description="Build searchable text variants from Stage 9 chunks (Stage 10).",
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
        "--overwrite",
        action="store_true",
        help="Replace existing enriched chunks.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        build_search_text(
            video_id=args.video_id,
            data_dir=args.data_dir,
            chunk_seconds=args.chunk_seconds,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as e:
        print(f"FAIL  {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1

    chunk_seconds_label = _format_chunk_seconds(args.chunk_seconds)
    output_path = (
        Path(args.data_dir)
        / "chunks"
        / f"{args.video_id}_{chunk_seconds_label}s_enriched.jsonl"
    )
    print(f"Wrote enriched chunks: {output_path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
