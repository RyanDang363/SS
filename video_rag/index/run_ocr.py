"""Stage 7: run OCR over sampled video frames.

Reads the Stage 5 frame manifest, runs OCR for each referenced frame, and
writes timestamped OCR results as JSONL.

This stage does NOT generate VLM captions, chunk, embed, retrieve, or answer
questions.
"""

from __future__ import annotations

import argparse
import sys
from numbers import Real
from pathlib import Path
from typing import Any

from video_rag.io_utils import read_jsonl, write_jsonl
from video_rag.schemas import FrameSample, OCRResult

PathLike = str | Path


def _resolve_frame_path(frame_path: str) -> Path:
    p = Path(frame_path)
    if p.is_absolute():
        return p
    return Path.cwd() / p


def _create_reader() -> Any:
    try:
        import easyocr
    except ImportError as e:
        raise RuntimeError(
            "EasyOCR is required to run Stage 7 OCR. Install easyocr first."
        ) from e
    return easyocr.Reader(["en"])


def _normalize_ocr_results(raw_results: list[Any]) -> tuple[str, float | None]:
    if not raw_results:
        return "", None

    text_parts: list[str] = []
    confidences: list[float] = []

    for result in raw_results:
        if not isinstance(result, (list, tuple)) or len(result) < 3:
            continue
        _bbox, text, confidence = result[:3]
        text_value = str(text).strip()
        if text_value:
            text_parts.append(text_value)
        if isinstance(confidence, Real):
            confidences.append(float(confidence))

    ocr_text = " ".join(text_parts).strip()
    avg_confidence = (
        sum(confidences) / len(confidences) if confidences else None
    )
    return ocr_text, avg_confidence


def _ocr_frame(reader: Any, frame_path: Path) -> tuple[str, float | None]:
    raw_results = reader.readtext(str(frame_path))
    return _normalize_ocr_results(raw_results)


def run_ocr(
    video_id: str,
    data_dir: PathLike = "data",
    overwrite: bool = False,
) -> list[OCRResult]:
    """Run OCR for sampled frames and return validated ``OCRResult`` records."""
    data_root = Path(data_dir)
    frame_manifest_path = data_root / "frames" / video_id / "frame_manifest.jsonl"
    ocr_output_path = data_root / "ocr" / f"{video_id}.jsonl"

    if not frame_manifest_path.exists():
        raise FileNotFoundError(
            f"missing frame manifest: {frame_manifest_path}. Run Stage 5 first."
        )

    frame_samples = list(read_jsonl(frame_manifest_path, FrameSample))

    for sample in frame_samples:
        resolved_frame_path = _resolve_frame_path(sample.frame_path)
        if not resolved_frame_path.exists() or not resolved_frame_path.is_file():
            raise FileNotFoundError(f"sampled frame file not found: {resolved_frame_path}")

    if ocr_output_path.exists() and not overwrite:
        raise FileExistsError(
            f"OCR output already exists: {ocr_output_path} (pass overwrite=True)"
        )

    reader = _create_reader()
    records: list[OCRResult] = []
    for sample in frame_samples:
        resolved_frame_path = _resolve_frame_path(sample.frame_path)
        ocr_text, confidence = _ocr_frame(reader, resolved_frame_path)
        records.append(
            OCRResult(
                video_id=sample.video_id,
                timestamp=sample.timestamp,
                frame_path=sample.frame_path,
                ocr_text=ocr_text,
                confidence=confidence,
            )
        )

    write_jsonl(ocr_output_path, records)
    return records


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_rag.index.run_ocr",
        description="Run OCR on sampled frames for a registered video (Stage 7).",
    )
    p.add_argument("--video-id", required=True, help="Registered video_id.")
    p.add_argument(
        "--data-dir",
        dest="data_dir",
        type=Path,
        default=Path("data"),
        help="Artifact root (default: data).",
    )
    p.add_argument("--overwrite", action="store_true", help="Replace existing OCR.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        run_ocr(
            video_id=args.video_id,
            data_dir=args.data_dir,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError) as e:
        print(f"FAIL  {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1

    output_path = Path(args.data_dir) / "ocr" / f"{args.video_id}.jsonl"
    print(f"Wrote OCR results: {output_path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
