"""Stage 9: build fixed-time chunks by aligning modalities on the video timeline.

Reads the Stage 2 media metadata and the Stage 4 transcript (both required), plus
optional Stage 5 frames, Stage 7 OCR, and Stage 8 VLM captions, and produces
fixed-time ``Chunk`` records. Each chunk covers one fixed window of the video and
gathers the transcript / OCR / caption / frame evidence overlapping that window.

This is custom timestamp alignment across modalities. It deliberately does NOT use
LangChain or any generic text splitter. It also does not build searchable text,
embed, store vectors, retrieve, rerank, answer questions, or evaluate -- those are
later stages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel

from video_rag.io_utils import read_json, read_jsonl, write_jsonl
from video_rag.schemas import (
    Chunk,
    FrameSample,
    MediaMetadata,
    OCRResult,
    TranscriptSegment,
    VLMCaption,
)

PathLike = str | Path

T = TypeVar("T", bound=BaseModel)


def _fmt(value: float) -> str:
    """Format a seconds value for filenames, dropping a trailing ``.0``."""
    if value == int(value):
        return str(int(value))
    return str(value)


def _output_path(
    data_root: Path,
    video_id: str,
    chunk_seconds: float,
    overlap_seconds: float,
) -> Path:
    """Resolve the chunk output path, adding an overlap suffix when needed."""
    name = f"{video_id}_{_fmt(chunk_seconds)}s"
    if overlap_seconds > 0:
        name += f"_overlap{_fmt(overlap_seconds)}s"
    return data_root / "chunks" / f"{name}.jsonl"


def _read_optional(path: Path, model: Type[T]) -> list[T]:
    """Read JSONL records, returning ``[]`` when the file does not exist."""
    if not path.exists():
        return []
    return list(read_jsonl(path, model))


def _build_windows(
    duration_seconds: float,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[tuple[float, float]]:
    """Build fixed-time windows over ``[0, duration_seconds]``.

    Windows start at ``0, stride, 2*stride, ...`` where ``stride = chunk_seconds
    - overlap_seconds``, while the start is still inside the video. The final
    window is clamped so it ends exactly at ``duration_seconds``.
    """
    stride = chunk_seconds - overlap_seconds
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_seconds:
        end = min(start + chunk_seconds, duration_seconds)
        windows.append((start, end))
        start += stride
    return windows


def build_chunks(
    video_id: str,
    data_dir: PathLike = "data",
    chunk_seconds: float = 30,
    overlap_seconds: float = 0,
    overwrite: bool = False,
) -> list[Chunk]:
    """Build fixed-time chunks for a video and return validated ``Chunk`` records."""
    chunk_seconds = float(chunk_seconds)
    overlap_seconds = float(overlap_seconds)

    if chunk_seconds <= 0:
        raise ValueError(f"chunk_seconds must be > 0, got {chunk_seconds}")
    if overlap_seconds < 0:
        raise ValueError(f"overlap_seconds must be >= 0, got {overlap_seconds}")
    if overlap_seconds >= chunk_seconds:
        raise ValueError(
            f"overlap_seconds ({overlap_seconds}) must be less than "
            f"chunk_seconds ({chunk_seconds})"
        )

    data_root = Path(data_dir)
    metadata_path = data_root / "manifests" / video_id / "media_metadata.json"
    transcript_path = data_root / "transcripts" / f"{video_id}.jsonl"
    frames_path = data_root / "frames" / video_id / "frame_manifest.jsonl"
    ocr_path = data_root / "ocr" / f"{video_id}.jsonl"
    captions_path = data_root / "captions" / f"{video_id}.jsonl"
    output_path = _output_path(data_root, video_id, chunk_seconds, overlap_seconds)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"missing media metadata: {metadata_path}. Run Stage 2 first."
        )
    if not transcript_path.exists():
        raise FileNotFoundError(
            f"missing transcript: {transcript_path}. Run Stage 4 first."
        )

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"chunk output already exists: {output_path} (pass overwrite=True)"
        )

    metadata = read_json(metadata_path, MediaMetadata)
    transcript = list(read_jsonl(transcript_path, TranscriptSegment))
    frames = _read_optional(frames_path, FrameSample)
    ocr = _read_optional(ocr_path, OCRResult)
    captions = _read_optional(captions_path, VLMCaption)

    duration = metadata.duration_seconds
    windows = _build_windows(duration, chunk_seconds, overlap_seconds)

    records: list[Chunk] = []
    for index, (start, end) in enumerate(windows):
        is_last = index == len(windows) - 1

        # Time-range overlap for ranged modalities.
        segs = [s for s in transcript if s.start_time < end and s.end_time > start]
        caps = [c for c in captions if c.start_time < end and c.end_time > start]

        # Point-in-window for instantaneous modalities. The final window also
        # includes records sitting exactly at the video duration.
        def _in_window(ts: float) -> bool:
            if start <= ts < end:
                return True
            return is_last and ts == end

        frs = [f for f in frames if _in_window(f.timestamp)]
        ocrs = [o for o in ocr if _in_window(o.timestamp)]

        segs.sort(key=lambda s: s.start_time)
        caps.sort(key=lambda c: c.start_time)
        frs.sort(key=lambda f: f.timestamp)
        ocrs.sort(key=lambda o: o.timestamp)

        transcript_text = " ".join(s.text.strip() for s in segs if s.text.strip())
        ocr_text = " ".join(o.ocr_text.strip() for o in ocrs if o.ocr_text.strip())
        vlm_caption = " ".join(c.caption.strip() for c in caps if c.caption.strip())
        frame_paths = [f.frame_path for f in frs]

        records.append(
            Chunk(
                chunk_id=f"{video_id}_chunk_{index:04d}",
                video_id=video_id,
                chunk_index=index,
                start_time=start,
                end_time=end,
                transcript_text=transcript_text,
                ocr_text=ocr_text,
                vlm_caption=vlm_caption,
                frame_paths=frame_paths,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
                chunking_strategy="fixed",
            )
        )

    write_jsonl(output_path, records)
    return records


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_rag.index.build_chunks",
        description="Build fixed-time multimodal chunks for a video (Stage 9).",
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
        help="Fixed chunk window length in seconds (default: 30).",
    )
    p.add_argument(
        "--overlap-seconds",
        type=float,
        default=0,
        help="Overlap between consecutive chunks in seconds (default: 0).",
    )
    p.add_argument("--overwrite", action="store_true", help="Replace existing chunks.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        build_chunks(
            video_id=args.video_id,
            data_dir=args.data_dir,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError) as e:
        print(f"FAIL  {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1

    output_path = _output_path(
        Path(args.data_dir),
        args.video_id,
        float(args.chunk_seconds),
        float(args.overlap_seconds),
    )
    print(f"Wrote chunks: {output_path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
