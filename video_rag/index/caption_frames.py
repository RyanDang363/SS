"""Stage 8: generate generic visual captions for sampled frame groups.

Reads the Stage 5 frame manifest, groups sampled frames into small windows, and
writes VLM-generated indexing-time captions as JSONL.

This stage does NOT OCR, chunk, embed, retrieve, or answer questions.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import sys
from pathlib import Path

from video_rag.io_utils import read_jsonl, write_jsonl
from video_rag.schemas import FrameSample, VLMCaption

PathLike = str | Path

DEFAULT_MODEL = "gpt-4o-mini"
CAPTION_PROMPT = (
    "Describe visible people, slides, diagrams, equations, board writing, "
    "objects, actions, and readable text. Do not guess beyond the images. "
    "Keep the caption to 1-3 concise sentences."
)


def _resolve_frame_path(frame_path: str) -> Path:
    p = Path(frame_path)
    if p.is_absolute():
        return p
    return Path.cwd() / p


def _group_frames(
    frame_samples: list[FrameSample],
    frames_per_caption: int,
) -> list[list[FrameSample]]:
    return [
        frame_samples[i : i + frames_per_caption]
        for i in range(0, len(frame_samples), frames_per_caption)
    ]


def _image_data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _caption_frame_group(
    frame_paths: list[Path],
    model: str = DEFAULT_MODEL,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "The openai package is required to run Stage 8 captioning."
        ) from e

    content = [{"type": "input_text", "text": CAPTION_PROMPT}]
    content.extend(
        {
            "type": "input_image",
            "image_url": _image_data_url(frame_path),
        }
        for frame_path in frame_paths
    )

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
    )
    return response.output_text


def _normalize_caption(caption: str) -> str:
    normalized = " ".join(caption.split())
    if not normalized:
        raise ValueError("VLM returned an empty caption")
    return normalized


def caption_frames(
    video_id: str,
    data_dir: PathLike = "data",
    frames_per_caption: int = 3,
    overwrite: bool = False,
) -> list[VLMCaption]:
    """Generate generic visual captions for sampled frame groups."""
    if (
        not isinstance(frames_per_caption, int)
        or isinstance(frames_per_caption, bool)
        or frames_per_caption <= 0
    ):
        raise ValueError("frames_per_caption must be a positive integer")

    data_root = Path(data_dir)
    frame_manifest_path = data_root / "frames" / video_id / "frame_manifest.jsonl"
    caption_output_path = data_root / "captions" / f"{video_id}.jsonl"

    if not frame_manifest_path.exists():
        raise FileNotFoundError(
            f"missing frame manifest: {frame_manifest_path}. Run Stage 5 first."
        )

    frame_samples = list(read_jsonl(frame_manifest_path, FrameSample))

    for sample in frame_samples:
        resolved_frame_path = _resolve_frame_path(sample.frame_path)
        if not resolved_frame_path.exists() or not resolved_frame_path.is_file():
            raise FileNotFoundError(f"sampled frame file not found: {resolved_frame_path}")

    if caption_output_path.exists() and not overwrite:
        raise FileExistsError(
            f"caption output already exists: {caption_output_path} "
            "(pass overwrite=True)"
        )

    records: list[VLMCaption] = []
    for group in _group_frames(frame_samples, frames_per_caption):
        resolved_paths = [_resolve_frame_path(sample.frame_path) for sample in group]
        caption = _normalize_caption(
            _caption_frame_group(resolved_paths, model=DEFAULT_MODEL)
        )
        records.append(
            VLMCaption(
                video_id=video_id,
                start_time=group[0].timestamp,
                end_time=group[-1].timestamp,
                frame_paths=[sample.frame_path for sample in group],
                caption=caption,
                caption_type="generic",
                model=DEFAULT_MODEL,
            )
        )

    write_jsonl(caption_output_path, records)
    return records


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_rag.index.caption_frames",
        description="Generate generic VLM captions for sampled frames (Stage 8).",
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
        "--frames-per-caption",
        type=int,
        default=3,
        help="Number of sampled frames per caption window (default: 3).",
    )
    p.add_argument("--overwrite", action="store_true", help="Replace existing captions.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        caption_frames(
            video_id=args.video_id,
            data_dir=args.data_dir,
            frames_per_caption=args.frames_per_caption,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError) as e:
        print(f"FAIL  {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1

    output_path = Path(args.data_dir) / "captions" / f"{args.video_id}.jsonl"
    print(f"Wrote frame captions: {output_path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
