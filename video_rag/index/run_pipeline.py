"""Run the full indexing pipeline for a single video.

Calls every stage's public function in order: register -> probe -> audio ->
transcribe -> frames -> OCR -> captions. Each stage is invoked with
``overwrite=True``. Failures are not caught here; the first exception
aborts the pipeline and surfaces a normal Python traceback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

from video_rag.index.caption_frames import caption_frames
from video_rag.index.extract_audio import extract_audio
from video_rag.index.probe_media import probe_media
from video_rag.index.register_video import register_video
from video_rag.index.run_ocr import run_ocr
from video_rag.index.sample_frames import sample_frames
from video_rag.index.transcribe_audio import transcribe_audio
from video_rag.index.transcription_providers import PROVIDERS

PathLike = str | Path
Mode = Literal["copy", "symlink"]

_TOTAL_STAGES = 7


def _step(n: int, name: str, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"[{n}/{_TOTAL_STAGES}] {name}{suffix}", flush=True)


def run_pipeline(
    video_path: PathLike | None = None,
    video_id: str | None = None,
    *,
    title: str | None = None,
    mode: Mode = "copy",
    transcription_provider: str = "openai",
    language: str | None = None,
    interval_seconds: int | float = 5,
    frames_per_caption: int = 3,
    data_dir: PathLike = "data",
) -> str:
    """Run all indexing stages for one video. Returns the final ``video_id``.

    Exactly one of ``video_path`` or ``video_id`` must be supplied. If
    ``video_path`` is given, Stage 1 runs first and its resulting
    ``video_id`` is used downstream. If only ``video_id`` is given, the
    video must already be registered.
    """
    if (video_path is None) == (video_id is None):
        raise ValueError("pass exactly one of video_path or video_id")

    data_root = Path(data_dir)

    if video_path is not None:
        _step(1, "register_video", f"video={Path(video_path).as_posix()}")
        manifest = register_video(
            video_path=video_path,
            title=title,
            video_id=video_id,
            mode=mode,
            overwrite=True,
            data_dir=data_dir,
        )
        resolved_id = manifest.video_id
    else:
        assert video_id is not None
        manifest_path = data_root / "manifests" / video_id / "video_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"missing video manifest: {manifest_path}. "
                "Register the video first (Stage 1) or pass --video."
            )
        _step(1, "register_video", f"video_id={video_id} (already registered, skipping)")
        resolved_id = video_id

    _step(2, "probe_media", f"video_id={resolved_id}")
    probe_media(resolved_id, data_dir=data_dir, overwrite=True)

    _step(3, "extract_audio", f"video_id={resolved_id}")
    extract_audio(resolved_id, data_dir=data_dir, overwrite=True)

    _step(4, "transcribe_audio", f"provider={transcription_provider}")
    transcribe_audio(
        video_id=resolved_id,
        provider=transcription_provider,
        language=language,
        overwrite=True,
        data_dir=data_dir,
    )

    _step(5, "sample_frames", f"interval={interval_seconds}s")
    sample_frames(
        video_id=resolved_id,
        data_dir=data_dir,
        interval_seconds=interval_seconds,
        overwrite=True,
    )

    _step(6, "run_ocr", f"video_id={resolved_id}")
    run_ocr(resolved_id, data_dir=data_dir, overwrite=True)

    _step(7, "caption_frames", f"frames_per_caption={frames_per_caption}")
    caption_frames(
        video_id=resolved_id,
        data_dir=data_dir,
        frames_per_caption=frames_per_caption,
        overwrite=True,
    )

    print(f"OK  pipeline complete for video_id={resolved_id}")
    return resolved_id


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_rag.index.run_pipeline",
        description="Run the full RAGGERS indexing pipeline for one video.",
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Path to a local video file (runs Stage 1 first).",
    )
    source.add_argument(
        "--video-id",
        dest="video_id",
        default=None,
        help="Existing registered video_id (skips Stage 1).",
    )

    p.add_argument("--title", default=None, help="Title forwarded to Stage 1.")
    p.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="copy",
        help="How Stage 1 places the video in data/videos/.",
    )
    p.add_argument(
        "--provider",
        dest="transcription_provider",
        choices=sorted(PROVIDERS.keys()),
        default="openai",
        help="Transcription provider (default: openai).",
    )
    p.add_argument("--language", default=None, help="Language hint for transcription.")
    p.add_argument(
        "--interval-seconds",
        dest="interval_seconds",
        type=int,
        default=5,
        help="Frame sampling interval in seconds (default: 5).",
    )
    p.add_argument(
        "--frames-per-caption",
        dest="frames_per_caption",
        type=int,
        default=3,
        help="Frames per VLM caption group (default: 3).",
    )
    p.add_argument(
        "--data-dir",
        dest="data_dir",
        type=Path,
        default=Path("data"),
        help="Artifact root (default: data).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_pipeline(
        video_path=args.video,
        video_id=args.video_id,
        title=args.title,
        mode=args.mode,
        transcription_provider=args.transcription_provider,
        language=args.language,
        interval_seconds=args.interval_seconds,
        frames_per_caption=args.frames_per_caption,
        data_dir=args.data_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
