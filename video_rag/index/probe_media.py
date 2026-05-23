"""Stage 2: probe technical metadata for a registered video.

Reads a ``VideoManifest``, resolves the registered video path, inspects it with
ffprobe, and writes ``MediaMetadata`` for later indexing stages.

This stage does NOT extract audio, sample frames, transcribe, OCR, caption,
chunk, embed, or retrieve anything.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from video_rag.io_utils import read_json, write_json
from video_rag.schemas import MediaMetadata, VideoManifest

PathLike = str | Path


def _resolve_source_path(source_path: str) -> Path:
    p = Path(source_path)
    if p.is_absolute():
        return p
    return Path.cwd() / p


def _parse_fps(value: str | None) -> float | None:
    if not value:
        return None

    parts = value.split("/")
    if len(parts) != 2:
        return None

    try:
        numerator = float(parts[0])
        denominator = float(parts[1])
    except ValueError:
        return None

    if numerator <= 0 or denominator <= 0:
        return None

    return numerator / denominator


def _find_video_stream(streams: list[dict]) -> dict:
    for stream in streams:
        if stream.get("codec_type") == "video":
            return stream
    raise ValueError("ffprobe output did not contain a video stream")


def _metadata_from_ffprobe(video_id: str, raw: dict) -> MediaMetadata:
    try:
        duration_seconds = float(raw["format"]["duration"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError("ffprobe output is missing a valid format.duration") from e

    streams = raw.get("streams")
    if not isinstance(streams, list):
        raise ValueError("ffprobe output is missing a valid streams list")

    video_stream = _find_video_stream(streams)
    try:
        width = int(video_stream["width"])
        height = int(video_stream["height"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError("ffprobe video stream is missing valid width/height") from e

    fps = _parse_fps(video_stream.get("avg_frame_rate"))
    if fps is None:
        fps = _parse_fps(video_stream.get("r_frame_rate"))
    if fps is None:
        raise ValueError(
            "ffprobe video stream is missing a valid FPS value in "
            "avg_frame_rate or r_frame_rate"
        )

    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)

    return MediaMetadata(
        video_id=video_id,
        duration_seconds=duration_seconds,
        fps=fps,
        width=width,
        height=height,
        has_audio=has_audio,
    )


def probe_media(
    video_id: str,
    data_dir: PathLike = "data",
    overwrite: bool = False,
) -> MediaMetadata:
    """Probe media metadata for ``video_id`` and persist ``MediaMetadata``."""
    data_root = Path(data_dir)
    manifest_dir = data_root / "manifests" / video_id
    video_manifest_path = manifest_dir / "video_manifest.json"
    media_metadata_path = manifest_dir / "media_metadata.json"

    if not video_manifest_path.exists():
        raise FileNotFoundError(
            f"missing video manifest: {video_manifest_path}. Run Stage 1 first."
        )

    manifest = read_json(video_manifest_path, VideoManifest)
    video_path = _resolve_source_path(manifest.source_path)
    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError(f"registered video file not found: {video_path}")

    if media_metadata_path.exists() and not overwrite:
        raise FileExistsError(
            f"media metadata already exists: {media_metadata_path} "
            "(pass overwrite=True)"
        )

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_streams",
        "-of",
        "json",
        str(video_path),
    ]

    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else "no stderr output"
        raise RuntimeError(f"ffprobe failed while probing media:\n{stderr}") from e

    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as e:
        raise ValueError(f"ffprobe returned invalid JSON: {e.msg}") from e

    metadata = _metadata_from_ffprobe(video_id, raw)
    write_json(media_metadata_path, metadata)
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_rag.index.probe_media",
        description="Probe registered video metadata with ffprobe (Stage 2).",
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
        "--overwrite",
        action="store_true",
        help="Replace existing media metadata.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        probe_media(
            video_id=args.video_id,
            data_dir=args.data_dir,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError) as e:
        print(f"FAIL  {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1

    metadata_path = (
        Path(args.data_dir) / "manifests" / args.video_id / "media_metadata.json"
    )
    print(f"Wrote media metadata: {metadata_path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
