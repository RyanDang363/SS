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


def _parse_positive_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _find_video_stream(streams: list[dict]) -> dict:
    for stream in streams:
        if stream.get("codec_type") == "video":
            return stream
    raise ValueError("ffprobe output did not contain a video stream")


def _duration_from_ffprobe_output(raw: dict) -> float | None:
    duration = _parse_positive_float(raw.get("format", {}).get("duration"))
    if duration is not None:
        return duration

    streams = raw.get("streams")
    if not isinstance(streams, list):
        return None

    for stream in streams:
        duration = _parse_positive_float(stream.get("duration"))
        if duration is not None:
            return duration

    return None


def _metadata_from_ffprobe(video_id: str, raw: dict) -> MediaMetadata:
    duration_seconds = _duration_from_ffprobe_output(raw)
    if duration_seconds is None:
        raise ValueError("ffprobe output is missing a valid duration")

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


def _duration_from_packet_timestamps(raw: dict) -> float | None:
    packets = raw.get("packets")
    if not isinstance(packets, list):
        return None

    latest_end: float | None = None
    for packet in packets:
        if not isinstance(packet, dict):
            continue

        timestamp = _parse_positive_float(packet.get("pts_time"))
        if timestamp is None:
            timestamp = _parse_positive_float(packet.get("dts_time"))
        if timestamp is None:
            continue

        duration = _parse_positive_float(packet.get("duration_time")) or 0.0
        packet_end = timestamp + duration
        latest_end = packet_end if latest_end is None else max(latest_end, packet_end)

    return latest_end


def _probe_duration_from_packets(video_path: Path) -> float | None:
    """Best-effort duration fallback for streams with no container duration."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_packets",
        "-show_entries",
        "packet=pts_time,dts_time,duration_time",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    raw = json.loads(completed.stdout)
    return _duration_from_packet_timestamps(raw)


def _probe_duration_by_counting_frames(video_path: Path) -> float | None:
    """Last-resort duration fallback for streams without usable timestamps."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    raw = json.loads(completed.stdout)
    streams = raw.get("streams")
    if not isinstance(streams, list) or not streams:
        return None

    stream = streams[0]
    try:
        frame_count = int(stream["nb_read_frames"])
    except (KeyError, TypeError, ValueError):
        return None

    fps = _parse_fps(stream.get("avg_frame_rate"))
    if fps is None:
        fps = _parse_fps(stream.get("r_frame_rate"))
    if fps is None:
        return None

    duration = frame_count / fps
    return duration if duration > 0 else None


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

    if _duration_from_ffprobe_output(raw) is None:
        try:
            fallback_duration = _probe_duration_from_packets(video_path)
            if fallback_duration is None:
                fallback_duration = _probe_duration_by_counting_frames(video_path)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            raise ValueError("ffprobe output is missing a valid duration") from e
        if fallback_duration is not None:
            raw.setdefault("format", {})["duration"] = str(fallback_duration)

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
