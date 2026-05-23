"""Stage 5: sample frames from a registered video at fixed time intervals.

Reads ``VideoManifest`` and ``MediaMetadata`` produced by earlier stages,
computes a fixed-interval timestamp schedule, and writes timestamped JPEG
frames using OpenCV. Validates each record against ``FrameSample`` and writes
``data/frames/{video_id}/frame_manifest.jsonl``.

This stage does NOT generate thumbnails, OCR text, captions, or chunks.

A standalone C++ extractor (``cpp/frame_extract/``) is also provided as an
optional high-throughput alternative. It shares the same output contract and
can be swapped in without changing downstream stages. See
``cpp/frame_extract/README.md`` for build instructions.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

from video_rag.io_utils import read_json, write_jsonl
from video_rag.schemas import FrameSample, MediaMetadata, VideoManifest

PathLike = str | Path

DEFAULT_INTERVAL_SECONDS = 5
DEFAULT_JPEG_QUALITY = 85


def _coerce_positive_int_interval(interval_seconds: int | float) -> int:
    """V0 only supports positive-integer intervals; reject anything else."""
    if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, (int, float)):
        raise ValueError(
            f"interval_seconds must be a positive integer, got {interval_seconds!r}"
        )
    if isinstance(interval_seconds, float) and not interval_seconds.is_integer():
        raise ValueError(
            f"interval_seconds must be a whole number for V0, got {interval_seconds!r}"
        )
    value = int(interval_seconds)
    if value <= 0:
        raise ValueError(f"interval_seconds must be > 0, got {interval_seconds!r}")
    return value


def _compute_schedule(duration_seconds: float, interval: int) -> list[float]:
    """Return ``[0, interval, 2*interval, ...]`` strictly less than duration."""
    if duration_seconds <= 0:
        return []
    timestamps: list[float] = []
    t = 0
    while t < duration_seconds:
        timestamps.append(float(t))
        t += interval
    return timestamps


def _resolve_video_path(data_dir: Path, manifest: VideoManifest) -> Path:
    """Resolve ``manifest.source_path`` (repo-root-relative) against ``data_dir``.

    The contracts doc requires repo-root-relative paths, so during normal use
    ``data_dir`` is ``./data`` and ``data_dir.parent`` is the repo root. In
    tests ``data_dir`` is ``tmp_path/"data"`` and the same anchoring works.
    """
    return (data_dir.parent / manifest.source_path).resolve()


def _is_dir_nonempty(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def _to_repo_relative(absolute: Path, repo_root: Path) -> str:
    """Best-effort repo-root-relative POSIX path. Falls back to absolute POSIX."""
    try:
        rel = absolute.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return absolute.resolve().as_posix()
    return rel.as_posix()


def _read_image_dimensions(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None or image.size == 0:
        raise RuntimeError(f"failed to read extracted JPEG at {path}")
    h, w = image.shape[:2]
    return w, h


def _extract_frames_with_ffmpeg(
    video_path: Path,
    out_dir: Path,
    timestamps: list[float],
    jpeg_quality: int,
) -> list[dict]:
    """Extract timestamped frames with ffmpeg.

    OpenCV timestamp seeking can land on the wrong frames for some
    WebM/Matroska-style recordings with irregular packet timestamps. ffmpeg's
    timestamp seek is more reliable for those files, so prefer it when present.
    """
    records: list[dict] = []
    qscale = max(2, min(31, round((100 - jpeg_quality) / 4) + 2))

    for ts in timestamps:
        out_path = out_dir / f"frame_{int(round(ts)):06d}.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{ts:.6f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            str(qscale),
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.strip() if e.stderr else "no stderr output"
            raise RuntimeError(
                f"ffmpeg failed to extract frame at {ts}s from {video_path}:\n{stderr}"
            ) from e

        if not out_path.is_file():
            raise RuntimeError(f"ffmpeg did not write expected JPEG to {out_path}")
        width, height = _read_image_dimensions(out_path)
        records.append(
            {
                "timestamp": ts,
                "frame_path": str(out_path),
                "width": width,
                "height": height,
            }
        )

    return records


def _extract_frames_with_opencv(
    video_path: Path,
    out_dir: Path,
    timestamps: list[float],
    jpeg_quality: int,
) -> list[dict]:
    """Open ``video_path``, seek to each timestamp, write a JPEG, return records.

    Isolated into its own function so tests can monkeypatch it without
    requiring a real video file or OpenCV installation.

    Returns:
        List of dicts with keys: ``timestamp``, ``frame_path`` (absolute str),
        ``width``, ``height``.

    Raises:
        RuntimeError: if the video cannot be opened, a frame cannot be decoded,
            or a JPEG cannot be written.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    records: list[dict] = []
    try:
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                raise RuntimeError(
                    f"failed to decode frame at {ts}s from {video_path}"
                )
            out_path = out_dir / f"frame_{int(round(ts)):06d}.jpg"
            if not cv2.imwrite(str(out_path), frame, encode_params):
                raise RuntimeError(f"failed to write JPEG to {out_path}")
            h, w = frame.shape[:2]
            records.append(
                {
                    "timestamp": ts,
                    "frame_path": str(out_path),
                    "width": w,
                    "height": h,
                }
            )
    finally:
        cap.release()

    return records


def _extract_frames(
    video_path: Path,
    out_dir: Path,
    timestamps: list[float],
    jpeg_quality: int,
) -> list[dict]:
    """Extract timestamped frames, preferring ffmpeg for reliable seeking."""
    if shutil.which("ffmpeg") is not None:
        return _extract_frames_with_ffmpeg(
            video_path=video_path,
            out_dir=out_dir,
            timestamps=timestamps,
            jpeg_quality=jpeg_quality,
        )
    return _extract_frames_with_opencv(
        video_path=video_path,
        out_dir=out_dir,
        timestamps=timestamps,
        jpeg_quality=jpeg_quality,
    )


def sample_frames(
    video_id: str,
    data_dir: PathLike = "data",
    interval_seconds: int | float = DEFAULT_INTERVAL_SECONDS,
    overwrite: bool = False,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> list[FrameSample]:
    """Sample frames at a fixed interval. Returns the persisted ``FrameSample`` records.

    Raises:
        ValueError: ``interval_seconds`` is not a positive integer or ``video_id``
            is empty.
        FileNotFoundError: required input artifact or video file is missing.
        FileExistsError: prior frame outputs exist and ``overwrite`` is ``False``.
        RuntimeError: OpenCV failed to decode or write a frame.
    """
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("video_id must be a non-empty string")

    interval = _coerce_positive_int_interval(interval_seconds)

    if not isinstance(jpeg_quality, int) or not (1 <= jpeg_quality <= 100):
        raise ValueError(f"jpeg_quality must be an int in [1, 100], got {jpeg_quality!r}")

    data_root = Path(data_dir)
    repo_root = data_root.resolve().parent

    manifest_path = data_root / "manifests" / video_id / "video_manifest.json"
    metadata_path = data_root / "manifests" / video_id / "media_metadata.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"video_manifest.json not found for video_id={video_id!r} at "
            f"{manifest_path}. Run Stage 1 (register_video) first."
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"media_metadata.json not found for video_id={video_id!r} at "
            f"{metadata_path}. Run Stage 2 (probe_media) first."
        )

    manifest = read_json(manifest_path, VideoManifest)
    metadata = read_json(metadata_path, MediaMetadata)

    if manifest.video_id != video_id:
        raise ValueError(
            f"video_id mismatch: argument {video_id!r} != manifest "
            f"{manifest.video_id!r}"
        )
    if metadata.video_id != video_id:
        raise ValueError(
            f"video_id mismatch: argument {video_id!r} != media_metadata "
            f"{metadata.video_id!r}"
        )

    video_path = _resolve_video_path(data_root, manifest)
    if not video_path.is_file():
        raise FileNotFoundError(
            f"registered video file not found at {video_path} "
            f"(referenced by {manifest_path})"
        )

    out_dir = data_root / "frames" / video_id
    manifest_out = out_dir / "frame_manifest.jsonl"

    if not overwrite and (manifest_out.exists() or _is_dir_nonempty(out_dir)):
        raise FileExistsError(
            f"frame outputs already exist at {out_dir} (pass overwrite=True to replace)"
        )
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamps = _compute_schedule(metadata.duration_seconds, interval)

    if not timestamps:
        write_jsonl(manifest_out, [])
        return []

    records = _extract_frames(video_path, out_dir.resolve(), timestamps, jpeg_quality)

    samples: list[FrameSample] = []
    for rec in records:
        frame_path_rel = _to_repo_relative(Path(rec["frame_path"]), repo_root)
        samples.append(
            FrameSample(
                video_id=video_id,
                timestamp=rec["timestamp"],
                frame_path=frame_path_rel,
                width=rec["width"],
                height=rec["height"],
                sampling_method="fixed_interval",
            )
        )

    write_jsonl(manifest_out, samples)
    return samples


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_rag.index.sample_frames",
        description="Sample frames from a registered video at fixed intervals (Stage 5).",
    )
    p.add_argument("--video-id", dest="video_id", required=True, help="Registered video_id.")
    p.add_argument(
        "--interval-seconds",
        dest="interval_seconds",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help=f"Sampling interval in whole seconds (default: {DEFAULT_INTERVAL_SECONDS}).",
    )
    p.add_argument(
        "--data-dir",
        dest="data_dir",
        type=Path,
        default=Path("data"),
        help="Artifact root (default: data).",
    )
    p.add_argument("--overwrite", action="store_true", help="Replace existing frame outputs.")
    p.add_argument(
        "--quality",
        dest="jpeg_quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help=f"JPEG quality 1-100 (default: {DEFAULT_JPEG_QUALITY}).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        samples = sample_frames(
            video_id=args.video_id,
            data_dir=args.data_dir,
            interval_seconds=args.interval_seconds,
            overwrite=args.overwrite,
            jpeg_quality=args.jpeg_quality,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError) as e:
        print(f"FAIL  {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1

    manifest_out = (
        Path(args.data_dir) / "frames" / args.video_id / "frame_manifest.jsonl"
    )
    print(f"Sampled {len(samples)} frame(s) for {args.video_id}")
    print(f"  manifest: {manifest_out.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
