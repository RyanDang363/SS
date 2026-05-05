"""Stage 5: sample frames from a registered video at fixed time intervals.

Reads ``VideoManifest`` and ``MediaMetadata`` produced by earlier stages,
computes a fixed-interval timestamp schedule, and delegates the heavy work
(decoding + JPEG encoding) to a small C++/OpenCV binary
(``raggers_frame_extract``). Validates each emitted record against
``FrameSample`` and writes ``data/frames/{video_id}/frame_manifest.jsonl``.

This stage does NOT generate thumbnails, OCR text, captions, or chunks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from video_rag.io_utils import read_json, write_jsonl
from video_rag.schemas import FrameSample, MediaMetadata, VideoManifest

PathLike = str | Path

DEFAULT_INTERVAL_SECONDS = 5
DEFAULT_JPEG_QUALITY = 85
DEFAULT_BINARY_RELATIVE = Path("cpp") / "frame_extract" / "build" / "raggers_frame_extract"
BINARY_ENV_VAR = "RAGGERS_FRAME_EXTRACT_BIN"
SUBPROCESS_TIMEOUT_SECONDS = 600


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


def _resolve_binary(explicit: PathLike | None) -> Path:
    """Find the ``raggers_frame_extract`` binary or raise with a helpful hint."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env_value = os.environ.get(BINARY_ENV_VAR)
    if env_value:
        candidates.append(Path(env_value))
    candidates.append(Path.cwd() / DEFAULT_BINARY_RELATIVE)

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "raggers_frame_extract binary not found. Build it with:\n"
        "  cmake -S cpp/frame_extract -B cpp/frame_extract/build "
        "-DCMAKE_BUILD_TYPE=Release\n"
        "  cmake --build cpp/frame_extract/build -j\n"
        f"Searched: {searched}"
    )


def _is_dir_nonempty(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def _to_repo_relative(absolute: Path, repo_root: Path) -> str:
    """Best-effort repo-root-relative POSIX path. Falls back to absolute POSIX."""
    try:
        rel = absolute.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return absolute.resolve().as_posix()
    return rel.as_posix()


def _parse_extractor_stdout(
    stdout: str, *, video_id: str, repo_root: Path
) -> list[FrameSample]:
    """Parse one ``FrameSample`` per non-blank stdout line emitted by the binary.

    The binary prints ``timestamp / frame_path / width / height`` per frame.
    Python is responsible for attaching ``video_id`` and ``sampling_method``
    so the binary stays agnostic to the wider artifact contract.
    """
    samples: list[FrameSample] = []
    for lineno, raw in enumerate(stdout.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"frame extractor emitted invalid JSON on line {lineno}: {exc.msg}"
            ) from exc
        if not isinstance(obj, dict):
            raise RuntimeError(
                f"frame extractor line {lineno} is not a JSON object: {obj!r}"
            )
        frame_path = obj.get("frame_path")
        if not isinstance(frame_path, str) or not frame_path:
            raise RuntimeError(
                f"frame extractor line {lineno} missing 'frame_path'"
            )
        record = {
            "video_id": video_id,
            "timestamp": obj.get("timestamp"),
            "frame_path": _to_repo_relative(Path(frame_path), repo_root),
            "width": obj.get("width"),
            "height": obj.get("height"),
            "sampling_method": "fixed_interval",
        }
        try:
            samples.append(FrameSample.model_validate(record))
        except ValidationError as exc:
            raise RuntimeError(
                f"frame extractor line {lineno} failed FrameSample validation:\n{exc}"
            ) from exc
    return samples


def _format_timestamps_arg(timestamps: Iterable[float]) -> str:
    return ",".join(f"{t:.3f}" for t in timestamps)


def _run_extractor(
    binary: Path,
    *,
    video_path: Path,
    out_dir: Path,
    timestamps: list[float],
    quality: int,
) -> str:
    """Invoke the C++ binary and return its stdout. Raise on failure."""
    cmd = [
        str(binary),
        "--video",
        str(video_path),
        "--out-dir",
        str(out_dir),
        "--timestamps",
        _format_timestamps_arg(timestamps),
        "--quality",
        str(quality),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"failed to execute frame extractor at {binary}: {exc}"
        ) from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or "(no stderr)"
        raise RuntimeError(
            f"frame extractor exited with code {result.returncode}: {stderr}"
        )
    return result.stdout


def sample_frames(
    video_id: str,
    data_dir: PathLike = "data",
    interval_seconds: int | float = DEFAULT_INTERVAL_SECONDS,
    overwrite: bool = False,
    binary_path: PathLike | None = None,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> list[FrameSample]:
    """Sample frames at a fixed interval. Returns the persisted ``FrameSample`` records.

    Raises:
        ValueError: ``interval_seconds`` is not a positive integer or ``video_id``
            is empty.
        FileNotFoundError: required input artifact, video file, or extractor
            binary is missing.
        FileExistsError: prior frame outputs exist and ``overwrite`` is ``False``.
        RuntimeError: the extractor failed or emitted malformed output.
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

    binary = _resolve_binary(binary_path)
    timestamps = _compute_schedule(metadata.duration_seconds, interval)

    if not timestamps:
        write_jsonl(manifest_out, [])
        return []

    stdout = _run_extractor(
        binary,
        video_path=video_path,
        out_dir=out_dir.resolve(),
        timestamps=timestamps,
        quality=jpeg_quality,
    )
    samples = _parse_extractor_stdout(stdout, video_id=video_id, repo_root=repo_root)
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
        "--binary-path",
        dest="binary_path",
        type=Path,
        default=None,
        help="Path to the raggers_frame_extract binary.",
    )
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
            binary_path=args.binary_path,
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
