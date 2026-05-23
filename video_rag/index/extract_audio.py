"""Stage 3: extract normalized audio from a registered, probed video.

Reads ``VideoManifest`` and ``MediaMetadata`` artifacts, verifies the video has
audio, and writes a mono 16 kHz WAV file for the later transcription stage.

This stage does NOT transcribe, sample frames, OCR, caption, chunk, embed, or
retrieve anything.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from video_rag.io_utils import read_json
from video_rag.schemas import MediaMetadata, VideoManifest

PathLike = str | Path


def _resolve_source_path(source_path: str) -> Path:
    p = Path(source_path)
    if p.is_absolute():
        return p
    return Path.cwd() / p


def extract_audio(
    video_id: str,
    data_dir: PathLike = "data",
    overwrite: bool = False,
) -> Path:
    """Extract mono 16 kHz WAV audio for ``video_id`` and return its path."""
    data_root = Path(data_dir)
    manifest_dir = data_root / "manifests" / video_id
    video_manifest_path = manifest_dir / "video_manifest.json"
    media_metadata_path = manifest_dir / "media_metadata.json"
    audio_output_path = data_root / "audio" / f"{video_id}.wav"

    if not video_manifest_path.exists():
        raise FileNotFoundError(
            f"missing video manifest: {video_manifest_path}. Run Stage 1 first."
        )

    if not media_metadata_path.exists():
        raise FileNotFoundError(
            f"missing media metadata: {media_metadata_path}. Run Stage 2 first."
        )

    manifest = read_json(video_manifest_path, VideoManifest)
    metadata = read_json(media_metadata_path, MediaMetadata)

    if not metadata.has_audio:
        raise ValueError(
            f"video {video_id!r} has_audio=false in {media_metadata_path}; "
            "cannot extract audio."
        )

    video_path = _resolve_source_path(manifest.source_path)
    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError(f"registered video file not found: {video_path}")

    if audio_output_path.exists() and not overwrite:
        raise FileExistsError(
            f"audio output already exists: {audio_output_path} (pass overwrite=True)"
        )

    audio_output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(audio_output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else "no stderr output"
        raise RuntimeError(f"ffmpeg failed while extracting audio:\n{stderr}") from e

    return audio_output_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_rag.index.extract_audio",
        description="Extract normalized WAV audio for a registered video (Stage 3).",
    )
    p.add_argument("--video-id", required=True, help="Registered video_id.")
    p.add_argument(
        "--data-dir",
        dest="data_dir",
        type=Path,
        default=Path("data"),
        help="Artifact root (default: data).",
    )
    p.add_argument("--overwrite", action="store_true", help="Replace existing audio.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        output_path = extract_audio(
            video_id=args.video_id,
            data_dir=args.data_dir,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError) as e:
        print(f"FAIL  {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1

    print(f"Extracted audio: {output_path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
