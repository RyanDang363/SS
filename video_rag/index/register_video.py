"""Stage 1: register a local video file into the RAGGERS artifact layout.

Copies (or symlinks) the input video into ``{data_dir}/videos/`` and writes a
``VideoManifest`` to ``{data_dir}/manifests/{video_id}/video_manifest.json``.

This stage does NOT inspect codecs, duration, fps, or audio — see Stage 2.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from video_rag.io_utils import write_json
from video_rag.schemas import VideoManifest

PathLike = str | Path
Mode = Literal["copy", "symlink"]

_DISALLOWED = re.compile(r"[^a-z0-9_-]+")
_MULTI_UNDERSCORE = re.compile(r"_+")


def sanitize_video_id(raw: str) -> str:
    """Lowercase, replace whitespace, drop disallowed chars, collapse separators."""
    s = raw.strip().lower().replace(" ", "_")
    s = _DISALLOWED.sub("_", s)
    s = _MULTI_UNDERSCORE.sub("_", s)
    return s.strip("_-")


def _derive_video_id(title: str | None, filename: str) -> str:
    base = (title or Path(filename).stem).strip()
    return sanitize_video_id(base)


def _id_taken(video_id: str, videos_dir: Path, manifests_dir: Path) -> bool:
    if (manifests_dir / video_id).exists():
        return True
    if videos_dir.exists():
        for child in videos_dir.iterdir():
            if child.stem == video_id:
                return True
    return False


def _find_unique_id(base: str, videos_dir: Path, manifests_dir: Path) -> str:
    candidate = base
    n = 2
    while _id_taken(candidate, videos_dir, manifests_dir):
        candidate = f"{base}_{n}"
        n += 1
    return candidate


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _remove_if_present(p: Path) -> None:
    if p.is_symlink() or p.exists():
        p.unlink()


def register_video(
    video_path: PathLike,
    title: str | None = None,
    video_id: str | None = None,
    mode: Mode = "copy",
    overwrite: bool = False,
    data_dir: PathLike = "data",
) -> VideoManifest:
    """Register a local video file. Returns the persisted ``VideoManifest``."""
    if mode not in ("copy", "symlink"):
        raise ValueError(f"unknown mode: {mode!r}")

    src = Path(video_path)
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"input video not found or not a file: {src}")

    data_root = Path(data_dir)
    videos_dir = data_root / "videos"
    manifests_dir = data_root / "manifests"

    if video_id is not None:
        chosen_id = sanitize_video_id(video_id)
        if not chosen_id:
            raise ValueError(f"video_id {video_id!r} sanitized to an empty string")
    else:
        derived = _derive_video_id(title, src.name)
        if not derived:
            raise ValueError("could not derive a non-empty video_id from title/filename")
        chosen_id = _find_unique_id(derived, videos_dir, manifests_dir)

    extension = src.suffix
    dest_video = videos_dir / f"{chosen_id}{extension}"
    dest_manifest_dir = manifests_dir / chosen_id
    dest_manifest = dest_manifest_dir / "video_manifest.json"

    video_exists = dest_video.is_symlink() or dest_video.exists()
    manifest_exists = dest_manifest.exists()
    if (video_exists or manifest_exists) and not overwrite:
        clash = dest_video if video_exists else dest_manifest
        raise FileExistsError(f"destination already exists: {clash} (pass overwrite=True)")

    videos_dir.mkdir(parents=True, exist_ok=True)
    _remove_if_present(dest_video)

    if mode == "copy":
        shutil.copy2(src, dest_video)
    else:
        dest_video.symlink_to(src.resolve())

    title_value = (title or "").strip() or src.stem

    manifest = VideoManifest(
        video_id=chosen_id,
        title=title_value,
        source_path=f"{data_root.name}/videos/{chosen_id}{extension}",
        original_filename=src.name,
        created_at=_now_iso_utc(),
    )
    write_json(dest_manifest, manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_rag.index.register_video",
        description="Register a local video into the RAGGERS artifact layout (Stage 1).",
    )
    p.add_argument("--video", required=True, type=Path, help="Path to a local video file.")
    p.add_argument("--title", default=None, help="Human-readable title.")
    p.add_argument("--video-id", dest="video_id", default=None, help="Explicit video_id.")
    p.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    p.add_argument("--overwrite", action="store_true", help="Replace existing artifacts.")
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
    try:
        manifest = register_video(
            video_path=args.video,
            title=args.title,
            video_id=args.video_id,
            mode=args.mode,
            overwrite=args.overwrite,
            data_dir=args.data_dir,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as e:
        print(f"FAIL  {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1

    manifest_path = (
        Path(args.data_dir) / "manifests" / manifest.video_id / "video_manifest.json"
    )
    print("Registered video:")
    print(f"  video_id: {manifest.video_id}")
    print(f"  video: {manifest.source_path}")
    print(f"  manifest: {manifest_path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
