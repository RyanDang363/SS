"""Validation CLI for Stage 0-lite artifacts.

Usage:
    python -m video_rag.validate <path> --type <video_manifest|media_metadata>

Exits 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Type

from pydantic import BaseModel, ValidationError

from video_rag.io_utils import read_json
from video_rag.schemas import MediaMetadata, VideoManifest

SCHEMAS: dict[str, Type[BaseModel]] = {
    "video_manifest": VideoManifest,
    "media_metadata": MediaMetadata,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video_rag.validate",
        description="Validate a RAGGERS Stage 0-lite artifact against its schema.",
    )
    parser.add_argument("path", type=Path, help="Path to a JSON artifact.")
    parser.add_argument(
        "--type",
        dest="artifact_type",
        required=True,
        choices=sorted(SCHEMAS.keys()),
        help="Artifact type to validate against.",
    )
    return parser


def validate_file(path: Path, artifact_type: str) -> tuple[bool, str]:
    """Return ``(ok, message)`` for a single file."""
    model = SCHEMAS[artifact_type]
    if not path.exists():
        return False, f"FAIL  {path}: file not found"
    try:
        read_json(path, model)
    except ValidationError as e:
        return False, f"FAIL  {path} ({artifact_type}):\n{e}"
    except ValueError as e:
        return False, f"FAIL  {e}"
    except Exception as e:
        return False, f"FAIL  {path}: {e.__class__.__name__}: {e}"
    return True, f"OK    {path} ({artifact_type})"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ok, message = validate_file(args.path, args.artifact_type)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
