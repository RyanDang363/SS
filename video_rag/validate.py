"""Validation CLI for RAGGERS artifacts.

Usage:
    python -m video_rag.validate <path> --type <video_manifest|media_metadata|transcript_segments>

Exits 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Type

from pydantic import BaseModel, ValidationError

from video_rag.io_utils import read_json, read_jsonl
from video_rag.schemas import MediaMetadata, TranscriptSegment, VideoManifest

JSON_SCHEMAS: dict[str, Type[BaseModel]] = {
    "video_manifest": VideoManifest,
    "media_metadata": MediaMetadata,
}

JSONL_SCHEMAS: dict[str, Type[BaseModel]] = {
    "transcript_segments": TranscriptSegment,
}

ALL_TYPES = sorted({*JSON_SCHEMAS, *JSONL_SCHEMAS})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video_rag.validate",
        description="Validate a RAGGERS artifact against its schema.",
    )
    parser.add_argument("path", type=Path, help="Path to a JSON or JSONL artifact.")
    parser.add_argument(
        "--type",
        dest="artifact_type",
        required=True,
        choices=ALL_TYPES,
        help="Artifact type to validate against.",
    )
    return parser


def _validate_json(path: Path, artifact_type: str) -> tuple[bool, str]:
    model = JSON_SCHEMAS[artifact_type]
    try:
        read_json(path, model)
    except ValidationError as e:
        return False, f"FAIL  {path} ({artifact_type}):\n{e}"
    except ValueError as e:
        return False, f"FAIL  {e}"
    except Exception as e:
        return False, f"FAIL  {path}: {e.__class__.__name__}: {e}"
    return True, f"OK    {path} ({artifact_type})"


def _validate_transcript_segments(path: Path) -> tuple[bool, str]:
    try:
        records = list(read_jsonl(path, TranscriptSegment))
    except ValueError as e:
        return False, f"FAIL  {e}"
    except Exception as e:
        return False, f"FAIL  {path}: {e.__class__.__name__}: {e}"

    if not records:
        return False, f"FAIL  {path}: transcript_segments file is empty"

    first_id = records[0].video_id
    for i, rec in enumerate(records, start=1):
        if rec.video_id != first_id:
            return False, (
                f"FAIL  {path}:{i}: mixed video_id "
                f"(expected {first_id!r}, got {rec.video_id!r})"
            )

    for i in range(1, len(records)):
        if records[i].start_time < records[i - 1].start_time:
            return False, (
                f"FAIL  {path}:{i + 1}: start_time {records[i].start_time} "
                f"is before previous start_time {records[i - 1].start_time}"
            )

    return True, f"OK    {path} (transcript_segments, {len(records)} segments)"


def validate_file(path: Path, artifact_type: str) -> tuple[bool, str]:
    """Return ``(ok, message)`` for a single file."""
    if not path.exists():
        return False, f"FAIL  {path}: file not found"

    if artifact_type in JSON_SCHEMAS:
        return _validate_json(path, artifact_type)
    if artifact_type == "transcript_segments":
        return _validate_transcript_segments(path)
    return False, f"FAIL  {path}: unknown artifact type {artifact_type!r}"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ok, message = validate_file(args.path, args.artifact_type)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
