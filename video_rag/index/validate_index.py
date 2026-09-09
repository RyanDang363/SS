"""Stage 13: validate an existing searchable video index.

This stage inspects artifacts produced by earlier stages and writes a validation
report. It does NOT create transcripts, chunks, embeddings, vector indexes,
retrieval results, or answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from video_rag.io_utils import read_json, read_jsonl, write_json
from video_rag.schemas import (
    Chunk,
    EmbeddingRecord,
    FrameSample,
    IndexValidationReport,
    MediaMetadata,
    OCRResult,
    TranscriptSegment,
    VLMCaption,
    VectorStoreManifest,
    VideoManifest,
)

PathLike = str | Path
T = TypeVar("T", bound=BaseModel)

VARIANT_TO_FIELD = {
    "transcript_only": "combined_text_transcript_only",
    "transcript_ocr": "combined_text_transcript_ocr",
    "transcript_vlm": "combined_text_transcript_vlm",
    "transcript_ocr_vlm": "combined_text_all",
}
EPSILON = 1e-6


def _format_chunk_seconds(chunk_seconds: int | float) -> str:
    value = float(chunk_seconds)
    if value <= 0:
        raise ValueError("chunk_seconds must be greater than 0")
    return f"{value:g}"


def _read_json_report(path: Path, model: Type[T], errors: list[str]) -> T | None:
    try:
        return read_json(path, model)
    except Exception as e:
        errors.append(f"{path}: invalid {model.__name__}: {e}")
        return None


def _read_jsonl_report(
    path: Path,
    model: Type[T],
    errors: list[str],
) -> list[T]:
    try:
        return list(read_jsonl(path, model))
    except Exception as e:
        errors.append(f"{path}: invalid {model.__name__} JSONL: {e}")
        return []


def _read_jsonl_raw(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    raw = json.loads(stripped)
                except json.JSONDecodeError as e:
                    errors.append(f"{path}:{lineno}: invalid JSON: {e.msg}")
                    continue
                if not isinstance(raw, dict):
                    errors.append(f"{path}:{lineno}: expected JSON object")
                    continue
                records.append(raw)
    except OSError as e:
        errors.append(f"{path}: could not read JSONL: {e}")
    return records


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _check_time_within_duration(
    label: str,
    end_time: float,
    duration_seconds: float | None,
    errors: list[str],
) -> None:
    if duration_seconds is not None and end_time > duration_seconds + EPSILON:
        errors.append(
            f"{label}: end_time {end_time} exceeds media duration "
            f"{duration_seconds}"
        )


def _validate_chunks(
    chunks_path: Path,
    video_id: str,
    variant: str,
    duration_seconds: float | None,
    errors: list[str],
    warnings: list[str],
) -> tuple[list[Chunk], set[str]]:
    raw_chunks = _read_jsonl_raw(chunks_path, errors)
    chunk_pairs: list[tuple[dict[str, Any], Chunk]] = []
    for index, raw in enumerate(raw_chunks, start=1):
        try:
            chunk_pairs.append((raw, Chunk.model_validate(raw)))
        except ValidationError as e:
            errors.append(f"{chunks_path}:{index}: validation failed\n{e}")
    chunks = [chunk for _raw, chunk in chunk_pairs]
    selected_field = VARIANT_TO_FIELD.get(variant)
    chunk_ids: set[str] = set()
    seen_indexes: set[int] = set()

    if selected_field is None:
        errors.append(f"unknown embedding variant: {variant}")

    for raw, chunk in chunk_pairs:
        label = f"{chunks_path}:{chunk.chunk_id}"
        if chunk.video_id != video_id:
            errors.append(
                f"{label}: video_id {chunk.video_id!r} does not match {video_id!r}"
            )
        _check_time_within_duration(label, chunk.end_time, duration_seconds, errors)
        if chunk.chunk_id in chunk_ids:
            errors.append(f"{label}: duplicate chunk_id {chunk.chunk_id!r}")
        chunk_ids.add(chunk.chunk_id)
        if chunk.chunk_index in seen_indexes:
            errors.append(f"{label}: duplicate chunk_index {chunk.chunk_index}")
        seen_indexes.add(chunk.chunk_index)
        if selected_field is not None:
            if selected_field not in raw:
                errors.append(f"{label}: missing selected search text field {selected_field}")
            elif _is_empty(raw.get(selected_field)):
                warnings.append(
                    f"{label}: selected search text field {selected_field} is empty"
                )
        if _is_empty(chunk.transcript_text):
            warnings.append(f"{label}: transcript_text is empty")

    return chunks, chunk_ids


def _validate_embeddings(
    embeddings_path: Path,
    video_id: str,
    variant: str,
    chunk_ids: set[str],
    errors: list[str],
    warnings: list[str],
) -> tuple[list[EmbeddingRecord], int | None]:
    embeddings = _read_jsonl_report(embeddings_path, EmbeddingRecord, errors)
    dims: set[int] = set()

    for embedding in embeddings:
        label = f"{embeddings_path}:{embedding.chunk_id}"
        if embedding.video_id != video_id:
            errors.append(
                f"{label}: video_id {embedding.video_id!r} does not match {video_id!r}"
            )
        if embedding.embedding_variant != variant:
            errors.append(
                f"{label}: embedding_variant {embedding.embedding_variant!r} "
                f"does not match {variant!r}"
            )
        vector_dim = len(embedding.vector)
        dims.add(vector_dim)
        if embedding.vector_dim is not None and embedding.vector_dim != vector_dim:
            errors.append(
                f"{label}: vector_dim {embedding.vector_dim} does not match "
                f"len(vector) {vector_dim}"
            )
        if chunk_ids and embedding.chunk_id not in chunk_ids:
            errors.append(
                f"{label}: chunk_id {embedding.chunk_id!r} was not found in chunks"
            )

    if len(dims) > 1:
        errors.append(f"{embeddings_path}: inconsistent vector dimensions {sorted(dims)}")

    embedded_chunk_ids = {embedding.chunk_id for embedding in embeddings}
    missing_embeddings = sorted(chunk_ids - embedded_chunk_ids)
    if missing_embeddings:
        warnings.append(
            f"{embeddings_path}: {len(missing_embeddings)} chunks have no embedding"
        )

    return embeddings, next(iter(dims)) if len(dims) == 1 else None


def _validate_vector_store(
    manifest_path: Path,
    video_id: str,
    chunk_seconds: float,
    variant: str,
    num_embeddings: int,
    embedding_dim: int | None,
    errors: list[str],
    warnings: list[str],
) -> int:
    if not manifest_path.exists():
        warnings.append(
            f"missing vector store manifest: {manifest_path}; "
            "skipping vector count checks"
        )
        return 0

    manifest = _read_json_report(manifest_path, VectorStoreManifest, errors)
    if manifest is None:
        return 0

    if manifest.video_id != video_id:
        errors.append(
            f"{manifest_path}: video_id {manifest.video_id!r} does not match "
            f"{video_id!r}"
        )
    if abs(float(manifest.chunk_seconds) - float(chunk_seconds)) > EPSILON:
        errors.append(
            f"{manifest_path}: chunk_seconds {manifest.chunk_seconds} does not match "
            f"{chunk_seconds}"
        )
    if manifest.embedding_variant != variant:
        errors.append(
            f"{manifest_path}: embedding_variant {manifest.embedding_variant!r} "
            f"does not match {variant!r}"
        )
    if manifest.backend is not None and not str(manifest.backend).strip():
        errors.append(f"{manifest_path}: backend is empty")
    if manifest.num_vectors != num_embeddings:
        errors.append(
            f"{manifest_path}: num_vectors {manifest.num_vectors} does not match "
            f"embedding records {num_embeddings}"
        )
    if (
        embedding_dim is not None
        and manifest.vector_dim is not None
        and manifest.vector_dim != embedding_dim
    ):
        errors.append(
            f"{manifest_path}: vector_dim {manifest.vector_dim} does not match "
            f"embedding vector dimension {embedding_dim}"
        )
    return manifest.num_vectors


def _validate_optional_timestamps(
    path: Path,
    model: Type[T],
    video_id: str,
    duration_seconds: float | None,
    errors: list[str],
) -> None:
    if not path.exists():
        return
    records = _read_jsonl_report(path, model, errors)
    for record in records:
        label = f"{path}:{getattr(record, 'video_id', 'unknown')}"
        record_video_id = getattr(record, "video_id", None)
        if record_video_id is not None and record_video_id != video_id:
            errors.append(
                f"{label}: video_id {record_video_id!r} does not match {video_id!r}"
            )
        if isinstance(record, (FrameSample, OCRResult)):
            timestamp = record.timestamp
            if duration_seconds is not None and timestamp > duration_seconds + EPSILON:
                errors.append(
                    f"{label}: timestamp {timestamp} exceeds media duration "
                    f"{duration_seconds}"
                )
        elif isinstance(record, VLMCaption):
            if record.end_time <= record.start_time:
                errors.append(
                    f"{label}: end_time {record.end_time} must be greater than "
                    f"start_time {record.start_time}"
                )
            _check_time_within_duration(label, record.end_time, duration_seconds, errors)


def validate_index(
    video_id: str,
    data_dir: PathLike = "data",
    chunk_seconds: int | float = 30,
    variant: str = "transcript_ocr_vlm",
) -> IndexValidationReport:
    """Validate existing Stage 1-12 artifacts and write a Stage 13 report."""
    chunk_seconds_label = _format_chunk_seconds(chunk_seconds)
    chunk_seconds_value = float(chunk_seconds)
    data_root = Path(data_dir)
    video_manifest_path = data_root / "manifests" / video_id / "video_manifest.json"
    media_metadata_path = data_root / "manifests" / video_id / "media_metadata.json"
    transcript_path = data_root / "transcripts" / f"{video_id}.jsonl"
    chunks_path = data_root / "chunks" / f"{video_id}_{chunk_seconds_label}s_enriched.jsonl"
    embeddings_path = (
        data_root / "embeddings" / f"{video_id}_{chunk_seconds_label}s_{variant}.jsonl"
    )
    index_dir = data_root / "indexes" / f"{video_id}_{chunk_seconds_label}s_{variant}"
    vector_store_manifest_path = index_dir / "vector_store_manifest.json"
    validation_output_path = (
        data_root
        / "validation"
        / f"{video_id}_{chunk_seconds_label}s_{variant}_validation.json"
    )
    frame_manifest_path = data_root / "frames" / video_id / "frame_manifest.jsonl"
    ocr_path = data_root / "ocr" / f"{video_id}.jsonl"
    captions_path = data_root / "captions" / f"{video_id}.jsonl"

    errors: list[str] = []
    warnings: list[str] = []
    duration_seconds: float | None = None
    chunks: list[Chunk] = []
    embeddings: list[EmbeddingRecord] = []
    embedding_dim: int | None = None
    num_vectors = 0

    for path, label in [
        (video_manifest_path, "video manifest"),
        (media_metadata_path, "media metadata"),
        (transcript_path, "transcript JSONL"),
        (chunks_path, "enriched chunks JSONL"),
        (embeddings_path, "embeddings JSONL"),
    ]:
        if not path.exists():
            errors.append(f"missing required {label}: {path}")

    if not index_dir.exists() or not index_dir.is_dir():
        errors.append(f"missing required vector index directory: {index_dir}")

    if video_manifest_path.exists():
        _read_json_report(video_manifest_path, VideoManifest, errors)

    if media_metadata_path.exists():
        media_metadata = _read_json_report(media_metadata_path, MediaMetadata, errors)
        if media_metadata is not None:
            duration_seconds = media_metadata.duration_seconds

    if transcript_path.exists():
        _read_jsonl_report(transcript_path, TranscriptSegment, errors)

    chunk_ids: set[str] = set()
    if chunks_path.exists():
        chunks, chunk_ids = _validate_chunks(
            chunks_path,
            video_id,
            variant,
            duration_seconds,
            errors,
            warnings,
        )

    if embeddings_path.exists():
        embeddings, embedding_dim = _validate_embeddings(
            embeddings_path,
            video_id,
            variant,
            chunk_ids,
            errors,
            warnings,
        )

    if index_dir.exists() and index_dir.is_dir():
        num_vectors = _validate_vector_store(
            vector_store_manifest_path,
            video_id,
            chunk_seconds_value,
            variant,
            len(embeddings),
            embedding_dim,
            errors,
            warnings,
        )

    _validate_optional_timestamps(
        frame_manifest_path, FrameSample, video_id, duration_seconds, errors
    )
    _validate_optional_timestamps(
        ocr_path, OCRResult, video_id, duration_seconds, errors
    )
    _validate_optional_timestamps(
        captions_path, VLMCaption, video_id, duration_seconds, errors
    )

    report = IndexValidationReport(
        video_id=video_id,
        chunk_seconds=chunk_seconds_value,
        embedding_variant=variant,
        status="failed" if errors else "passed",
        num_chunks=len(chunks),
        num_embeddings=len(embeddings),
        num_vectors=num_vectors,
        errors=errors,
        warnings=warnings,
    )
    write_json(validation_output_path, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_rag.index.validate_index",
        description="Validate an existing searchable video index (Stage 13).",
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
        "--chunk-seconds",
        type=float,
        default=30,
        help="Chunk window size used by Stage 9 (default: 30).",
    )
    p.add_argument(
        "--variant",
        default="transcript_ocr_vlm",
        help="Embedding/search-text variant (default: transcript_ocr_vlm).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = validate_index(
            video_id=args.video_id,
            data_dir=args.data_dir,
            chunk_seconds=args.chunk_seconds,
            variant=args.variant,
        )
    except ValueError as e:
        print(f"FAIL  {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1

    chunk_seconds_label = _format_chunk_seconds(args.chunk_seconds)
    report_path = (
        Path(args.data_dir)
        / "validation"
        / f"{args.video_id}_{chunk_seconds_label}s_{args.variant}_validation.json"
    )
    print(f"Wrote index validation report: {report_path.as_posix()} ({report.status})")
    return 0 if report.status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
