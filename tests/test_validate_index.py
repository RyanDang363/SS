from __future__ import annotations

import json
from pathlib import Path

from video_rag.index import validate_index as validate_index_module
from video_rag.index.validate_index import main, validate_index
from video_rag.io_utils import read_json, write_json, write_jsonl
from video_rag.schemas import (
    IndexValidationReport,
    MediaMetadata,
    TranscriptSegment,
    VectorStoreManifest,
    VideoManifest,
)


def _chunk(**overrides) -> dict:
    base = {
        "chunk_id": "lecture_001_chunk_0000",
        "video_id": "lecture_001",
        "chunk_index": 0,
        "start_time": 0.0,
        "end_time": 30.0,
        "transcript_text": "Bayes theorem updates probability.",
        "ocr_text": "P(A|B) = P(B|A)P(A)/P(B)",
        "vlm_caption": "The instructor points at a theorem slide.",
        "frame_paths": ["data/frames/lecture_001/frame_000000.jpg"],
        "chunk_seconds": 30.0,
        "overlap_seconds": 0.0,
        "chunking_strategy": "fixed",
        "combined_text_transcript_only": "Transcript:\nBayes theorem updates probability.",
        "combined_text_transcript_ocr": (
            "Transcript:\nBayes theorem updates probability.\n\n"
            "On-screen text:\nP(A|B) = P(B|A)P(A)/P(B)"
        ),
        "combined_text_transcript_vlm": (
            "Transcript:\nBayes theorem updates probability.\n\n"
            "Visual caption:\nThe instructor points at a theorem slide."
        ),
        "combined_text_all": (
            "Transcript:\nBayes theorem updates probability.\n\n"
            "On-screen text:\nP(A|B) = P(B|A)P(A)/P(B)\n\n"
            "Visual caption:\nThe instructor points at a theorem slide."
        ),
    }
    base.update(overrides)
    return base


def _embedding(**overrides) -> dict:
    base = {
        "chunk_id": "lecture_001_chunk_0000",
        "video_id": "lecture_001",
        "start_time": 0.0,
        "end_time": 30.0,
        "embedding_model": "text-embedding-3-small",
        "embedding_variant": "transcript_ocr_vlm",
        "vector": [0.1, 0.2, 0.3],
        "vector_dim": 3,
    }
    base.update(overrides)
    return base


def _write_valid_artifacts(
    data_dir: Path,
    chunks: list[dict] | None = None,
    embeddings: list[dict] | None = None,
    manifest: dict | None = None,
) -> None:
    video_id = "lecture_001"
    write_json(
        data_dir / "manifests" / video_id / "video_manifest.json",
        VideoManifest(
            video_id=video_id,
            title="Bayes Lecture",
            source_path="data/videos/lecture_001.mp4",
        ),
    )
    write_json(
        data_dir / "manifests" / video_id / "media_metadata.json",
        MediaMetadata(video_id=video_id, duration_seconds=60.0, has_audio=True),
    )
    write_jsonl(
        data_dir / "transcripts" / f"{video_id}.jsonl",
        [
            TranscriptSegment(
                video_id=video_id,
                start_time=0.0,
                end_time=10.0,
                text="Bayes theorem updates probability.",
            )
        ],
    )
    write_jsonl(
        data_dir / "chunks" / f"{video_id}_30s_enriched.jsonl",
        chunks if chunks is not None else [_chunk()],
    )
    write_jsonl(
        data_dir / "embeddings" / f"{video_id}_30s_transcript_ocr_vlm.jsonl",
        embeddings if embeddings is not None else [_embedding()],
    )
    index_dir = data_dir / "indexes" / f"{video_id}_30s_transcript_ocr_vlm"
    index_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        index_dir / "vector_store_manifest.json",
        manifest
        if manifest is not None
        else VectorStoreManifest(
            video_id=video_id,
            chunk_seconds=30,
            embedding_variant="transcript_ocr_vlm",
            backend="chroma",
            index_path="data/indexes/lecture_001_30s_transcript_ocr_vlm",
            num_vectors=1,
            vector_dim=3,
        ),
    )


def _messages(report: IndexValidationReport) -> str:
    return "\n".join(report.errors + report.warnings)


def test_successful_validation_with_small_fixture_artifacts(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_valid_artifacts(data_dir)

    report = validate_index("lecture_001", data_dir=data_dir)

    assert report.status == "passed"
    assert report.num_chunks == 1
    assert report.num_embeddings == 1
    assert report.num_vectors == 1
    assert report.errors == []


def test_missing_required_artifact_causes_failed_status(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_valid_artifacts(data_dir)
    (data_dir / "embeddings" / "lecture_001_30s_transcript_ocr_vlm.jsonl").unlink()

    report = validate_index("lecture_001", data_dir=data_dir)

    assert report.status == "failed"
    assert "missing required embeddings JSONL" in _messages(report)


def test_invalid_chunk_timestamps_cause_failed_status(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_valid_artifacts(data_dir, chunks=[_chunk(end_time=0.0)])

    report = validate_index("lecture_001", data_dir=data_dir)

    assert report.status == "failed"
    assert "validation failed" in _messages(report)


def test_chunk_beyond_media_duration_causes_failed_status(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_valid_artifacts(data_dir, chunks=[_chunk(end_time=61.0)])

    report = validate_index("lecture_001", data_dir=data_dir)

    assert report.status == "failed"
    assert "exceeds media duration" in _messages(report)


def test_duplicate_chunk_ids_cause_failed_status(tmp_path: Path):
    data_dir = tmp_path / "data"
    chunks = [_chunk(), _chunk(chunk_index=1)]
    _write_valid_artifacts(data_dir, chunks=chunks)

    report = validate_index("lecture_001", data_dir=data_dir)

    assert report.status == "failed"
    assert "duplicate chunk_id" in _messages(report)


def test_duplicate_chunk_indexes_cause_failed_status(tmp_path: Path):
    data_dir = tmp_path / "data"
    chunks = [_chunk(), _chunk(chunk_id="lecture_001_chunk_0001")]
    _write_valid_artifacts(data_dir, chunks=chunks)

    report = validate_index("lecture_001", data_dir=data_dir)

    assert report.status == "failed"
    assert "duplicate chunk_index" in _messages(report)


def test_missing_selected_search_text_field_causes_failed_status(tmp_path: Path):
    data_dir = tmp_path / "data"
    chunk = _chunk()
    del chunk["combined_text_all"]
    _write_valid_artifacts(data_dir, chunks=[chunk])

    report = validate_index("lecture_001", data_dir=data_dir)

    assert report.status == "failed"
    assert "missing selected search text field combined_text_all" in _messages(report)


def test_empty_selected_search_text_creates_warning(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_valid_artifacts(data_dir, chunks=[_chunk(combined_text_all="  ")])

    report = validate_index("lecture_001", data_dir=data_dir)

    assert report.status == "passed"
    assert "selected search text field combined_text_all is empty" in _messages(report)


def test_embedding_references_missing_chunk_id_causes_failed_status(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_valid_artifacts(
        data_dir,
        embeddings=[_embedding(chunk_id="lecture_001_chunk_missing")],
    )

    report = validate_index("lecture_001", data_dir=data_dir)

    assert report.status == "failed"
    assert "was not found in chunks" in _messages(report)


def test_inconsistent_vector_dimensions_cause_failed_status(tmp_path: Path):
    data_dir = tmp_path / "data"
    chunks = [_chunk(), _chunk(chunk_id="lecture_001_chunk_0001", chunk_index=1)]
    embeddings = [
        _embedding(),
        _embedding(
            chunk_id="lecture_001_chunk_0001",
            vector=[0.1, 0.2],
            vector_dim=2,
        ),
    ]
    _write_valid_artifacts(data_dir, chunks=chunks, embeddings=embeddings)

    report = validate_index("lecture_001", data_dir=data_dir)

    assert report.status == "failed"
    assert "inconsistent vector dimensions" in _messages(report)


def test_vector_dim_mismatch_with_len_vector_causes_failed_status(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_valid_artifacts(data_dir, embeddings=[_embedding(vector_dim=4)])

    report = validate_index("lecture_001", data_dir=data_dir)

    assert report.status == "failed"
    assert "vector_dim 4 does not match len(vector) 3" in _messages(report)


def test_wrong_embedding_variant_causes_failed_status(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_valid_artifacts(
        data_dir,
        embeddings=[_embedding(embedding_variant="transcript_only")],
    )

    report = validate_index("lecture_001", data_dir=data_dir)

    assert report.status == "failed"
    assert "embedding_variant 'transcript_only' does not match" in _messages(report)


def test_vector_count_mismatch_from_manifest_causes_failed_status(tmp_path: Path):
    data_dir = tmp_path / "data"
    manifest = {
        "video_id": "lecture_001",
        "chunk_seconds": 30,
        "embedding_variant": "transcript_ocr_vlm",
        "backend": "chroma",
        "index_path": "data/indexes/lecture_001_30s_transcript_ocr_vlm",
        "num_vectors": 2,
        "vector_dim": 3,
    }
    _write_valid_artifacts(data_dir, manifest=manifest)

    report = validate_index("lecture_001", data_dir=data_dir)

    assert report.status == "failed"
    assert "num_vectors 2 does not match embedding records 1" in _messages(report)


def test_optional_frame_artifact_with_invalid_timestamp_is_reported(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_valid_artifacts(data_dir)
    write_jsonl(
        data_dir / "frames" / "lecture_001" / "frame_manifest.jsonl",
        [
            {
                "video_id": "lecture_001",
                "timestamp": 61.0,
                "frame_path": "data/frames/lecture_001/frame_000061.jpg",
                "width": 1920,
                "height": 1080,
                "sampling_method": "fixed_interval",
            }
        ],
    )

    report = validate_index("lecture_001", data_dir=data_dir)

    assert report.status == "failed"
    assert "timestamp 61.0 exceeds media duration" in _messages(report)


def test_optional_ocr_artifact_with_invalid_timestamp_is_reported(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_valid_artifacts(data_dir)
    write_jsonl(
        data_dir / "ocr" / "lecture_001.jsonl",
        [
            {
                "video_id": "lecture_001",
                "timestamp": 61.0,
                "frame_path": "data/frames/lecture_001/frame_000061.jpg",
                "ocr_text": "Too late",
            }
        ],
    )

    report = validate_index("lecture_001", data_dir=data_dir)

    assert report.status == "failed"
    assert "timestamp 61.0 exceeds media duration" in _messages(report)


def test_optional_captions_artifact_with_invalid_time_range_is_reported(
    tmp_path: Path,
):
    data_dir = tmp_path / "data"
    _write_valid_artifacts(data_dir)
    write_jsonl(
        data_dir / "captions" / "lecture_001.jsonl",
        [
            {
                "video_id": "lecture_001",
                "start_time": 20.0,
                "end_time": 20.0,
                "frame_paths": ["data/frames/lecture_001/frame_000020.jpg"],
                "caption": "A slide is visible.",
                "caption_type": "generic",
                "model": "gpt-4o-mini",
            }
        ],
    )

    report = validate_index("lecture_001", data_dir=data_dir)

    assert report.status == "failed"
    assert "must be greater than start_time" in _messages(report)


def test_report_json_is_written(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_valid_artifacts(data_dir)

    report = validate_index("lecture_001", data_dir=data_dir)

    report_path = (
        data_dir
        / "validation"
        / "lecture_001_30s_transcript_ocr_vlm_validation.json"
    )
    loaded = read_json(report_path, IndexValidationReport)
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded == report
    assert raw["status"] == "passed"


def test_cli_success_path_exits_zero(tmp_path: Path, monkeypatch, capsys):
    def fake_validate_index(video_id, data_dir, chunk_seconds, variant):
        assert video_id == "lecture_001"
        assert data_dir == tmp_path / "data"
        assert chunk_seconds == 30.0
        assert variant == "transcript_ocr_vlm"
        return IndexValidationReport(
            video_id=video_id,
            chunk_seconds=30,
            embedding_variant=variant,
            status="passed",
            num_chunks=1,
            num_embeddings=1,
            num_vectors=1,
            errors=[],
            warnings=[],
        )

    monkeypatch.setattr(validate_index_module, "validate_index", fake_validate_index)

    exit_code = main(
        [
            "--video-id",
            "lecture_001",
            "--data-dir",
            str(tmp_path / "data"),
            "--chunk-seconds",
            "30",
            "--variant",
            "transcript_ocr_vlm",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Wrote index validation report:" in captured.out
    assert "(passed)" in captured.out


def test_cli_failure_path_exits_nonzero(tmp_path: Path, monkeypatch, capsys):
    def fake_validate_index(video_id, data_dir, chunk_seconds, variant):
        return IndexValidationReport(
            video_id=video_id,
            chunk_seconds=chunk_seconds,
            embedding_variant=variant,
            status="failed",
            num_chunks=0,
            num_embeddings=0,
            num_vectors=0,
            errors=["missing required transcript JSONL"],
            warnings=[],
        )

    monkeypatch.setattr(validate_index_module, "validate_index", fake_validate_index)

    exit_code = main(
        ["--video-id", "lecture_001", "--data-dir", str(tmp_path / "data")]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "(failed)" in captured.out
