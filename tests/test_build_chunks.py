from __future__ import annotations

from pathlib import Path

import pytest

from video_rag.index import build_chunks as build_chunks_module
from video_rag.index.build_chunks import build_chunks, main
from video_rag.io_utils import read_jsonl, write_json, write_jsonl
from video_rag.schemas import (
    Chunk,
    FrameSample,
    MediaMetadata,
    OCRResult,
    TranscriptSegment,
    VLMCaption,
)

VIDEO_ID = "lecture_001"


def _media_metadata(**overrides) -> MediaMetadata:
    base = dict(video_id=VIDEO_ID, duration_seconds=95.0, has_audio=True)
    base.update(overrides)
    return MediaMetadata(**base)


def _transcript_segment(**overrides) -> TranscriptSegment:
    base = dict(
        video_id=VIDEO_ID,
        start_time=0.0,
        end_time=5.0,
        text="Sample transcript text.",
    )
    base.update(overrides)
    return TranscriptSegment(**base)


def _ocr_result(**overrides) -> OCRResult:
    base = dict(
        video_id=VIDEO_ID,
        timestamp=10.0,
        frame_path=f"data/frames/{VIDEO_ID}/frame_000010.jpg",
        ocr_text="P(A|B)",
        confidence=0.9,
    )
    base.update(overrides)
    return OCRResult(**base)


def _frame_sample(**overrides) -> FrameSample:
    base = dict(
        video_id=VIDEO_ID,
        timestamp=10.0,
        frame_path=f"data/frames/{VIDEO_ID}/frame_000010.jpg",
        width=1920,
        height=1080,
    )
    base.update(overrides)
    return FrameSample(**base)


def _vlm_caption(**overrides) -> VLMCaption:
    base = dict(
        video_id=VIDEO_ID,
        start_time=0.0,
        end_time=15.0,
        frame_paths=[f"data/frames/{VIDEO_ID}/frame_000000.jpg"],
        caption="An instructor stands beside a slide.",
        model="gpt-4o-mini",
    )
    base.update(overrides)
    return VLMCaption(**base)


def _write_metadata(data_dir: Path, metadata: MediaMetadata | None = None) -> Path:
    metadata = metadata or _media_metadata()
    path = data_dir / "manifests" / VIDEO_ID / "media_metadata.json"
    write_json(path, metadata)
    return path


def _write_transcript(
    data_dir: Path, segments: list[TranscriptSegment] | None = None
) -> Path:
    if segments is None:
        segments = [_transcript_segment()]
    path = data_dir / "transcripts" / f"{VIDEO_ID}.jsonl"
    write_jsonl(path, segments)
    return path


def _write_ocr(data_dir: Path, results: list[OCRResult]) -> Path:
    path = data_dir / "ocr" / f"{VIDEO_ID}.jsonl"
    write_jsonl(path, results)
    return path


def _write_frames(data_dir: Path, samples: list[FrameSample]) -> Path:
    path = data_dir / "frames" / VIDEO_ID / "frame_manifest.jsonl"
    write_jsonl(path, samples)
    return path


def _write_captions(data_dir: Path, captions: list[VLMCaption]) -> Path:
    path = data_dir / "captions" / f"{VIDEO_ID}.jsonl"
    write_jsonl(path, captions)
    return path


# 1. Correct fixed 30s windows from media duration.
def test_creates_fixed_30s_windows(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_metadata(data_dir, _media_metadata(duration_seconds=95.0))
    _write_transcript(data_dir)

    chunks = build_chunks(VIDEO_ID, data_dir=data_dir, chunk_seconds=30)

    windows = [(c.start_time, c.end_time) for c in chunks]
    assert windows == [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0), (90.0, 95.0)]
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]
    assert chunks[0].chunk_id == "lecture_001_chunk_0000"
    assert chunks[3].chunk_id == "lecture_001_chunk_0003"


# 2. Last chunk ends at actual duration.
def test_last_chunk_ends_at_duration(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_metadata(data_dir, _media_metadata(duration_seconds=95.0))
    _write_transcript(data_dir)

    chunks = build_chunks(VIDEO_ID, data_dir=data_dir, chunk_seconds=30)

    assert chunks[-1].end_time == 95.0


# 3. Transcript segments attach when overlapping a chunk.
def test_transcript_attaches_when_overlapping(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_metadata(data_dir)
    segments = [
        _transcript_segment(start_time=0.0, end_time=10.0, text="First."),
        # Spans the 30s boundary; must appear in both chunk 0 and chunk 1.
        _transcript_segment(start_time=28.0, end_time=35.0, text="Boundary."),
        _transcript_segment(start_time=70.0, end_time=75.0, text="Later."),
    ]
    _write_transcript(data_dir, segments)

    chunks = build_chunks(VIDEO_ID, data_dir=data_dir, chunk_seconds=30)

    assert chunks[0].transcript_text == "First. Boundary."
    assert chunks[1].transcript_text == "Boundary."
    assert chunks[2].transcript_text == "Later."
    assert chunks[3].transcript_text == ""


# 4. OCR records attach when timestamp falls inside a chunk.
def test_ocr_attaches_by_timestamp(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_metadata(data_dir)
    _write_transcript(data_dir)
    _write_ocr(
        data_dir,
        [
            _ocr_result(timestamp=10.0, ocr_text="early"),
            _ocr_result(timestamp=30.0, ocr_text="boundary"),  # belongs to chunk 1
            _ocr_result(timestamp=65.0, ocr_text="late"),
        ],
    )

    chunks = build_chunks(VIDEO_ID, data_dir=data_dir, chunk_seconds=30)

    assert chunks[0].ocr_text == "early"
    assert chunks[1].ocr_text == "boundary"
    assert chunks[2].ocr_text == "late"


# 5. Frame records attach when timestamp falls inside a chunk.
def test_frames_attach_by_timestamp(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_metadata(data_dir)
    _write_transcript(data_dir)
    _write_frames(
        data_dir,
        [
            _frame_sample(timestamp=5.0, frame_path="data/frames/x/f5.jpg"),
            _frame_sample(timestamp=30.0, frame_path="data/frames/x/f30.jpg"),
            _frame_sample(timestamp=95.0, frame_path="data/frames/x/f95.jpg"),
        ],
    )

    chunks = build_chunks(VIDEO_ID, data_dir=data_dir, chunk_seconds=30)

    assert chunks[0].frame_paths == ["data/frames/x/f5.jpg"]
    assert chunks[1].frame_paths == ["data/frames/x/f30.jpg"]
    # Frame exactly at duration belongs to the final chunk.
    assert chunks[3].frame_paths == ["data/frames/x/f95.jpg"]


# 6. VLM captions attach when overlapping a chunk.
def test_captions_attach_when_overlapping(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_metadata(data_dir)
    _write_transcript(data_dir)
    _write_captions(
        data_dir,
        [
            _vlm_caption(start_time=0.0, end_time=15.0, caption="intro"),
            _vlm_caption(start_time=25.0, end_time=40.0, caption="spans"),
        ],
    )

    chunks = build_chunks(VIDEO_ID, data_dir=data_dir, chunk_seconds=30)

    assert chunks[0].vlm_caption == "intro spans"
    assert chunks[1].vlm_caption == "spans"
    assert chunks[2].vlm_caption == ""


# 7. Missing optional frames/OCR/captions do not fail.
def test_missing_optional_inputs_produce_empty_fields(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_metadata(data_dir)
    _write_transcript(data_dir)

    chunks = build_chunks(VIDEO_ID, data_dir=data_dir, chunk_seconds=30)

    assert chunks
    for c in chunks:
        assert c.ocr_text == ""
        assert c.vlm_caption == ""
        assert c.frame_paths == []


# 8. Missing required transcript fails clearly.
def test_missing_transcript_fails(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_metadata(data_dir)

    with pytest.raises(FileNotFoundError) as exc_info:
        build_chunks(VIDEO_ID, data_dir=data_dir)

    assert "transcript" in str(exc_info.value)
    assert "Run Stage 4 first" in str(exc_info.value)


# 9. Missing required media metadata fails clearly.
def test_missing_media_metadata_fails(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_transcript(data_dir)

    with pytest.raises(FileNotFoundError) as exc_info:
        build_chunks(VIDEO_ID, data_dir=data_dir)

    assert "media metadata" in str(exc_info.value)
    assert "Run Stage 2 first" in str(exc_info.value)


# 10. Invalid chunk_seconds fails clearly.
@pytest.mark.parametrize("chunk_seconds", [0, -1, -0.5])
def test_invalid_chunk_seconds_fails(tmp_path: Path, chunk_seconds):
    data_dir = tmp_path / "data"
    _write_metadata(data_dir)
    _write_transcript(data_dir)

    with pytest.raises(ValueError) as exc_info:
        build_chunks(VIDEO_ID, data_dir=data_dir, chunk_seconds=chunk_seconds)

    assert "chunk_seconds" in str(exc_info.value)


# 11. Invalid overlap_seconds fails clearly.
@pytest.mark.parametrize("overlap_seconds", [-1, 30, 45])
def test_invalid_overlap_seconds_fails(tmp_path: Path, overlap_seconds):
    data_dir = tmp_path / "data"
    _write_metadata(data_dir)
    _write_transcript(data_dir)

    with pytest.raises(ValueError) as exc_info:
        build_chunks(
            VIDEO_ID,
            data_dir=data_dir,
            chunk_seconds=30,
            overlap_seconds=overlap_seconds,
        )

    assert "overlap_seconds" in str(exc_info.value)


# 12. Existing output fails without overwrite.
def test_existing_output_without_overwrite_fails(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_metadata(data_dir)
    _write_transcript(data_dir)
    output_path = data_dir / "chunks" / f"{VIDEO_ID}_30s.jsonl"
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"existing": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_chunks(VIDEO_ID, data_dir=data_dir, chunk_seconds=30)


# 13. Overwrite replaces only the target chunk file.
def test_overwrite_replaces_only_target_file(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_metadata(data_dir)
    _write_transcript(data_dir)
    chunks_dir = data_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    target = chunks_dir / f"{VIDEO_ID}_30s.jsonl"
    target.write_text('{"existing": true}\n', encoding="utf-8")
    # An unrelated chunk file (different chunk size) must survive.
    sibling = chunks_dir / f"{VIDEO_ID}_60s.jsonl"
    sibling.write_text("KEEP ME\n", encoding="utf-8")

    chunks = build_chunks(
        VIDEO_ID, data_dir=data_dir, chunk_seconds=30, overwrite=True
    )

    assert sibling.read_text(encoding="utf-8") == "KEEP ME\n"
    assert list(read_jsonl(target, Chunk)) == chunks


# 14. Output JSONL contains valid Chunk records.
def test_output_jsonl_round_trips(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_metadata(data_dir)
    _write_transcript(
        data_dir, [_transcript_segment(start_time=0.0, end_time=20.0, text="hi")]
    )

    chunks = build_chunks(VIDEO_ID, data_dir=data_dir, chunk_seconds=30)
    output_path = data_dir / "chunks" / f"{VIDEO_ID}_30s.jsonl"

    assert list(read_jsonl(output_path, Chunk)) == chunks
    assert all(isinstance(c, Chunk) for c in chunks)


# Overlap windows use a distinct filename to avoid collisions.
def test_overlap_uses_suffixed_filename(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_metadata(data_dir, _media_metadata(duration_seconds=60.0))
    _write_transcript(data_dir)

    chunks = build_chunks(
        VIDEO_ID, data_dir=data_dir, chunk_seconds=30, overlap_seconds=10
    )

    output_path = data_dir / "chunks" / f"{VIDEO_ID}_30s_overlap10s.jsonl"
    assert output_path.exists()
    # stride = 20 -> starts 0, 20, 40 (60 is not < 60).
    assert [(c.start_time, c.end_time) for c in chunks] == [
        (0.0, 30.0),
        (20.0, 50.0),
        (40.0, 60.0),
    ]
    assert all(c.overlap_seconds == 10.0 for c in chunks)


# 15. CLI success and failure paths.
def test_cli_success(tmp_path: Path, monkeypatch, capsys):
    captured_args = {}

    def fake_build_chunks(video_id, data_dir, chunk_seconds, overlap_seconds, overwrite):
        captured_args.update(
            video_id=video_id,
            data_dir=data_dir,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            overwrite=overwrite,
        )
        return []

    monkeypatch.setattr(build_chunks_module, "build_chunks", fake_build_chunks)

    exit_code = main(
        [
            "--video-id",
            VIDEO_ID,
            "--data-dir",
            str(tmp_path / "data"),
            "--chunk-seconds",
            "30",
            "--overwrite",
        ]
    )

    captured = capsys.readouterr()
    expected_path = tmp_path / "data" / "chunks" / f"{VIDEO_ID}_30s.jsonl"
    assert exit_code == 0
    assert "Wrote chunks:" in captured.out
    assert expected_path.as_posix() in captured.out
    assert captured_args["overwrite"] is True
    assert captured_args["chunk_seconds"] == 30.0


def test_cli_failure_on_missing_transcript(tmp_path: Path, capsys):
    data_dir = tmp_path / "data"
    _write_metadata(data_dir)

    exit_code = main(["--video-id", VIDEO_ID, "--data-dir", str(data_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL" in captured.err
    assert "FileNotFoundError" in captured.err
