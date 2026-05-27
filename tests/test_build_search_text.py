from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_rag.index import build_search_text as build_search_text_module
from video_rag.index.build_search_text import build_search_text, main
from video_rag.io_utils import read_jsonl, write_jsonl
from video_rag.schemas import Chunk


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
        "source_metadata": {"course": "stats"},
    }
    base.update(overrides)
    return base


def _write_chunks(
    data_dir: Path,
    video_id: str = "lecture_001",
    chunk_seconds: int = 30,
    records: list[dict] | None = None,
) -> Path:
    path = data_dir / "chunks" / f"{video_id}_{chunk_seconds}s.jsonl"
    write_jsonl(path, records if records is not None else [_chunk()])
    return path


def test_creates_all_four_text_variants(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_chunks(data_dir)

    records = build_search_text("lecture_001", data_dir=data_dir)

    record = records[0]
    assert record.combined_text_transcript_only == (
        "Transcript:\nBayes theorem updates probability."
    )
    assert record.combined_text_transcript_ocr == (
        "Transcript:\nBayes theorem updates probability.\n\n"
        "On-screen text:\nP(A|B) = P(B|A)P(A)/P(B)"
    )
    assert record.combined_text_transcript_vlm == (
        "Transcript:\nBayes theorem updates probability.\n\n"
        "Visual caption:\nThe instructor points at a theorem slide."
    )
    assert record.combined_text_all == (
        "Transcript:\nBayes theorem updates probability.\n\n"
        "On-screen text:\nP(A|B) = P(B|A)P(A)/P(B)\n\n"
        "Visual caption:\nThe instructor points at a theorem slide."
    )
    assert record.combined_text == record.combined_text_all


def test_text_variants_include_only_requested_modalities(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_chunks(data_dir)

    record = build_search_text("lecture_001", data_dir=data_dir)[0]

    assert "P(A|B)" not in record.combined_text_transcript_only
    assert "instructor points" not in record.combined_text_transcript_only
    assert "instructor points" not in record.combined_text_transcript_ocr
    assert "P(A|B)" not in record.combined_text_transcript_vlm
    assert "Bayes theorem" in record.combined_text_all
    assert "P(A|B)" in record.combined_text_all
    assert "instructor points" in record.combined_text_all


def test_empty_sections_are_omitted_cleanly(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_chunks(
        data_dir,
        records=[
            _chunk(transcript_text="  ", ocr_text="Slide equation", vlm_caption=None),
            _chunk(
                chunk_id="lecture_001_chunk_0001",
                chunk_index=1,
                transcript_text="",
                ocr_text=" \n\t ",
                vlm_caption=None,
            ),
        ],
    )

    records = build_search_text("lecture_001", data_dir=data_dir)

    assert records[0].combined_text_transcript_only == ""
    assert records[0].combined_text_transcript_ocr == "On-screen text:\nSlide equation"
    assert records[0].combined_text_transcript_vlm == ""
    assert records[0].combined_text_all == "On-screen text:\nSlide equation"
    assert records[1].combined_text_transcript_only == ""
    assert records[1].combined_text_transcript_ocr == ""
    assert records[1].combined_text_transcript_vlm == ""
    assert records[1].combined_text_all == ""


def test_whitespace_is_normalized(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_chunks(
        data_dir,
        records=[
            _chunk(
                transcript_text="  Bayes\n\n theorem\t updates   probability. ",
                ocr_text="  P(A|B)\n=\tP(B|A)  ",
                vlm_caption="  Slide\n\nshows   an equation.  ",
            )
        ],
    )

    record = build_search_text("lecture_001", data_dir=data_dir)[0]

    assert record.combined_text_all == (
        "Transcript:\nBayes theorem updates probability.\n\n"
        "On-screen text:\nP(A|B) = P(B|A)\n\n"
        "Visual caption:\nSlide shows an equation."
    )


def test_timestamps_and_metadata_are_preserved(tmp_path: Path):
    data_dir = tmp_path / "data"
    original = _chunk(
        start_time=15.0,
        end_time=45.0,
        frame_paths=["data/frames/lecture_001/frame_000015.jpg"],
        source_metadata={"course": "stats", "week": 2},
    )
    _write_chunks(data_dir, records=[original])

    record = build_search_text("lecture_001", data_dir=data_dir)[0]
    dumped = record.model_dump(mode="json")

    for key, value in original.items():
        assert dumped[key] == value


def test_existing_enriched_output_fails_without_overwrite(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_chunks(data_dir)
    output_path = data_dir / "chunks" / "lecture_001_30s_enriched.jsonl"
    output_path.write_text('{"existing": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_search_text("lecture_001", data_dir=data_dir)


def test_overwrite_behavior_works(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_chunks(data_dir, records=[_chunk(transcript_text="Replacement text")])
    output_path = data_dir / "chunks" / "lecture_001_30s_enriched.jsonl"
    output_path.write_text('{"existing": true}\n', encoding="utf-8")

    records = build_search_text("lecture_001", data_dir=data_dir, overwrite=True)

    assert records[0].combined_text_transcript_only == "Transcript:\nReplacement text"
    assert list(read_jsonl(output_path, Chunk)) == records


def test_missing_input_chunk_file_mentions_stage_9(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as exc_info:
        build_search_text("lecture_001", data_dir=tmp_path / "data")

    assert "Run Stage 9 first" in str(exc_info.value)


def test_output_jsonl_contains_valid_enriched_records(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_chunks(data_dir)

    records = build_search_text("lecture_001", data_dir=data_dir)
    output_path = data_dir / "chunks" / "lecture_001_30s_enriched.jsonl"

    raw_lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 1
    assert json.loads(raw_lines[0])["combined_text_all"] == records[0].combined_text_all
    assert list(read_jsonl(output_path, Chunk)) == records


def test_cli_success(tmp_path: Path, monkeypatch, capsys):
    def fake_build_search_text(video_id, data_dir, chunk_seconds, overwrite):
        assert video_id == "lecture_001"
        assert data_dir == tmp_path / "data"
        assert chunk_seconds == 30.0
        assert overwrite is True
        return [Chunk.model_validate(_chunk())]

    monkeypatch.setattr(
        build_search_text_module,
        "build_search_text",
        fake_build_search_text,
    )

    exit_code = main(
        [
            "--video-id",
            "lecture_001",
            "--data-dir",
            str(tmp_path / "data"),
            "--chunk-seconds",
            "30",
            "--overwrite",
        ]
    )

    captured = capsys.readouterr()
    expected_path = tmp_path / "data" / "chunks" / "lecture_001_30s_enriched.jsonl"
    assert exit_code == 0
    assert "Wrote enriched chunks:" in captured.out
    assert expected_path.as_posix() in captured.out


def test_cli_failure(tmp_path: Path, capsys):
    exit_code = main(
        ["--video-id", "lecture_001", "--data-dir", str(tmp_path / "data")]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL  FileNotFoundError" in captured.err
    assert "Run Stage 9 first" in captured.err
