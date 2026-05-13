from __future__ import annotations

from pathlib import Path

import pytest

from video_rag.validate import main

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"


def test_cli_valid_video_manifest(capsys):
    rc = main([str(EXAMPLES / "video_manifest.example.json"), "--type", "video_manifest"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.startswith("OK")


def test_cli_valid_media_metadata(capsys):
    rc = main([str(EXAMPLES / "media_metadata.example.json"), "--type", "media_metadata"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.startswith("OK")


def test_cli_bad_video_manifest(capsys):
    rc = main([str(EXAMPLES / "bad_video_manifest.example.json"), "--type", "video_manifest"])
    captured = capsys.readouterr()
    assert rc != 0
    assert "FAIL" in captured.out


def test_cli_bad_media_metadata(capsys):
    rc = main([str(EXAMPLES / "bad_media_metadata.example.json"), "--type", "media_metadata"])
    captured = capsys.readouterr()
    assert rc != 0
    assert "FAIL" in captured.out


def test_cli_missing_file(capsys, tmp_path: Path):
    missing = tmp_path / "nope.json"
    rc = main([str(missing), "--type", "video_manifest"])
    captured = capsys.readouterr()
    assert rc != 0
    assert "FAIL" in captured.out


def test_cli_unknown_type_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        main(["whatever.json", "--type", "not_a_real_type"])
    assert exc.value.code != 0


# --- transcript_segments (JSONL) --------------------------------------------


def test_cli_valid_transcript_segments(capsys):
    rc = main(
        [
            str(EXAMPLES / "transcript_segment.example.jsonl"),
            "--type",
            "transcript_segments",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.startswith("OK")
    assert "3 segments" in captured.out


def test_cli_bad_transcript_segments_reports_line_number(capsys):
    rc = main(
        [
            str(EXAMPLES / "bad_transcript_segment.example.jsonl"),
            "--type",
            "transcript_segments",
        ]
    )
    captured = capsys.readouterr()
    assert rc != 0
    assert "FAIL" in captured.out
    assert ":2:" in captured.out


def test_cli_empty_transcript_jsonl_fails(tmp_path: Path, capsys):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    rc = main([str(empty), "--type", "transcript_segments"])
    captured = capsys.readouterr()
    assert rc != 0
    assert "empty" in captured.out


def test_cli_transcript_mixed_video_id_fails(tmp_path: Path, capsys):
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(
        '{"video_id": "lecture_001", "start_time": 0.0, "end_time": 1.0, "text": "a"}\n'
        '{"video_id": "lecture_002", "start_time": 1.0, "end_time": 2.0, "text": "b"}\n',
        encoding="utf-8",
    )
    rc = main([str(mixed), "--type", "transcript_segments"])
    captured = capsys.readouterr()
    assert rc != 0
    assert "mixed video_id" in captured.out


def test_cli_transcript_out_of_order_start_time_fails(tmp_path: Path, capsys):
    out_of_order = tmp_path / "ooo.jsonl"
    out_of_order.write_text(
        '{"video_id": "v", "start_time": 5.0, "end_time": 6.0, "text": "a"}\n'
        '{"video_id": "v", "start_time": 1.0, "end_time": 2.0, "text": "b"}\n',
        encoding="utf-8",
    )
    rc = main([str(out_of_order), "--type", "transcript_segments"])
    captured = capsys.readouterr()
    assert rc != 0
    assert "before previous start_time" in captured.out


def test_cli_transcript_overlap_allowed(tmp_path: Path, capsys):
    overlap = tmp_path / "overlap.jsonl"
    overlap.write_text(
        '{"video_id": "v", "start_time": 0.0, "end_time": 2.0, "text": "a"}\n'
        '{"video_id": "v", "start_time": 1.5, "end_time": 3.0, "text": "b"}\n',
        encoding="utf-8",
    )
    rc = main([str(overlap), "--type", "transcript_segments"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.startswith("OK")
