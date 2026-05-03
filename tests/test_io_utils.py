from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_rag.io_utils import read_json, read_jsonl, write_json, write_jsonl
from video_rag.schemas import MediaMetadata, VideoManifest


def _manifest(**overrides) -> VideoManifest:
    base = dict(
        video_id="lecture_001",
        title="Bayes Theorem Lecture",
        source_path="data/videos/lecture_001.mp4",
    )
    base.update(overrides)
    return VideoManifest(**base)


def test_json_roundtrip(tmp_path: Path):
    out = tmp_path / "nested" / "manifest.json"
    original = _manifest(original_filename="lecture final.mp4")

    write_json(out, original)
    assert out.exists(), "writer should create parent directories"

    loaded = read_json(out, VideoManifest)
    assert loaded == original


def test_json_read_validation_error(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"video_id": "", "title": "t", "source_path": "p"}), encoding="utf-8")

    with pytest.raises(ValueError):
        read_json(p, VideoManifest)


def test_jsonl_roundtrip(tmp_path: Path):
    out = tmp_path / "manifests.jsonl"
    records = [
        _manifest(video_id="v1", title="t1", source_path="p1"),
        _manifest(video_id="v2", title="t2", source_path="p2"),
        _manifest(video_id="v3", title="t3", source_path="p3"),
    ]

    write_jsonl(out, records)

    loaded = list(read_jsonl(out, VideoManifest))
    assert loaded == records


def test_jsonl_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "withblanks.jsonl"
    rec = {"video_id": "v", "duration_seconds": 1.0, "has_audio": True}
    p.write_text(
        "\n"
        f"{json.dumps(rec)}\n"
        "\n"
        f"{json.dumps(rec)}\n"
        "   \n",
        encoding="utf-8",
    )

    loaded = list(read_jsonl(p, MediaMetadata))
    assert len(loaded) == 2


def test_jsonl_invalid_line_reports_line_number(tmp_path: Path):
    p = tmp_path / "broken.jsonl"
    good = json.dumps({"video_id": "v", "duration_seconds": 1.0, "has_audio": True})
    p.write_text(good + "\n" + "{not json\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        list(read_jsonl(p, MediaMetadata))
    msg = str(exc_info.value)
    assert ":2:" in msg
    assert str(p) in msg


def test_jsonl_validation_error_reports_line_number(tmp_path: Path):
    p = tmp_path / "invalid.jsonl"
    good = json.dumps({"video_id": "v", "duration_seconds": 1.0, "has_audio": True})
    bad = json.dumps({"video_id": "", "duration_seconds": 1.0, "has_audio": True})
    p.write_text(good + "\n" + good + "\n" + bad + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        list(read_jsonl(p, MediaMetadata))
    msg = str(exc_info.value)
    assert ":3:" in msg


def test_write_json_accepts_dict(tmp_path: Path):
    p = tmp_path / "raw.json"
    write_json(p, {"video_id": "v", "title": "t", "source_path": "p"})

    loaded = read_json(p, VideoManifest)
    assert loaded.video_id == "v"
