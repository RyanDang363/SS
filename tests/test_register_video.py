from __future__ import annotations

from pathlib import Path

import pytest

from video_rag.index.register_video import (
    _derive_video_id,
    register_video,
    sanitize_video_id,
)
from video_rag.io_utils import read_json
from video_rag.schemas import VideoManifest


def _make_fake_video(path: Path, payload: bytes = b"fake mp4 bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


# --- helpers / pure functions -----------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Bayes Theorem Lecture", "bayes_theorem_lecture"),
        ("My Lecture!! 01", "my_lecture_01"),
        ("lecture final v2", "lecture_final_v2"),
        ("  spaced  out  ", "spaced_out"),
        ("Already-Valid_ID", "already-valid_id"),
        ("___leading_trailing___", "leading_trailing"),
    ],
)
def test_sanitize_video_id(raw, expected):
    assert sanitize_video_id(raw) == expected


def test_sanitize_to_empty():
    assert sanitize_video_id("!!!") == ""
    assert sanitize_video_id("   ") == ""


def test_derive_from_filename_when_no_title():
    assert _derive_video_id(None, "lecture final v2.mp4") == "lecture_final_v2"


# --- main register_video flow -----------------------------------------------


def test_registers_with_explicit_video_id(tmp_path: Path):
    src = _make_fake_video(tmp_path / "src" / "lecture.mp4")
    data_dir = tmp_path / "data"

    manifest = register_video(
        video_path=src,
        title="Bayes Lecture",
        video_id="lecture_001",
        data_dir=data_dir,
    )

    assert manifest.video_id == "lecture_001"
    assert manifest.title == "Bayes Lecture"
    assert manifest.original_filename == "lecture.mp4"
    assert manifest.source_path == "data/videos/lecture_001.mp4"
    assert manifest.created_at and manifest.created_at.endswith("Z")

    dest_video = data_dir / "videos" / "lecture_001.mp4"
    dest_manifest = data_dir / "manifests" / "lecture_001" / "video_manifest.json"
    assert dest_video.is_file()
    assert dest_manifest.is_file()
    assert dest_video.read_bytes() == src.read_bytes()

    loaded = read_json(dest_manifest, VideoManifest)
    assert loaded == manifest


def test_video_id_derived_from_title(tmp_path: Path):
    src = _make_fake_video(tmp_path / "src" / "anything.mp4")

    manifest = register_video(
        video_path=src,
        title="Bayes Theorem Lecture",
        data_dir=tmp_path / "data",
    )

    assert manifest.video_id == "bayes_theorem_lecture"
    assert (tmp_path / "data" / "videos" / "bayes_theorem_lecture.mp4").is_file()


def test_video_id_derived_from_filename_when_no_title(tmp_path: Path):
    src = _make_fake_video(tmp_path / "src" / "lecture final v2.mp4")

    manifest = register_video(video_path=src, data_dir=tmp_path / "data")

    assert manifest.video_id == "lecture_final_v2"
    # title falls back to the filename stem (must be non-empty per schema)
    assert manifest.title == "lecture final v2"


def test_duplicate_id_gets_numeric_suffix(tmp_path: Path):
    src1 = _make_fake_video(tmp_path / "src" / "a.mp4", b"a")
    src2 = _make_fake_video(tmp_path / "src" / "b.mp4", b"b")
    src3 = _make_fake_video(tmp_path / "src" / "c.mp4", b"c")
    data_dir = tmp_path / "data"

    m1 = register_video(video_path=src1, title="Bayes Theorem Lecture", data_dir=data_dir)
    m2 = register_video(video_path=src2, title="Bayes Theorem Lecture", data_dir=data_dir)
    m3 = register_video(video_path=src3, title="Bayes Theorem Lecture", data_dir=data_dir)

    assert m1.video_id == "bayes_theorem_lecture"
    assert m2.video_id == "bayes_theorem_lecture_2"
    assert m3.video_id == "bayes_theorem_lecture_3"


def test_fails_when_input_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        register_video(
            video_path=tmp_path / "does_not_exist.mp4",
            title="x",
            data_dir=tmp_path / "data",
        )


def test_fails_when_destination_exists_and_no_overwrite(tmp_path: Path):
    src = _make_fake_video(tmp_path / "src" / "a.mp4")
    data_dir = tmp_path / "data"

    register_video(video_path=src, title="t", video_id="vid", data_dir=data_dir)

    with pytest.raises(FileExistsError):
        register_video(video_path=src, title="t", video_id="vid", data_dir=data_dir)


def test_overwrite_replaces_existing(tmp_path: Path):
    src1 = _make_fake_video(tmp_path / "src" / "v1.mp4", b"first")
    src2 = _make_fake_video(tmp_path / "src" / "v2.mp4", b"second")
    data_dir = tmp_path / "data"

    register_video(video_path=src1, title="t1", video_id="vid", data_dir=data_dir)
    manifest = register_video(
        video_path=src2,
        title="t2",
        video_id="vid",
        overwrite=True,
        data_dir=data_dir,
    )

    dest = data_dir / "videos" / "vid.mp4"
    assert dest.read_bytes() == b"second"
    assert manifest.title == "t2"
    loaded = read_json(
        data_dir / "manifests" / "vid" / "video_manifest.json", VideoManifest
    )
    assert loaded.title == "t2"


def test_symlink_mode(tmp_path: Path):
    src = _make_fake_video(tmp_path / "src" / "lecture.mp4", b"linked-bytes")
    data_dir = tmp_path / "data"

    try:
        register_video(
            video_path=src,
            title="t",
            video_id="vid",
            mode="symlink",
            data_dir=data_dir,
        )
    except OSError as e:
        pytest.skip(f"symlinks not supported in this environment: {e}")

    dest = data_dir / "videos" / "vid.mp4"
    assert dest.is_symlink()
    assert dest.resolve() == src.resolve()
    assert dest.read_bytes() == b"linked-bytes"


def test_sanitizes_weird_explicit_id(tmp_path: Path):
    src = _make_fake_video(tmp_path / "src" / "a.mp4")

    manifest = register_video(
        video_path=src,
        title="t",
        video_id="My Lecture!! 01",
        data_dir=tmp_path / "data",
    )
    assert manifest.video_id == "my_lecture_01"


def test_rejects_id_that_sanitizes_to_empty(tmp_path: Path):
    src = _make_fake_video(tmp_path / "src" / "a.mp4")

    with pytest.raises(ValueError):
        register_video(
            video_path=src,
            title="t",
            video_id="!!!",
            data_dir=tmp_path / "data",
        )
