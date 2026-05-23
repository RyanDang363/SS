from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from video_rag.index import sample_frames as sf
from video_rag.io_utils import read_jsonl, write_json
from video_rag.schemas import FrameSample, MediaMetadata, VideoManifest


# --- helpers ----------------------------------------------------------------


def _write_video_manifest(data_dir: Path, video_id: str = "lecture_001") -> Path:
    manifest = VideoManifest(
        video_id=video_id,
        title="t",
        source_path=f"{data_dir.name}/videos/{video_id}.mp4",
        original_filename=f"{video_id}.mp4",
        created_at="2026-05-03T12:00:00Z",
    )
    out = data_dir / "manifests" / video_id / "video_manifest.json"
    write_json(out, manifest)
    return out


def _write_media_metadata(
    data_dir: Path,
    video_id: str = "lecture_001",
    duration_seconds: float = 30.0,
) -> Path:
    metadata = MediaMetadata(
        video_id=video_id,
        duration_seconds=duration_seconds,
        fps=30.0,
        width=1920,
        height=1080,
        has_audio=True,
    )
    out = data_dir / "manifests" / video_id / "media_metadata.json"
    write_json(out, metadata)
    return out


def _write_video_file(
    data_dir: Path, video_id: str = "lecture_001", payload: bytes = b"fake mp4"
) -> Path:
    out = data_dir / "videos" / f"{video_id}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    return out


def _seed(
    tmp_path: Path,
    *,
    video_id: str = "lecture_001",
    duration_seconds: float = 30.0,
    write_video: bool = True,
) -> Path:
    """Set up Stage 1 + 2 artifacts. Returns data_dir."""
    data_dir = tmp_path / "data"
    _write_video_manifest(data_dir, video_id=video_id)
    _write_media_metadata(data_dir, video_id=video_id, duration_seconds=duration_seconds)
    if write_video:
        _write_video_file(data_dir, video_id=video_id)
    return data_dir


def _fake_records(out_dir: Path, timestamps: list[float]) -> list[dict]:
    """Return fake _extract_frames output for the given timestamps."""
    return [
        {
            "timestamp": ts,
            "frame_path": str(out_dir.resolve() / f"frame_{int(round(ts)):06d}.jpg"),
            "width": 1920,
            "height": 1080,
        }
        for ts in timestamps
    ]


# --- pure helpers -----------------------------------------------------------


@pytest.mark.parametrize(
    "duration, interval, expected",
    [
        (30.0, 5, [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]),
        (10.0, 5, [0.0, 5.0]),
        (10.0, 3, [0.0, 3.0, 6.0, 9.0]),
        (1.0, 5, [0.0]),
        (5.0, 5, [0.0]),
        (5.5, 5, [0.0, 5.0]),
        (0.5, 5, [0.0]),
        (0.0, 5, []),
    ],
)
def test_compute_schedule(duration, interval, expected):
    assert sf._compute_schedule(duration, interval) == expected


@pytest.mark.parametrize("bad", [0, -1, -0.5, 0.5, 1.5, "5", None, True, False])
def test_coerce_positive_int_interval_rejects(bad):
    with pytest.raises(ValueError):
        sf._coerce_positive_int_interval(bad)


@pytest.mark.parametrize("good, expected", [(1, 1), (5, 5), (5.0, 5), (10, 10)])
def test_coerce_positive_int_interval_accepts(good, expected):
    assert sf._coerce_positive_int_interval(good) == expected


# --- happy path -------------------------------------------------------------


def test_successful_sampling(tmp_path: Path, monkeypatch):
    data_dir = _seed(tmp_path, duration_seconds=15.0)
    out_dir = data_dir / "frames" / "lecture_001"

    captured: dict[str, Any] = {}

    def fake_extract(video_path, out_dir, timestamps, jpeg_quality):
        captured["video_path"] = video_path
        captured["out_dir"] = out_dir
        captured["timestamps"] = timestamps
        captured["jpeg_quality"] = jpeg_quality
        return _fake_records(out_dir, timestamps)

    monkeypatch.setattr(sf, "_extract_frames", fake_extract)

    samples = sf.sample_frames(
        video_id="lecture_001",
        data_dir=data_dir,
        interval_seconds=5,
    )

    assert len(samples) == 3
    assert all(isinstance(s, FrameSample) for s in samples)
    assert [s.timestamp for s in samples] == [0.0, 5.0, 10.0]
    assert all(s.video_id == "lecture_001" for s in samples)
    assert all(s.sampling_method == "fixed_interval" for s in samples)
    assert all(s.thumbnail_path is None for s in samples)

    assert captured["timestamps"] == [0.0, 5.0, 10.0]
    assert captured["jpeg_quality"] == 85
    assert captured["video_path"].is_file()

    manifest_out = out_dir / "frame_manifest.jsonl"
    loaded = list(read_jsonl(manifest_out, FrameSample))
    assert loaded == samples


def test_frame_paths_are_repo_root_relative(tmp_path: Path, monkeypatch):
    data_dir = _seed(tmp_path, duration_seconds=10.0)

    monkeypatch.setattr(
        sf,
        "_extract_frames",
        lambda video_path, out_dir, timestamps, jpeg_quality: _fake_records(
            out_dir, timestamps
        ),
    )

    samples = sf.sample_frames(
        video_id="lecture_001",
        data_dir=data_dir,
        interval_seconds=5,
    )

    for s in samples:
        assert not Path(s.frame_path).is_absolute()
        assert "\\" not in s.frame_path


def test_short_video_samples_only_zero(tmp_path: Path, monkeypatch):
    data_dir = _seed(tmp_path, duration_seconds=0.5)

    monkeypatch.setattr(
        sf,
        "_extract_frames",
        lambda video_path, out_dir, timestamps, jpeg_quality: _fake_records(
            out_dir, timestamps
        ),
    )

    samples = sf.sample_frames(
        video_id="lecture_001",
        data_dir=data_dir,
        interval_seconds=5,
    )
    assert len(samples) == 1
    assert samples[0].timestamp == 0.0


def test_overwrite_replaces_prior_frames(tmp_path: Path, monkeypatch):
    data_dir = _seed(tmp_path, duration_seconds=10.0)
    out_dir = data_dir / "frames" / "lecture_001"
    out_dir.mkdir(parents=True)
    (out_dir / "stale.jpg").write_bytes(b"stale")
    (out_dir / "frame_manifest.jsonl").write_text("stale", encoding="utf-8")

    monkeypatch.setattr(
        sf,
        "_extract_frames",
        lambda video_path, out_dir, timestamps, jpeg_quality: _fake_records(
            out_dir, timestamps
        ),
    )

    samples = sf.sample_frames(
        video_id="lecture_001",
        data_dir=data_dir,
        interval_seconds=5,
        overwrite=True,
    )
    assert len(samples) == 2
    assert not (out_dir / "stale.jpg").exists()


def test_returns_records_in_order(tmp_path: Path, monkeypatch):
    data_dir = _seed(tmp_path, duration_seconds=20.0)

    monkeypatch.setattr(
        sf,
        "_extract_frames",
        lambda video_path, out_dir, timestamps, jpeg_quality: _fake_records(
            out_dir, timestamps
        ),
    )

    samples = sf.sample_frames(
        video_id="lecture_001",
        data_dir=data_dir,
        interval_seconds=5,
    )
    assert [s.timestamp for s in samples] == [0.0, 5.0, 10.0, 15.0]


# --- failure paths ----------------------------------------------------------


def test_missing_video_manifest(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_media_metadata(data_dir)
    _write_video_file(data_dir)
    with pytest.raises(FileNotFoundError, match="video_manifest.json"):
        sf.sample_frames(video_id="lecture_001", data_dir=data_dir, interval_seconds=5)


def test_missing_media_metadata(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_video_manifest(data_dir)
    _write_video_file(data_dir)
    with pytest.raises(FileNotFoundError, match="media_metadata.json"):
        sf.sample_frames(video_id="lecture_001", data_dir=data_dir, interval_seconds=5)


def test_missing_video_file(tmp_path: Path):
    data_dir = _seed(tmp_path, write_video=False)
    with pytest.raises(FileNotFoundError, match="registered video file not found"):
        sf.sample_frames(video_id="lecture_001", data_dir=data_dir, interval_seconds=5)


@pytest.mark.parametrize("bad", [0, -1, -5, 0.5, "5", None])
def test_invalid_interval(tmp_path: Path, bad):
    data_dir = _seed(tmp_path)
    with pytest.raises(ValueError):
        sf.sample_frames(video_id="lecture_001", data_dir=data_dir, interval_seconds=bad)


def test_existing_outputs_without_overwrite(tmp_path: Path):
    data_dir = _seed(tmp_path)
    out_dir = data_dir / "frames" / "lecture_001"
    out_dir.mkdir(parents=True)
    (out_dir / "frame_manifest.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        sf.sample_frames(video_id="lecture_001", data_dir=data_dir, interval_seconds=5)


def test_extract_frames_error_propagates(tmp_path: Path, monkeypatch):
    """RuntimeError from _extract_frames surfaces to the caller."""
    data_dir = _seed(tmp_path)

    def raise_error(*args, **kwargs):
        raise RuntimeError("cv2 failed to decode")

    monkeypatch.setattr(sf, "_extract_frames", raise_error)

    with pytest.raises(RuntimeError, match="cv2 failed to decode"):
        sf.sample_frames(video_id="lecture_001", data_dir=data_dir, interval_seconds=5)


def test_video_id_mismatch(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_video_manifest(data_dir, video_id="lecture_001")
    _write_video_file(data_dir, video_id="lecture_001")
    bad_metadata = MediaMetadata(
        video_id="not_lecture_001",
        duration_seconds=30.0,
        has_audio=True,
    )
    write_json(
        data_dir / "manifests" / "lecture_001" / "media_metadata.json", bad_metadata
    )
    with pytest.raises(ValueError, match="video_id mismatch"):
        sf.sample_frames(video_id="lecture_001", data_dir=data_dir, interval_seconds=5)


def test_empty_video_id():
    with pytest.raises(ValueError, match="video_id"):
        sf.sample_frames(video_id="", data_dir="data", interval_seconds=5)


# --- CLI --------------------------------------------------------------------


def test_cli_happy_path(tmp_path: Path, monkeypatch, capsys):
    data_dir = _seed(tmp_path, duration_seconds=15.0)

    monkeypatch.setattr(
        sf,
        "_extract_frames",
        lambda video_path, out_dir, timestamps, jpeg_quality: _fake_records(
            out_dir, timestamps
        ),
    )

    rc = sf.main(
        [
            "--video-id",
            "lecture_001",
            "--data-dir",
            str(data_dir),
            "--interval-seconds",
            "5",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Sampled 3 frame(s)" in out
    assert (data_dir / "frames" / "lecture_001" / "frame_manifest.jsonl").is_file()


def test_cli_failure_returns_one(tmp_path: Path, capsys):
    data_dir = tmp_path / "data"
    _write_media_metadata(data_dir)
    _write_video_file(data_dir)
    rc = sf.main(
        [
            "--video-id",
            "lecture_001",
            "--data-dir",
            str(data_dir),
            "--interval-seconds",
            "5",
        ]
    )
    assert rc == 1
    assert "FAIL" in capsys.readouterr().err


def test_cli_invalid_interval_returns_one(tmp_path: Path, capsys):
    data_dir = _seed(tmp_path)
    rc = sf.main(
        [
            "--video-id",
            "lecture_001",
            "--data-dir",
            str(data_dir),
            "--interval-seconds",
            "0",
        ]
    )
    assert rc == 1
    assert "FAIL" in capsys.readouterr().err
