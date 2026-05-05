from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from video_rag.index import sample_frames as sf
from video_rag.io_utils import read_jsonl, write_json  # noqa: F401  (used via helpers + tests)
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


def _make_fake_binary(tmp_path: Path) -> Path:
    """A real on-disk executable so _resolve_binary accepts it. The body is
    irrelevant because subprocess.run is monkeypatched in every test that
    invokes the orchestrator's full path."""
    binary = tmp_path / "fake_extractor"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary


def _seed(
    tmp_path: Path,
    *,
    video_id: str = "lecture_001",
    duration_seconds: float = 30.0,
    write_video: bool = True,
) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    _write_video_manifest(data_dir, video_id=video_id)
    _write_media_metadata(data_dir, video_id=video_id, duration_seconds=duration_seconds)
    if write_video:
        _write_video_file(data_dir, video_id=video_id)
    binary = _make_fake_binary(tmp_path)
    return data_dir, binary


def _fake_completed(returncode: int, stdout: str = "", stderr: str = "") -> Any:
    return subprocess.CompletedProcess(
        args=["raggers_frame_extract"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _make_extractor_stdout(out_dir: Path, timestamps: list[float]) -> str:
    lines = []
    for ts in timestamps:
        rec = {
            "timestamp": ts,
            "frame_path": str(out_dir.resolve() / f"frame_{int(round(ts)):06d}.jpg"),
            "width": 1920,
            "height": 1080,
        }
        lines.append(json.dumps(rec))
    return "\n".join(lines) + ("\n" if lines else "")


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
    data_dir, binary = _seed(tmp_path, duration_seconds=15.0)
    out_dir = data_dir / "frames" / "lecture_001"

    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _fake_completed(
            0, stdout=_make_extractor_stdout(out_dir, [0.0, 5.0, 10.0])
        )

    monkeypatch.setattr(sf.subprocess, "run", fake_run)

    samples = sf.sample_frames(
        video_id="lecture_001",
        data_dir=data_dir,
        interval_seconds=5,
        binary_path=binary,
    )

    assert len(samples) == 3
    assert all(isinstance(s, FrameSample) for s in samples)
    assert [s.timestamp for s in samples] == [0.0, 5.0, 10.0]
    assert all(s.video_id == "lecture_001" for s in samples)
    assert all(s.sampling_method == "fixed_interval" for s in samples)
    assert all(s.thumbnail_path is None for s in samples)

    cmd = captured["cmd"]
    assert cmd[0] == str(binary.resolve())
    assert "--video" in cmd
    assert "--out-dir" in cmd
    assert "--timestamps" in cmd
    timestamps_arg = cmd[cmd.index("--timestamps") + 1]
    assert timestamps_arg == "0.000,5.000,10.000"
    assert "--quality" in cmd
    assert cmd[cmd.index("--quality") + 1] == "85"

    manifest_out = out_dir / "frame_manifest.jsonl"
    loaded = list(read_jsonl(manifest_out, FrameSample))
    assert loaded == samples


def test_frame_paths_are_repo_root_relative(tmp_path: Path, monkeypatch):
    data_dir, binary = _seed(tmp_path, duration_seconds=10.0)
    out_dir = data_dir / "frames" / "lecture_001"

    monkeypatch.setattr(
        sf.subprocess,
        "run",
        lambda cmd, **kw: _fake_completed(
            0, stdout=_make_extractor_stdout(out_dir, [0.0, 5.0])
        ),
    )

    samples = sf.sample_frames(
        video_id="lecture_001",
        data_dir=data_dir,
        interval_seconds=5,
        binary_path=binary,
    )

    expected = f"{data_dir.name}/frames/lecture_001/frame_{0:06d}.jpg"
    assert samples[0].frame_path == expected
    assert samples[1].frame_path.endswith(f"/frames/lecture_001/frame_{5:06d}.jpg")
    for s in samples:
        assert "\\" not in s.frame_path
        assert not Path(s.frame_path).is_absolute()


def test_short_video_samples_only_zero(tmp_path: Path, monkeypatch):
    data_dir, binary = _seed(tmp_path, duration_seconds=0.5)
    out_dir = data_dir / "frames" / "lecture_001"

    monkeypatch.setattr(
        sf.subprocess,
        "run",
        lambda cmd, **kw: _fake_completed(0, stdout=_make_extractor_stdout(out_dir, [0.0])),
    )

    samples = sf.sample_frames(
        video_id="lecture_001",
        data_dir=data_dir,
        interval_seconds=5,
        binary_path=binary,
    )
    assert len(samples) == 1
    assert samples[0].timestamp == 0.0


def test_overwrite_replaces_prior_frames(tmp_path: Path, monkeypatch):
    data_dir, binary = _seed(tmp_path, duration_seconds=10.0)
    out_dir = data_dir / "frames" / "lecture_001"
    out_dir.mkdir(parents=True)
    (out_dir / "stale.jpg").write_bytes(b"stale")
    (out_dir / "frame_manifest.jsonl").write_text("stale", encoding="utf-8")

    monkeypatch.setattr(
        sf.subprocess,
        "run",
        lambda cmd, **kw: _fake_completed(
            0, stdout=_make_extractor_stdout(out_dir, [0.0, 5.0])
        ),
    )

    samples = sf.sample_frames(
        video_id="lecture_001",
        data_dir=data_dir,
        interval_seconds=5,
        overwrite=True,
        binary_path=binary,
    )
    assert len(samples) == 2
    assert not (out_dir / "stale.jpg").exists()


def test_returns_records_in_order(tmp_path: Path, monkeypatch):
    data_dir, binary = _seed(tmp_path, duration_seconds=20.0)
    out_dir = data_dir / "frames" / "lecture_001"
    monkeypatch.setattr(
        sf.subprocess,
        "run",
        lambda cmd, **kw: _fake_completed(
            0, stdout=_make_extractor_stdout(out_dir, [0.0, 5.0, 10.0, 15.0])
        ),
    )
    samples = sf.sample_frames(
        video_id="lecture_001",
        data_dir=data_dir,
        interval_seconds=5,
        binary_path=binary,
    )
    assert [s.timestamp for s in samples] == [0.0, 5.0, 10.0, 15.0]


# --- failure paths ----------------------------------------------------------


def test_missing_video_manifest(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_media_metadata(data_dir)
    _write_video_file(data_dir)
    binary = _make_fake_binary(tmp_path)
    with pytest.raises(FileNotFoundError, match="video_manifest.json"):
        sf.sample_frames(
            video_id="lecture_001",
            data_dir=data_dir,
            interval_seconds=5,
            binary_path=binary,
        )


def test_missing_media_metadata(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_video_manifest(data_dir)
    _write_video_file(data_dir)
    binary = _make_fake_binary(tmp_path)
    with pytest.raises(FileNotFoundError, match="media_metadata.json"):
        sf.sample_frames(
            video_id="lecture_001",
            data_dir=data_dir,
            interval_seconds=5,
            binary_path=binary,
        )


def test_missing_video_file(tmp_path: Path):
    data_dir, binary = _seed(tmp_path, write_video=False)
    with pytest.raises(FileNotFoundError, match="registered video file not found"):
        sf.sample_frames(
            video_id="lecture_001",
            data_dir=data_dir,
            interval_seconds=5,
            binary_path=binary,
        )


@pytest.mark.parametrize("bad", [0, -1, -5, 0.5, "5", None])
def test_invalid_interval(tmp_path: Path, bad):
    data_dir, binary = _seed(tmp_path)
    with pytest.raises(ValueError):
        sf.sample_frames(
            video_id="lecture_001",
            data_dir=data_dir,
            interval_seconds=bad,
            binary_path=binary,
        )


def test_existing_outputs_without_overwrite(tmp_path: Path, monkeypatch):
    data_dir, binary = _seed(tmp_path)
    out_dir = data_dir / "frames" / "lecture_001"
    out_dir.mkdir(parents=True)
    (out_dir / "frame_manifest.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        sf.subprocess,
        "run",
        lambda cmd, **kw: pytest.fail("extractor must not be called"),
    )
    with pytest.raises(FileExistsError):
        sf.sample_frames(
            video_id="lecture_001",
            data_dir=data_dir,
            interval_seconds=5,
            binary_path=binary,
        )


def test_extractor_nonzero_exit_raises(tmp_path: Path, monkeypatch):
    data_dir, binary = _seed(tmp_path)
    monkeypatch.setattr(
        sf.subprocess,
        "run",
        lambda cmd, **kw: _fake_completed(7, stdout="", stderr="bad video"),
    )
    with pytest.raises(RuntimeError, match="exited with code 7"):
        sf.sample_frames(
            video_id="lecture_001",
            data_dir=data_dir,
            interval_seconds=5,
            binary_path=binary,
        )


def test_extractor_stdout_invalid_json_raises(tmp_path: Path, monkeypatch):
    data_dir, binary = _seed(tmp_path)
    monkeypatch.setattr(
        sf.subprocess,
        "run",
        lambda cmd, **kw: _fake_completed(0, stdout="not json\n"),
    )
    with pytest.raises(RuntimeError, match="invalid JSON"):
        sf.sample_frames(
            video_id="lecture_001",
            data_dir=data_dir,
            interval_seconds=5,
            binary_path=binary,
        )


def test_binary_missing_with_no_env(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    _write_video_manifest(data_dir)
    _write_media_metadata(data_dir)
    _write_video_file(data_dir)
    monkeypatch.delenv(sf.BINARY_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="raggers_frame_extract binary not found"):
        sf.sample_frames(
            video_id="lecture_001",
            data_dir=data_dir,
            interval_seconds=5,
        )


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
    binary = _make_fake_binary(tmp_path)
    with pytest.raises(ValueError, match="video_id mismatch"):
        sf.sample_frames(
            video_id="lecture_001",
            data_dir=data_dir,
            interval_seconds=5,
            binary_path=binary,
        )


def test_empty_video_id():
    with pytest.raises(ValueError, match="video_id"):
        sf.sample_frames(video_id="", data_dir="data", interval_seconds=5)


# --- CLI --------------------------------------------------------------------


def test_cli_happy_path(tmp_path: Path, monkeypatch, capsys):
    data_dir, binary = _seed(tmp_path, duration_seconds=15.0)
    out_dir = data_dir / "frames" / "lecture_001"
    monkeypatch.setattr(
        sf.subprocess,
        "run",
        lambda cmd, **kw: _fake_completed(
            0, stdout=_make_extractor_stdout(out_dir, [0.0, 5.0, 10.0])
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
            "--binary-path",
            str(binary),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "Sampled 3 frame(s)" in captured.out
    assert (out_dir / "frame_manifest.jsonl").is_file()


def test_cli_failure_returns_one(tmp_path: Path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    _write_media_metadata(data_dir)
    _write_video_file(data_dir)
    binary = _make_fake_binary(tmp_path)
    rc = sf.main(
        [
            "--video-id",
            "lecture_001",
            "--data-dir",
            str(data_dir),
            "--interval-seconds",
            "5",
            "--binary-path",
            str(binary),
        ]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "FAIL" in captured.err


def test_cli_invalid_interval_returns_one(tmp_path: Path, capsys):
    data_dir, binary = _seed(tmp_path)
    rc = sf.main(
        [
            "--video-id",
            "lecture_001",
            "--data-dir",
            str(data_dir),
            "--interval-seconds",
            "0",
            "--binary-path",
            str(binary),
        ]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "FAIL" in captured.err
