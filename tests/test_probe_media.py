from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from video_rag.index import probe_media as probe_media_module
from video_rag.index.probe_media import _parse_fps, main, probe_media
from video_rag.io_utils import read_json, write_json
from video_rag.schemas import MediaMetadata, VideoManifest


def _make_fake_video(path: Path, payload: bytes = b"fake video bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_manifest(
    data_dir: Path,
    video_id: str = "lecture_001",
    source_path: str = "data/videos/lecture_001.mp4",
) -> Path:
    manifest_path = data_dir / "manifests" / video_id / "video_manifest.json"
    write_json(
        manifest_path,
        VideoManifest(
            video_id=video_id,
            title="Lecture 001",
            source_path=source_path,
            original_filename="lecture_001.mp4",
            created_at="2026-05-05T00:00:00Z",
        ),
    )
    return manifest_path


def _ffprobe_output(
    *,
    duration: str = "1840.5",
    avg_frame_rate: str = "30000/1001",
    r_frame_rate: str = "30/1",
    has_audio: bool = True,
) -> str:
    streams = [
        {
            "codec_type": "video",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": avg_frame_rate,
            "r_frame_rate": r_frame_rate,
        }
    ]
    if has_audio:
        streams.append({"codec_type": "audio"})
    return json.dumps({"format": {"duration": duration}, "streams": streams})


def test_probe_media_success_with_mocked_ffprobe(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    video_path = _make_fake_video(data_dir / "videos" / "lecture_001.mp4")
    manifest_path = _write_manifest(data_dir)
    manifest_before = manifest_path.read_bytes()
    calls = []

    def fake_run(cmd, check, capture_output, text):
        calls.append(
            {
                "cmd": cmd,
                "check": check,
                "capture_output": capture_output,
                "text": text,
            }
        )
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=_ffprobe_output(),
            stderr="",
        )

    monkeypatch.setattr(probe_media_module.subprocess, "run", fake_run)

    metadata = probe_media("lecture_001", data_dir=data_dir)

    output_path = data_dir / "manifests" / "lecture_001" / "media_metadata.json"
    assert metadata == MediaMetadata(
        video_id="lecture_001",
        duration_seconds=1840.5,
        fps=30000 / 1001,
        width=1920,
        height=1080,
        has_audio=True,
    )
    assert read_json(output_path, MediaMetadata) == metadata
    assert manifest_path.read_bytes() == manifest_before
    assert calls == [
        {
            "cmd": [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-show_streams",
                "-of",
                "json",
                str(video_path),
            ],
            "check": True,
            "capture_output": True,
            "text": True,
        }
    ]


def test_probe_media_video_with_no_audio_stream(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    _make_fake_video(data_dir / "videos" / "lecture_001.mp4")
    _write_manifest(data_dir)

    def fake_run(cmd, check, capture_output, text):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=_ffprobe_output(has_audio=False),
            stderr="",
        )

    monkeypatch.setattr(probe_media_module.subprocess, "run", fake_run)

    metadata = probe_media("lecture_001", data_dir=data_dir)

    assert metadata.has_audio is False


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("30000/1001", 30000 / 1001),
        ("30/1", 30.0),
        ("25/1", 25.0),
    ],
)
def test_parse_fps_valid_values(raw: str, expected: float):
    assert _parse_fps(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw",
    [
        "0/0",
        "0/1",
        "",
        None,
        "not-a-rate",
        "30",
        "30/abc",
    ],
)
def test_parse_fps_invalid_values(raw: str | None):
    assert _parse_fps(raw) is None


def test_probe_media_uses_r_frame_rate_when_avg_is_invalid(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    _make_fake_video(data_dir / "videos" / "lecture_001.mp4")
    _write_manifest(data_dir)

    def fake_run(cmd, check, capture_output, text):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=_ffprobe_output(avg_frame_rate="0/0", r_frame_rate="30/1"),
            stderr="",
        )

    monkeypatch.setattr(probe_media_module.subprocess, "run", fake_run)

    metadata = probe_media("lecture_001", data_dir=data_dir)

    assert metadata.fps == 30.0


def test_probe_media_invalid_fps_fails_clearly(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    _make_fake_video(data_dir / "videos" / "lecture_001.mp4")
    _write_manifest(data_dir)

    def fake_run(cmd, check, capture_output, text):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=_ffprobe_output(avg_frame_rate="0/0", r_frame_rate="0/1"),
            stderr="",
        )

    monkeypatch.setattr(probe_media_module.subprocess, "run", fake_run)

    with pytest.raises(ValueError) as exc_info:
        probe_media("lecture_001", data_dir=data_dir)

    assert "valid FPS" in str(exc_info.value)


def test_missing_video_manifest_says_stage_1_first(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as exc_info:
        probe_media("lecture_001", data_dir=tmp_path / "data")

    assert "Run Stage 1 first" in str(exc_info.value)


def test_missing_registered_video_file_reports_resolved_path(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    _write_manifest(data_dir)

    with pytest.raises(FileNotFoundError) as exc_info:
        probe_media("lecture_001", data_dir=data_dir)

    msg = str(exc_info.value)
    assert "registered video file not found" in msg
    assert str(tmp_path / "data" / "videos" / "lecture_001.mp4") in msg


def test_existing_metadata_without_overwrite_fails(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    _make_fake_video(data_dir / "videos" / "lecture_001.mp4")
    _write_manifest(data_dir)
    write_json(
        data_dir / "manifests" / "lecture_001" / "media_metadata.json",
        MediaMetadata(
            video_id="lecture_001",
            duration_seconds=1.0,
            fps=30.0,
            width=640,
            height=480,
            has_audio=True,
        ),
    )

    with pytest.raises(FileExistsError):
        probe_media("lecture_001", data_dir=data_dir)


def test_overwrite_allows_replacement(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    _make_fake_video(data_dir / "videos" / "lecture_001.mp4")
    _write_manifest(data_dir)
    metadata_path = data_dir / "manifests" / "lecture_001" / "media_metadata.json"
    write_json(
        metadata_path,
        MediaMetadata(
            video_id="lecture_001",
            duration_seconds=1.0,
            fps=30.0,
            width=640,
            height=480,
            has_audio=True,
        ),
    )

    def fake_run(cmd, check, capture_output, text):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=_ffprobe_output(duration="20.0", avg_frame_rate="25/1"),
            stderr="",
        )

    monkeypatch.setattr(probe_media_module.subprocess, "run", fake_run)

    metadata = probe_media("lecture_001", data_dir=data_dir, overwrite=True)

    assert metadata.duration_seconds == 20.0
    assert metadata.fps == 25.0
    assert read_json(metadata_path, MediaMetadata) == metadata


def test_ffprobe_failure_includes_stderr(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    _make_fake_video(data_dir / "videos" / "lecture_001.mp4")
    _write_manifest(data_dir)

    def fake_run(cmd, check, capture_output, text):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=cmd,
            stderr="ffprobe error details",
        )

    monkeypatch.setattr(probe_media_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        probe_media("lecture_001", data_dir=data_dir)

    assert "ffprobe error details" in str(exc_info.value)


def test_cli_success(tmp_path: Path, monkeypatch, capsys):
    def fake_probe_media(video_id, data_dir, overwrite):
        assert video_id == "lecture_001"
        assert data_dir == tmp_path / "data"
        assert overwrite is True
        return MediaMetadata(
            video_id=video_id,
            duration_seconds=10.0,
            fps=30.0,
            width=1920,
            height=1080,
            has_audio=True,
        )

    monkeypatch.setattr(probe_media_module, "probe_media", fake_probe_media)

    exit_code = main(
        [
            "--video-id",
            "lecture_001",
            "--data-dir",
            str(tmp_path / "data"),
            "--overwrite",
        ]
    )

    captured = capsys.readouterr()
    expected_path = tmp_path / "data" / "manifests" / "lecture_001" / "media_metadata.json"
    assert exit_code == 0
    assert "Wrote media metadata:" in captured.out
    assert str(expected_path) in captured.out


def test_cli_failure(tmp_path: Path, capsys):
    exit_code = main(
        ["--video-id", "lecture_001", "--data-dir", str(tmp_path / "data")]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL  FileNotFoundError" in captured.err
    assert "Run Stage 1 first" in captured.err
