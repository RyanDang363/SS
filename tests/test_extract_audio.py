from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from video_rag.index import extract_audio as extract_audio_module
from video_rag.index.extract_audio import extract_audio, main
from video_rag.io_utils import write_json
from video_rag.schemas import MediaMetadata, VideoManifest


def _make_fake_video(path: Path, payload: bytes = b"fake video bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_stage_artifacts(
    data_dir: Path,
    video_id: str = "lecture_001",
    source_path: str = "data/videos/lecture_001.mp4",
    has_audio: bool = True,
) -> tuple[Path, Path]:
    manifest_path = data_dir / "manifests" / video_id / "video_manifest.json"
    metadata_path = data_dir / "manifests" / video_id / "media_metadata.json"

    write_json(
        manifest_path,
        VideoManifest(
            video_id=video_id,
            title="Lecture 001",
            source_path=source_path,
            original_filename="lecture_001.mp4",
            created_at="2026-05-04T00:00:00Z",
        ),
    )
    write_json(
        metadata_path,
        MediaMetadata(
            video_id=video_id,
            duration_seconds=10.0,
            has_audio=has_audio,
            fps=30.0,
            width=1920,
            height=1080,
        ),
    )
    return manifest_path, metadata_path


def test_extract_audio_success_with_mocked_ffmpeg(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    video_path = _make_fake_video(data_dir / "videos" / "lecture_001.mp4")
    _write_stage_artifacts(data_dir)
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
        Path(cmd[-1]).write_bytes(b"fake wav bytes")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(extract_audio_module.subprocess, "run", fake_run)

    output_path = extract_audio("lecture_001", data_dir=data_dir)

    assert output_path == data_dir / "audio" / "lecture_001.wav"
    assert output_path.read_bytes() == b"fake wav bytes"
    assert calls == [
        {
            "cmd": [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                str(output_path),
            ],
            "check": True,
            "capture_output": True,
            "text": True,
        }
    ]


def test_missing_video_manifest_says_stage_1_first(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as exc_info:
        extract_audio("lecture_001", data_dir=tmp_path / "data")

    assert "Run Stage 1 first" in str(exc_info.value)


def test_missing_media_metadata_says_stage_2_first(tmp_path: Path):
    data_dir = tmp_path / "data"
    write_json(
        data_dir / "manifests" / "lecture_001" / "video_manifest.json",
        VideoManifest(
            video_id="lecture_001",
            title="Lecture 001",
            source_path="data/videos/lecture_001.mp4",
        ),
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        extract_audio("lecture_001", data_dir=data_dir)

    assert "Run Stage 2 first" in str(exc_info.value)


def test_has_audio_false_fails_without_creating_audio(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    _make_fake_video(data_dir / "videos" / "lecture_001.mp4")
    _write_stage_artifacts(data_dir, has_audio=False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("ffmpeg should not run when has_audio=false")

    monkeypatch.setattr(extract_audio_module.subprocess, "run", fail_if_called)

    with pytest.raises(ValueError) as exc_info:
        extract_audio("lecture_001", data_dir=data_dir)

    assert "has_audio=false" in str(exc_info.value)
    assert not (data_dir / "audio" / "lecture_001.wav").exists()


def test_missing_registered_video_file_reports_resolved_path(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    _write_stage_artifacts(data_dir)

    with pytest.raises(FileNotFoundError) as exc_info:
        extract_audio("lecture_001", data_dir=data_dir)

    msg = str(exc_info.value)
    assert "registered video file not found" in msg
    assert str(tmp_path / "data" / "videos" / "lecture_001.mp4") in msg


def test_existing_audio_without_overwrite_fails(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    _make_fake_video(data_dir / "videos" / "lecture_001.mp4")
    _write_stage_artifacts(data_dir)
    audio_path = data_dir / "audio" / "lecture_001.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        extract_audio("lecture_001", data_dir=data_dir)


def test_overwrite_allows_replacement(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    _make_fake_video(data_dir / "videos" / "lecture_001.mp4")
    _write_stage_artifacts(data_dir)
    audio_path = data_dir / "audio" / "lecture_001.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"existing")

    def fake_run(cmd, check, capture_output, text):
        Path(cmd[-1]).write_bytes(b"replacement")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(extract_audio_module.subprocess, "run", fake_run)

    assert extract_audio("lecture_001", data_dir=data_dir, overwrite=True) == audio_path
    assert audio_path.read_bytes() == b"replacement"


def test_ffmpeg_failure_includes_stderr(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    _make_fake_video(data_dir / "videos" / "lecture_001.mp4")
    _write_stage_artifacts(data_dir)

    def fake_run(cmd, check, capture_output, text):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=cmd,
            stderr="ffmpeg error details",
        )

    monkeypatch.setattr(extract_audio_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        extract_audio("lecture_001", data_dir=data_dir)

    assert "ffmpeg error details" in str(exc_info.value)


def test_success_does_not_modify_stage_1_or_stage_2_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    _make_fake_video(data_dir / "videos" / "lecture_001.mp4")
    manifest_path, metadata_path = _write_stage_artifacts(data_dir)
    manifest_before = manifest_path.read_bytes()
    metadata_before = metadata_path.read_bytes()

    def fake_run(cmd, check, capture_output, text):
        Path(cmd[-1]).write_bytes(b"fake wav bytes")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(extract_audio_module.subprocess, "run", fake_run)

    extract_audio("lecture_001", data_dir=data_dir)

    assert manifest_path.read_bytes() == manifest_before
    assert metadata_path.read_bytes() == metadata_before


def test_cli_success(tmp_path: Path, monkeypatch, capsys):
    audio_path = tmp_path / "data" / "audio" / "lecture_001.wav"

    def fake_extract_audio(video_id, data_dir, overwrite):
        assert video_id == "lecture_001"
        assert data_dir == tmp_path / "data"
        assert overwrite is True
        return audio_path

    monkeypatch.setattr(extract_audio_module, "extract_audio", fake_extract_audio)

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
    assert exit_code == 0
    assert "Extracted audio:" in captured.out
    assert str(audio_path) in captured.out


def test_cli_failure(tmp_path: Path, capsys):
    exit_code = main(
        ["--video-id", "lecture_001", "--data-dir", str(tmp_path / "data")]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL  FileNotFoundError" in captured.err
    assert "Run Stage 1 first" in captured.err
