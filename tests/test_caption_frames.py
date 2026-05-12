from __future__ import annotations

from pathlib import Path

import pytest

from video_rag.index import caption_frames as caption_frames_module
from video_rag.index.caption_frames import caption_frames, main
from video_rag.io_utils import read_jsonl, write_jsonl
from video_rag.schemas import FrameSample, VLMCaption


def _make_fake_frame(path: Path, payload: bytes = b"fake image bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _samples(
    video_id: str = "lecture_001",
    count: int = 4,
) -> list[FrameSample]:
    return [
        FrameSample(
            video_id=video_id,
            timestamp=float(i * 15),
            frame_path=f"data/frames/{video_id}/frame_{i:06d}.jpg",
        )
        for i in range(count)
    ]


def _write_frame_manifest(
    data_dir: Path,
    video_id: str = "lecture_001",
    samples: list[FrameSample] | None = None,
) -> Path:
    records = samples if samples is not None else _samples(video_id=video_id)
    manifest_path = data_dir / "frames" / video_id / "frame_manifest.jsonl"
    write_jsonl(manifest_path, records)
    return manifest_path


def _write_frame_files(tmp_path: Path, samples: list[FrameSample]) -> None:
    for sample in samples:
        _make_fake_frame(tmp_path / sample.frame_path)


def test_caption_frames_success_with_mocked_vlm_response(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    samples = _samples(count=2)
    _write_frame_files(tmp_path, samples)
    manifest_path = _write_frame_manifest(data_dir, samples=samples)
    manifest_before = manifest_path.read_bytes()
    calls = []

    def fake_caption_frame_group(frame_paths, model):
        calls.append((frame_paths, model))
        return "  A lecturer stands beside a Bayes theorem slide.  "

    monkeypatch.setattr(
        caption_frames_module,
        "_caption_frame_group",
        fake_caption_frame_group,
    )

    records = caption_frames("lecture_001", data_dir=data_dir, frames_per_caption=2)

    assert records == [
        VLMCaption(
            video_id="lecture_001",
            start_time=0.0,
            end_time=15.0,
            frame_paths=[
                "data/frames/lecture_001/frame_000000.jpg",
                "data/frames/lecture_001/frame_000001.jpg",
            ],
            caption="A lecturer stands beside a Bayes theorem slide.",
            caption_type="generic",
            model="gpt-4o-mini",
        )
    ]
    assert calls == [
        (
            [
                tmp_path / "data/frames/lecture_001/frame_000000.jpg",
                tmp_path / "data/frames/lecture_001/frame_000001.jpg",
            ],
            "gpt-4o-mini",
        )
    ]
    assert manifest_path.read_bytes() == manifest_before


def test_correct_grouping_of_frame_records_into_caption_windows(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    samples = _samples(count=5)
    _write_frame_files(tmp_path, samples)
    _write_frame_manifest(data_dir, samples=samples)
    captions = iter(["Group one.", "Group two.", "Group three."])

    def fake_caption_frame_group(frame_paths, model):
        return next(captions)

    monkeypatch.setattr(
        caption_frames_module,
        "_caption_frame_group",
        fake_caption_frame_group,
    )

    records = caption_frames("lecture_001", data_dir=data_dir, frames_per_caption=2)

    assert [(r.start_time, r.end_time, r.frame_paths) for r in records] == [
        (
            0.0,
            15.0,
            [
                "data/frames/lecture_001/frame_000000.jpg",
                "data/frames/lecture_001/frame_000001.jpg",
            ],
        ),
        (
            30.0,
            45.0,
            [
                "data/frames/lecture_001/frame_000002.jpg",
                "data/frames/lecture_001/frame_000003.jpg",
            ],
        ),
        (60.0, 60.0, ["data/frames/lecture_001/frame_000004.jpg"]),
    ]


def test_output_jsonl_contains_valid_vlm_caption_records(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    samples = _samples(count=3)
    _write_frame_files(tmp_path, samples)
    _write_frame_manifest(data_dir, samples=samples)
    monkeypatch.setattr(
        caption_frames_module,
        "_caption_frame_group",
        lambda frame_paths, model: "A slide with a diagram is visible.",
    )

    records = caption_frames("lecture_001", data_dir=data_dir)

    output_path = data_dir / "captions" / "lecture_001.jsonl"
    assert list(read_jsonl(output_path, VLMCaption)) == records


def test_missing_frame_manifest_says_stage_5_first(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as exc_info:
        caption_frames("lecture_001", data_dir=tmp_path / "data")

    assert "Run Stage 5 first" in str(exc_info.value)


def test_missing_referenced_frame_file_reports_resolved_path(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    _write_frame_manifest(data_dir, samples=_samples(count=1))

    with pytest.raises(FileNotFoundError) as exc_info:
        caption_frames("lecture_001", data_dir=data_dir)

    msg = str(exc_info.value)
    assert "sampled frame file not found" in msg
    assert str(tmp_path / "data" / "frames" / "lecture_001" / "frame_000000.jpg") in msg


def test_empty_vlm_response_fails_clearly(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    samples = _samples(count=1)
    _write_frame_files(tmp_path, samples)
    _write_frame_manifest(data_dir, samples=samples)
    monkeypatch.setattr(
        caption_frames_module,
        "_caption_frame_group",
        lambda frame_paths, model: "   ",
    )

    with pytest.raises(ValueError) as exc_info:
        caption_frames("lecture_001", data_dir=data_dir)

    assert "empty caption" in str(exc_info.value)


def test_existing_captions_without_overwrite_fails(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    samples = _samples(count=1)
    _write_frame_files(tmp_path, samples)
    _write_frame_manifest(data_dir, samples=samples)
    output_path = data_dir / "captions" / "lecture_001.jsonl"
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"existing": true}\n', encoding="utf-8")

    def fail_if_called(frame_paths, model):
        raise AssertionError("VLM should not be called when output exists")

    monkeypatch.setattr(caption_frames_module, "_caption_frame_group", fail_if_called)

    with pytest.raises(FileExistsError):
        caption_frames("lecture_001", data_dir=data_dir)


def test_overwrite_allows_replacement(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    samples = _samples(count=1)
    _write_frame_files(tmp_path, samples)
    _write_frame_manifest(data_dir, samples=samples)
    output_path = data_dir / "captions" / "lecture_001.jsonl"
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"existing": true}\n', encoding="utf-8")
    monkeypatch.setattr(
        caption_frames_module,
        "_caption_frame_group",
        lambda frame_paths, model: "Replacement caption.",
    )

    records = caption_frames("lecture_001", data_dir=data_dir, overwrite=True)

    assert records[0].caption == "Replacement caption."
    assert list(read_jsonl(output_path, VLMCaption)) == records


@pytest.mark.parametrize("frames_per_caption", [0, -1])
def test_invalid_frames_per_caption_fails_clearly(
    tmp_path: Path,
    frames_per_caption: int,
):
    with pytest.raises(ValueError) as exc_info:
        caption_frames(
            "lecture_001",
            data_dir=tmp_path / "data",
            frames_per_caption=frames_per_caption,
        )

    assert "positive integer" in str(exc_info.value)


def test_cli_success(tmp_path: Path, monkeypatch, capsys):
    def fake_caption_frames(video_id, data_dir, frames_per_caption, overwrite):
        assert video_id == "lecture_001"
        assert data_dir == tmp_path / "data"
        assert frames_per_caption == 2
        assert overwrite is True
        return [
            VLMCaption(
                video_id=video_id,
                start_time=0.0,
                end_time=15.0,
                frame_paths=["data/frames/lecture_001/frame_000000.jpg"],
                caption="A lecturer is visible.",
                caption_type="generic",
                model="gpt-4o-mini",
            )
        ]

    monkeypatch.setattr(caption_frames_module, "caption_frames", fake_caption_frames)

    exit_code = main(
        [
            "--video-id",
            "lecture_001",
            "--data-dir",
            str(tmp_path / "data"),
            "--frames-per-caption",
            "2",
            "--overwrite",
        ]
    )

    captured = capsys.readouterr()
    expected_path = tmp_path / "data" / "captions" / "lecture_001.jsonl"
    assert exit_code == 0
    assert "Wrote frame captions:" in captured.out
    assert str(expected_path) in captured.out


def test_cli_failure(tmp_path: Path, capsys):
    exit_code = main(
        ["--video-id", "lecture_001", "--data-dir", str(tmp_path / "data")]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL  FileNotFoundError" in captured.err
    assert "Run Stage 5 first" in captured.err
