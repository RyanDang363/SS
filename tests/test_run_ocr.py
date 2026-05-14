from __future__ import annotations

from pathlib import Path

import pytest

from video_rag.index import run_ocr as run_ocr_module
from video_rag.index.run_ocr import main, run_ocr
from video_rag.io_utils import read_jsonl, write_jsonl
from video_rag.schemas import FrameSample, OCRResult


class FakeReader:
    def __init__(self, results_by_path: dict[str, list]):
        self.results_by_path = results_by_path

    def readtext(self, frame_path: str):
        return self.results_by_path.get(frame_path, [])


def _make_fake_frame(path: Path, payload: bytes = b"fake image bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_frame_manifest(
    data_dir: Path,
    video_id: str = "lecture_001",
    samples: list[FrameSample] | None = None,
) -> Path:
    if samples is None:
        samples = [
            FrameSample(
                video_id=video_id,
                timestamp=15.0,
                frame_path=f"data/frames/{video_id}/frame_000015.jpg",
            )
        ]
    manifest_path = data_dir / "frames" / video_id / "frame_manifest.jsonl"
    write_jsonl(manifest_path, samples)
    return manifest_path


def test_run_ocr_success_with_mocked_ocr_engine(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    sample = FrameSample(
        video_id="lecture_001",
        timestamp=15.0,
        frame_path="data/frames/lecture_001/frame_000015.jpg",
    )
    frame_path = _make_fake_frame(tmp_path / sample.frame_path)
    manifest_path = _write_frame_manifest(data_dir, samples=[sample])
    manifest_before = manifest_path.read_bytes()
    fake_reader = FakeReader(
        {
            str(frame_path): [
                ("box1", "Bayes", 0.91),
                ("box2", "Theorem", 0.85),
            ]
        }
    )
    monkeypatch.setattr(run_ocr_module, "_create_reader", lambda: fake_reader)

    records = run_ocr("lecture_001", data_dir=data_dir)

    output_path = data_dir / "ocr" / "lecture_001.jsonl"
    assert records == [
        OCRResult(
            video_id="lecture_001",
            timestamp=15.0,
            frame_path="data/frames/lecture_001/frame_000015.jpg",
            ocr_text="Bayes Theorem",
            confidence=0.88,
        )
    ]
    assert list(read_jsonl(output_path, OCRResult)) == records
    assert manifest_path.read_bytes() == manifest_before


def test_output_jsonl_contains_valid_ocr_result_records(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    sample = FrameSample(
        video_id="lecture_001",
        timestamp=30.0,
        frame_path="data/frames/lecture_001/frame_000030.jpg",
    )
    frame_path = _make_fake_frame(tmp_path / sample.frame_path)
    _write_frame_manifest(data_dir, samples=[sample])
    fake_reader = FakeReader({str(frame_path): [("box", "Slide Title", 0.75)]})
    monkeypatch.setattr(run_ocr_module, "_create_reader", lambda: fake_reader)

    run_ocr("lecture_001", data_dir=data_dir)

    loaded = list(read_jsonl(data_dir / "ocr" / "lecture_001.jsonl", OCRResult))
    assert loaded == [
        OCRResult(
            video_id="lecture_001",
            timestamp=30.0,
            frame_path="data/frames/lecture_001/frame_000030.jpg",
            ocr_text="Slide Title",
            confidence=0.75,
        )
    ]


def test_empty_ocr_result_still_writes_valid_record(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    sample = FrameSample(
        video_id="lecture_001",
        timestamp=20.0,
        frame_path="data/frames/lecture_001/frame_000020.jpg",
    )
    frame_path = _make_fake_frame(tmp_path / sample.frame_path)
    _write_frame_manifest(data_dir, samples=[sample])
    fake_reader = FakeReader({str(frame_path): []})
    monkeypatch.setattr(run_ocr_module, "_create_reader", lambda: fake_reader)

    records = run_ocr("lecture_001", data_dir=data_dir)

    assert records == [
        OCRResult(
            video_id="lecture_001",
            timestamp=20.0,
            frame_path="data/frames/lecture_001/frame_000020.jpg",
            ocr_text="",
            confidence=None,
        )
    ]
    assert list(read_jsonl(data_dir / "ocr" / "lecture_001.jsonl", OCRResult)) == records


def test_missing_frame_manifest_says_stage_5_first(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as exc_info:
        run_ocr("lecture_001", data_dir=tmp_path / "data")

    assert "Run Stage 5 first" in str(exc_info.value)


def test_missing_frame_file_reports_resolved_path(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    _write_frame_manifest(data_dir)

    with pytest.raises(FileNotFoundError) as exc_info:
        run_ocr("lecture_001", data_dir=data_dir)

    msg = str(exc_info.value)
    assert "sampled frame file not found" in msg
    assert str(tmp_path / "data" / "frames" / "lecture_001" / "frame_000015.jpg") in msg


def test_existing_ocr_output_without_overwrite_fails(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    sample = FrameSample(
        video_id="lecture_001",
        timestamp=15.0,
        frame_path="data/frames/lecture_001/frame_000015.jpg",
    )
    _make_fake_frame(tmp_path / sample.frame_path)
    _write_frame_manifest(data_dir, samples=[sample])
    output_path = data_dir / "ocr" / "lecture_001.jsonl"
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"existing": true}\n', encoding="utf-8")

    def fail_if_called():
        raise AssertionError("OCR reader should not be created when output exists")

    monkeypatch.setattr(run_ocr_module, "_create_reader", fail_if_called)

    with pytest.raises(FileExistsError):
        run_ocr("lecture_001", data_dir=data_dir)


def test_overwrite_allows_replacement(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    sample = FrameSample(
        video_id="lecture_001",
        timestamp=15.0,
        frame_path="data/frames/lecture_001/frame_000015.jpg",
    )
    frame_path = _make_fake_frame(tmp_path / sample.frame_path)
    _write_frame_manifest(data_dir, samples=[sample])
    output_path = data_dir / "ocr" / "lecture_001.jsonl"
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"existing": true}\n', encoding="utf-8")
    fake_reader = FakeReader({str(frame_path): [("box", "Replacement", 0.9)]})
    monkeypatch.setattr(run_ocr_module, "_create_reader", lambda: fake_reader)

    records = run_ocr("lecture_001", data_dir=data_dir, overwrite=True)

    assert records[0].ocr_text == "Replacement"
    assert list(read_jsonl(output_path, OCRResult)) == records


def test_confidence_normalization_averages_multiple_boxes(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    sample = FrameSample(
        video_id="lecture_001",
        timestamp=15.0,
        frame_path="data/frames/lecture_001/frame_000015.jpg",
    )
    frame_path = _make_fake_frame(tmp_path / sample.frame_path)
    _write_frame_manifest(data_dir, samples=[sample])
    fake_reader = FakeReader(
        {str(frame_path): [("box1", "A", 0.6), ("box2", "B", 0.8)]}
    )
    monkeypatch.setattr(run_ocr_module, "_create_reader", lambda: fake_reader)

    records = run_ocr("lecture_001", data_dir=data_dir)

    assert records[0].ocr_text == "A B"
    assert records[0].confidence == pytest.approx(0.7)


def test_cli_success(tmp_path: Path, monkeypatch, capsys):
    def fake_run_ocr(video_id, data_dir, overwrite):
        assert video_id == "lecture_001"
        assert data_dir == tmp_path / "data"
        assert overwrite is True
        return [
            OCRResult(
                video_id=video_id,
                timestamp=15.0,
                frame_path="data/frames/lecture_001/frame_000015.jpg",
                ocr_text="Bayes",
                confidence=0.9,
            )
        ]

    monkeypatch.setattr(run_ocr_module, "run_ocr", fake_run_ocr)

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
    expected_path = tmp_path / "data" / "ocr" / "lecture_001.jsonl"
    assert exit_code == 0
    assert "Wrote OCR results:" in captured.out
    assert str(expected_path) in captured.out


def test_cli_failure(tmp_path: Path, capsys):
    exit_code = main(
        ["--video-id", "lecture_001", "--data-dir", str(tmp_path / "data")]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL  FileNotFoundError" in captured.err
    assert "Run Stage 5 first" in captured.err
