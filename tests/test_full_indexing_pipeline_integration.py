"""End-to-end smoke test for the implemented indexing pipeline.

This test uses real ffmpeg/ffprobe, real audio extraction, and real OpenCV
frame decoding. Network-heavy AI stages are represented by deterministic test
doubles so the pipeline can run in CI without external credentials.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from video_rag.index import caption_frames as caption_frames_module
from video_rag.index import run_ocr as run_ocr_module
from video_rag.index.caption_frames import caption_frames
from video_rag.index.extract_audio import extract_audio
from video_rag.index.probe_media import probe_media
from video_rag.index.register_video import register_video
from video_rag.index.run_ocr import run_ocr
from video_rag.index.sample_frames import sample_frames
from video_rag.index.transcribe_audio import transcribe_audio
from video_rag.index.transcription_providers import ProviderSegment
from video_rag.io_utils import read_json, read_jsonl
from video_rag.schemas import (
    FrameSample,
    MediaMetadata,
    OCRResult,
    TranscriptSegment,
    VLMCaption,
    VideoManifest,
)
from video_rag.validate import validate_file

FIXTURE_DURATION_SECONDS = 3
FIXTURE_FPS = 30
FIXTURE_WIDTH = 64
FIXTURE_HEIGHT = 48


def _have_media_tools() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


pytestmark = pytest.mark.skipif(
    not _have_media_tools(),
    reason="ffmpeg and ffprobe are required for the full indexing integration test",
)


class _FixedTranscriptionProvider:
    name = "fixed"

    def transcribe(self, audio_path: Path, *, language: str | None = None):
        assert audio_path.is_file()
        return [
            ProviderSegment(start=0.0, end=1.25, text="Opening title card."),
            ProviderSegment(start=1.25, end=2.75, text="Pipeline fixture audio."),
        ]


class _FixedOcrReader:
    def readtext(self, frame_path: str):
        return [("box", f"OCR for {Path(frame_path).stem}", 0.9)]


def _synthesize_fixture_video(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        (
            f"testsrc=duration={FIXTURE_DURATION_SECONDS}"
            f":size={FIXTURE_WIDTH}x{FIXTURE_HEIGHT}"
            f":rate={FIXTURE_FPS}"
        ),
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=1000:duration={FIXTURE_DURATION_SECONDS}",
        "-shortest",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def test_full_indexing_pipeline_smoke(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    video_id = "full_pipeline_fixture"
    source_video = tmp_path / "fixtures" / "fixture.mp4"
    _synthesize_fixture_video(source_video)

    manifest = register_video(
        source_video,
        title="Full Pipeline Fixture",
        video_id=video_id,
        data_dir=data_dir,
    )
    metadata = probe_media(video_id, data_dir=data_dir)
    audio_path = extract_audio(video_id, data_dir=data_dir)
    transcript_segments = transcribe_audio(
        video_id,
        provider=_FixedTranscriptionProvider(),
        data_dir=data_dir,
    )
    frame_samples = sample_frames(
        video_id,
        data_dir=data_dir,
        interval_seconds=1,
    )

    monkeypatch.setattr(
        caption_frames_module,
        "_caption_frame_group",
        lambda frame_paths, model: f"Caption covering {len(frame_paths)} frames.",
    )
    captions = caption_frames(
        video_id,
        data_dir=data_dir,
        frames_per_caption=2,
    )

    monkeypatch.setattr(run_ocr_module, "_create_reader", _FixedOcrReader)
    ocr_records = run_ocr(video_id, data_dir=data_dir)

    assert manifest.video_id == video_id
    assert metadata.video_id == video_id
    assert metadata.has_audio is True
    assert metadata.width == FIXTURE_WIDTH
    assert metadata.height == FIXTURE_HEIGHT
    assert audio_path.is_file()
    assert audio_path.stat().st_size > 0

    assert len(transcript_segments) == 2
    assert [segment.video_id for segment in transcript_segments] == [video_id, video_id]
    assert [sample.timestamp for sample in frame_samples] == [0.0, 1.0, 2.0]
    assert all((tmp_path / sample.frame_path).is_file() for sample in frame_samples)
    assert len(captions) == 2
    assert len(ocr_records) == len(frame_samples)
    assert all(record.ocr_text.startswith("OCR for frame_") for record in ocr_records)

    manifest_path = data_dir / "manifests" / video_id / "video_manifest.json"
    metadata_path = data_dir / "manifests" / video_id / "media_metadata.json"
    transcript_path = data_dir / "transcripts" / f"{video_id}.jsonl"
    frame_manifest_path = data_dir / "frames" / video_id / "frame_manifest.jsonl"
    caption_path = data_dir / "captions" / f"{video_id}.jsonl"
    ocr_path = data_dir / "ocr" / f"{video_id}.jsonl"

    assert read_json(manifest_path, VideoManifest) == manifest
    assert read_json(metadata_path, MediaMetadata) == metadata
    assert list(read_jsonl(transcript_path, TranscriptSegment)) == transcript_segments
    assert list(read_jsonl(frame_manifest_path, FrameSample)) == frame_samples
    assert list(read_jsonl(caption_path, VLMCaption)) == captions
    assert list(read_jsonl(ocr_path, OCRResult)) == ocr_records

    for path, artifact_type in [
        (manifest_path, "video_manifest"),
        (metadata_path, "media_metadata"),
        (transcript_path, "transcript_segments"),
        (frame_manifest_path, "frame_sample"),
    ]:
        ok, message = validate_file(path, artifact_type)
        assert ok, message
