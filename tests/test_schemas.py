from __future__ import annotations

import pytest
from pydantic import ValidationError

from video_rag.schemas import FrameSample, MediaMetadata, OCRResult, VideoManifest


# --- VideoManifest -----------------------------------------------------------


def test_video_manifest_valid_minimal():
    m = VideoManifest(
        video_id="lecture_001",
        title="Bayes Theorem Lecture",
        source_path="data/videos/lecture_001.mp4",
    )
    assert m.video_id == "lecture_001"
    assert m.original_filename is None
    assert m.created_at is None


def test_video_manifest_valid_full():
    m = VideoManifest(
        video_id="lecture_001",
        title="Bayes Theorem Lecture",
        source_path="data/videos/lecture_001.mp4",
        original_filename="lecture final.mp4",
        created_at="2026-05-03T12:00:00Z",
    )
    assert m.original_filename == "lecture final.mp4"


def test_video_manifest_empty_video_id_fails():
    with pytest.raises(ValidationError):
        VideoManifest(video_id="", title="t", source_path="p")


def test_video_manifest_empty_title_fails():
    with pytest.raises(ValidationError):
        VideoManifest(video_id="v", title="", source_path="p")


def test_video_manifest_empty_source_path_fails():
    with pytest.raises(ValidationError):
        VideoManifest(video_id="v", title="t", source_path="")


# --- MediaMetadata -----------------------------------------------------------


def test_media_metadata_valid_minimal():
    m = MediaMetadata(video_id="lecture_001", duration_seconds=10.0, has_audio=True)
    assert m.fps is None and m.width is None and m.height is None


def test_media_metadata_valid_full():
    m = MediaMetadata(
        video_id="lecture_001",
        duration_seconds=1840.5,
        fps=30.0,
        width=1920,
        height=1080,
        has_audio=True,
    )
    assert m.fps == 30.0


def test_media_metadata_empty_video_id_fails():
    with pytest.raises(ValidationError):
        MediaMetadata(video_id="", duration_seconds=10.0, has_audio=True)


@pytest.mark.parametrize("duration", [0, -1, -0.5])
def test_media_metadata_nonpositive_duration_fails(duration):
    with pytest.raises(ValidationError):
        MediaMetadata(video_id="v", duration_seconds=duration, has_audio=True)


@pytest.mark.parametrize("fps", [0, -1, -0.5])
def test_media_metadata_nonpositive_fps_fails(fps):
    with pytest.raises(ValidationError):
        MediaMetadata(video_id="v", duration_seconds=1.0, has_audio=True, fps=fps)


@pytest.mark.parametrize("width", [0, -1])
def test_media_metadata_nonpositive_width_fails(width):
    with pytest.raises(ValidationError):
        MediaMetadata(video_id="v", duration_seconds=1.0, has_audio=True, width=width)


@pytest.mark.parametrize("height", [0, -1])
def test_media_metadata_nonpositive_height_fails(height):
    with pytest.raises(ValidationError):
        MediaMetadata(video_id="v", duration_seconds=1.0, has_audio=True, height=height)


# --- FrameSample -------------------------------------------------------------


def test_frame_sample_valid():
    sample = FrameSample(
        video_id="lecture_001",
        timestamp=15.0,
        frame_path="data/frames/lecture_001/frame_000015.jpg",
    )
    assert sample.timestamp == 15.0


def test_frame_sample_empty_frame_path_fails():
    with pytest.raises(ValidationError):
        FrameSample(video_id="lecture_001", timestamp=15.0, frame_path="")


def test_frame_sample_negative_timestamp_fails():
    with pytest.raises(ValidationError):
        FrameSample(
            video_id="lecture_001",
            timestamp=-1.0,
            frame_path="data/frames/lecture_001/frame.jpg",
        )


# --- OCRResult ---------------------------------------------------------------


def test_ocr_result_valid_with_text():
    result = OCRResult(
        video_id="lecture_001",
        timestamp=15.0,
        frame_path="data/frames/lecture_001/frame_000015.jpg",
        ocr_text="Bayes Theorem",
        confidence=0.88,
    )
    assert result.confidence == 0.88


def test_ocr_result_valid_empty_text_without_confidence():
    result = OCRResult(
        video_id="lecture_001",
        timestamp=20.0,
        frame_path="data/frames/lecture_001/frame_000020.jpg",
        ocr_text="",
        confidence=None,
    )
    assert result.ocr_text == ""


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_ocr_result_confidence_out_of_range_fails(confidence):
    with pytest.raises(ValidationError):
        OCRResult(
            video_id="lecture_001",
            timestamp=15.0,
            frame_path="data/frames/lecture_001/frame.jpg",
            ocr_text="text",
            confidence=confidence,
        )
