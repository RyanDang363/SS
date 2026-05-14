from __future__ import annotations

import pytest
from pydantic import ValidationError

from video_rag.schemas import FrameSample, MediaMetadata, OCRResult, VideoManifest, TranscriptSegment, VLMCaption

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
    assert sample.frame_path.endswith(".jpg")
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


# --- VLMCaption --------------------------------------------------------------


def test_vlm_caption_valid():
    caption = VLMCaption(
        video_id="lecture_001",
        start_time=0.0,
        end_time=30.0,
        frame_paths=[
            "data/frames/lecture_001/frame_000000.jpg",
            "data/frames/lecture_001/frame_000015.jpg",
        ],
        caption="A lecturer stands beside a slide.",
        caption_type="generic",
        model="gpt-4o-mini",
    )
    assert caption.caption_type == "generic"


def test_vlm_caption_empty_caption_fails():
    with pytest.raises(ValidationError):
        VLMCaption(
            video_id="lecture_001",
            start_time=0.0,
            end_time=30.0,
            frame_paths=["data/frames/lecture_001/frame.jpg"],
            caption="",
            caption_type="generic",
            model="gpt-4o-mini",
        )


def test_vlm_caption_empty_frame_paths_fails():
    with pytest.raises(ValidationError):
        VLMCaption(
            video_id="lecture_001",
            start_time=0.0,
            end_time=30.0,
            frame_paths=[],
            caption="A lecturer stands beside a slide.",
            caption_type="generic",
            model="gpt-4o-mini",
        )


def test_vlm_caption_non_generic_caption_type_fails():
    with pytest.raises(ValidationError):
        VLMCaption(
            video_id="lecture_001",
            start_time=0.0,
            end_time=30.0,
            frame_paths=["data/frames/lecture_001/frame.jpg"],
            caption="A lecturer stands beside a slide.",
            caption_type="query_aware",
            model="gpt-4o-mini",
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
# --- TranscriptSegment -------------------------------------------------------


def test_transcript_segment_valid():
    s = TranscriptSegment(
        video_id="lecture_001",
        start_time=12.4,
        end_time=18.9,
        text="Today we are going to introduce Bayes theorem.",
    )
    assert s.video_id == "lecture_001"
    assert s.start_time == 12.4
    assert s.end_time == 18.9


def test_transcript_segment_round_trip():
    payload = {
        "video_id": "lecture_001",
        "start_time": 0.0,
        "end_time": 1.5,
        "text": "Hello.",
    }
    s = TranscriptSegment.model_validate(payload)
    assert s.model_dump(mode="json") == payload


def test_transcript_segment_start_time_zero_ok():
    s = TranscriptSegment(video_id="v", start_time=0.0, end_time=0.5, text="x")
    assert s.start_time == 0.0


def test_transcript_segment_extra_field_forbidden():
    with pytest.raises(ValidationError):
        TranscriptSegment.model_validate(
            {
                "video_id": "v",
                "start_time": 0.0,
                "end_time": 1.0,
                "text": "hi",
                "speaker": "alice",
            }
        )


def test_transcript_segment_empty_video_id_fails():
    with pytest.raises(ValidationError):
        TranscriptSegment(video_id="", start_time=0.0, end_time=1.0, text="x")


@pytest.mark.parametrize("start_time", [-0.1, -1.0])
def test_transcript_segment_negative_start_time_fails(start_time):
    with pytest.raises(ValidationError):
        TranscriptSegment(
            video_id="v", start_time=start_time, end_time=1.0, text="x"
        )


@pytest.mark.parametrize("end_time", [0, -0.5])
def test_transcript_segment_nonpositive_end_time_fails(end_time):
    with pytest.raises(ValidationError):
        TranscriptSegment(
            video_id="v", start_time=0.0, end_time=end_time, text="x"
        )


@pytest.mark.parametrize(
    "start_time, end_time",
    [(1.0, 1.0), (2.0, 1.5), (0.5, 0.5)],
)
def test_transcript_segment_end_not_after_start_fails(start_time, end_time):
    with pytest.raises(ValidationError):
        TranscriptSegment(
            video_id="v", start_time=start_time, end_time=end_time, text="x"
        )


@pytest.mark.parametrize("text", ["", "   ", "\t\n"])
def test_transcript_segment_empty_or_whitespace_text_fails(text):
    with pytest.raises(ValidationError):
        TranscriptSegment(video_id="v", start_time=0.0, end_time=1.0, text=text)
