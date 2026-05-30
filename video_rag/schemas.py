"""Stage 0-lite schemas.

Conventions used across RAGGERS artifacts:
    - ``video_id`` identifies the source video on every record.
    - Timestamps are seconds as ``float``.
    - Time ranges use ``start_time`` / ``end_time``; single points use ``timestamp``.
    - File paths are stored as plain strings.

Future modules (transcripts, OCR, captions, chunks, embeddings, ...) will add
their own schemas alongside these as they are implemented.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VideoManifest(BaseModel):
    """A registered source video."""

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    original_filename: str | None = None
    created_at: str | None = None


class MediaMetadata(BaseModel):
    """Probed media-level facts about a video."""

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1)
    duration_seconds: float = Field(gt=0)
    has_audio: bool
    fps: float | None = Field(default=None, gt=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)


SamplingMethod = Literal["fixed_interval"]


class FrameSample(BaseModel):
    """A single sampled frame from a video.

    One ``FrameSample`` per JPEG written by Stage 5. The collection of records
    for a video is persisted as JSONL at
    ``data/frames/{video_id}/frame_manifest.jsonl``.
    """

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1)
    timestamp: float = Field(ge=0)
    frame_path: str = Field(min_length=1)
    thumbnail_path: str | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sampling_method: SamplingMethod = "fixed_interval"


class TranscriptSegment(BaseModel):
    """One timestamped transcript segment for a video."""

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1)
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_segment(self) -> "TranscriptSegment":
        if self.end_time <= self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be greater than "
                f"start_time ({self.start_time})"
            )
        if not self.text.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return self


class VLMCaption(BaseModel):
    """Generic visual caption for a group of sampled frames."""

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1)
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    frame_paths: list[str] = Field(min_length=1)
    caption: str = Field(min_length=1)
    caption_type: Literal["generic"] = "generic"
    model: str = Field(min_length=1)


class OCRResult(BaseModel):
    """OCR text detected for a sampled frame."""

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1)
    timestamp: float = Field(ge=0)
    frame_path: str = Field(min_length=1)
    ocr_text: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class Chunk(BaseModel):
    """A fixed-time retrieval unit aligning all modalities on a shared timeline.

    Written by Stage 9 to
    ``data/chunks/{video_id}_{chunk_seconds}s.jsonl``. Each chunk covers one
    fixed time window of the video and gathers the transcript, OCR, VLM caption,
    and frame evidence that falls within that window. This is custom timestamp
    alignment across modalities, not text-splitter document chunking.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    transcript_text: str = ""
    ocr_text: str = ""
    vlm_caption: str = ""
    frame_paths: list[str] = Field(default_factory=list)
    chunk_seconds: float = Field(gt=0)
    overlap_seconds: float = Field(default=0.0, ge=0)
    chunking_strategy: Literal["fixed"] = "fixed"

    @model_validator(mode="after")
    def _check_chunk(self) -> "Chunk":
        if self.end_time <= self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be greater than "
                f"start_time ({self.start_time})"
            )
        if self.overlap_seconds >= self.chunk_seconds:
            raise ValueError(
                f"overlap_seconds ({self.overlap_seconds}) must be less than "
                f"chunk_seconds ({self.chunk_seconds})"
            )
        return self
