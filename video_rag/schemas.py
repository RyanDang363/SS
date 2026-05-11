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

from pydantic import BaseModel, ConfigDict, Field


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


class FrameSample(BaseModel):
    """A sampled frame from a source video."""

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1)
    timestamp: float = Field(ge=0)
    frame_path: str = Field(min_length=1)


class OCRResult(BaseModel):
    """OCR text detected for a sampled frame."""

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1)
    timestamp: float = Field(ge=0)
    frame_path: str = Field(min_length=1)
    ocr_text: str
    confidence: float | None = Field(default=None, ge=0, le=1)
