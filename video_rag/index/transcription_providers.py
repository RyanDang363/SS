"""Transcription provider adapters for Stage 4.

The stage code (``transcribe_audio``) is decoupled from any specific
transcription backend via the :class:`TranscriptionProvider` protocol.
Each provider takes a path to a ``.wav`` file and returns a list of
:class:`ProviderSegment` records; the stage attaches ``video_id``,
sorts, and validates them as ``TranscriptSegment``.

This module ships a deterministic ``MockTranscriptionProvider`` for
tests and CLI smoke checks. Real providers (e.g. OpenAI) are added in
sibling modules and registered into :data:`PROVIDERS`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ProviderSegment:
    """One transcription segment as returned by a provider, before
    ``video_id`` is attached and ``TranscriptSegment`` validation runs.
    """

    start: float
    end: float
    text: str


class TranscriptionProvider(Protocol):
    """Adapter interface for a transcription backend."""

    name: str

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
    ) -> list[ProviderSegment]: ...


class MockTranscriptionProvider:
    """Deterministic, offline provider for tests and smoke runs."""

    name = "mock"

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
    ) -> list[ProviderSegment]:
        return [
            ProviderSegment(start=0.0, end=1.5, text="Mock segment one."),
            ProviderSegment(start=1.5, end=3.0, text="Mock segment two."),
            ProviderSegment(start=3.0, end=4.25, text="Mock segment three."),
        ]


class OpenAITranscriptionProvider:
    """Real provider backed by the OpenAI ``whisper-1`` endpoint.

    The ``openai`` package is imported lazily so the base install does
    not need it; install with ``pip install -e .[transcribe]``.
    Reads ``OPENAI_API_KEY`` from the environment.
    """

    name = "openai"
    model = "whisper-1"

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
    ) -> list[ProviderSegment]:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "install the 'transcribe' extra to use the openai provider: "
                "pip install -e .[transcribe]"
            ) from e

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set; cannot use the openai provider"
            )

        client = OpenAI(api_key=api_key)
        kwargs: dict = {
            "model": self.model,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if language is not None:
            kwargs["language"] = language

        try:
            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(file=f, **kwargs)
        except Exception as e:
            raise RuntimeError(f"openai transcription failed: {e}") from e

        segments = getattr(response, "segments", None) or []
        return [
            ProviderSegment(start=s.start, end=s.end, text=s.text)
            for s in segments
        ]


PROVIDERS: dict[str, type[TranscriptionProvider]] = {
    "mock": MockTranscriptionProvider,
    "openai": OpenAITranscriptionProvider,
}


def get_provider(name: str) -> TranscriptionProvider:
    """Resolve a provider by registry name. Raises ``ValueError`` if unknown."""
    cls = PROVIDERS.get(name)
    if cls is None:
        known = ", ".join(sorted(PROVIDERS)) or "(none)"
        raise ValueError(f"unknown provider {name!r} (known: {known})")
    return cls()
