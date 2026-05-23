from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from video_rag.index.transcription_providers import (
    PROVIDERS,
    MockTranscriptionProvider,
    OpenAITranscriptionProvider,
    ProviderSegment,
    get_provider,
)


def test_mock_provider_returns_segments(tmp_path: Path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"")
    provider = MockTranscriptionProvider()

    segments = provider.transcribe(audio)

    assert len(segments) >= 1
    assert all(isinstance(s, ProviderSegment) for s in segments)


def test_mock_provider_segments_are_well_formed(tmp_path: Path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"")
    segments = MockTranscriptionProvider().transcribe(audio)

    for s in segments:
        assert s.end > s.start
        assert s.text.strip() != ""


def test_mock_provider_is_deterministic(tmp_path: Path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"")
    provider = MockTranscriptionProvider()

    a = provider.transcribe(audio)
    b = provider.transcribe(audio)
    assert a == b


def test_mock_provider_accepts_language_kwarg(tmp_path: Path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"")
    segments = MockTranscriptionProvider().transcribe(audio, language="en")
    assert len(segments) >= 1


def test_mock_provider_name():
    assert MockTranscriptionProvider.name == "mock"


def test_get_provider_resolves_mock():
    p = get_provider("mock")
    assert isinstance(p, MockTranscriptionProvider)


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("does-not-exist")


def test_registry_contains_mock():
    assert "mock" in PROVIDERS
    assert PROVIDERS["mock"] is MockTranscriptionProvider


def test_provider_segment_is_frozen():
    seg = ProviderSegment(start=0.0, end=1.0, text="hi")
    with pytest.raises(Exception):
        seg.text = "changed"  # type: ignore[misc]


# --- OpenAITranscriptionProvider --------------------------------------------


class _FakeSegment:
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text


class _FakeResponse:
    def __init__(self, segments):
        self.segments = segments


class _FakeTranscriptions:
    def __init__(self):
        self.calls: list[dict] = []
        self.next_segments = [_FakeSegment(0.0, 1.0, "hello")]
        self.next_error: Exception | None = None

    def create(self, **kwargs):
        # consume the file-like so the open() in the provider exercises real I/O
        kwargs["file"].read()
        self.calls.append(kwargs)
        if self.next_error is not None:
            raise self.next_error
        return _FakeResponse(segments=list(self.next_segments))


@pytest.fixture
def fake_openai(monkeypatch):
    """Inject a fake ``openai`` module. Yields the recording transcriptions
    object so tests can configure responses and assert on call kwargs."""
    fake_transcriptions = _FakeTranscriptions()

    class _FakeAudio:
        transcriptions = fake_transcriptions

    class _FakeOpenAI:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.audio = _FakeAudio()

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return fake_transcriptions


def _wav(tmp_path: Path) -> Path:
    p = tmp_path / "x.wav"
    p.write_bytes(b"fake-wav-bytes")
    return p


def test_openai_provider_name_and_registry():
    assert OpenAITranscriptionProvider.name == "openai"
    assert "openai" in PROVIDERS
    assert PROVIDERS["openai"] is OpenAITranscriptionProvider
    assert isinstance(get_provider("openai"), OpenAITranscriptionProvider)


def test_openai_happy_path(fake_openai, tmp_path: Path):
    fake_openai.next_segments = [
        _FakeSegment(0.0, 1.5, "hello"),
        _FakeSegment(1.5, 3.0, "world"),
    ]

    segments = OpenAITranscriptionProvider().transcribe(_wav(tmp_path))

    assert [(s.start, s.end, s.text) for s in segments] == [
        (0.0, 1.5, "hello"),
        (1.5, 3.0, "world"),
    ]
    assert all(isinstance(s, ProviderSegment) for s in segments)


def test_openai_request_uses_verbose_json_with_segment_timestamps(
    fake_openai, tmp_path: Path
):
    OpenAITranscriptionProvider().transcribe(_wav(tmp_path))
    call = fake_openai.calls[-1]
    assert call["model"] == "whisper-1"
    assert call["response_format"] == "verbose_json"
    assert call["timestamp_granularities"] == ["segment"]


def test_openai_language_forwarded_when_set(fake_openai, tmp_path: Path):
    OpenAITranscriptionProvider().transcribe(_wav(tmp_path), language="en")
    assert fake_openai.calls[-1]["language"] == "en"


def test_openai_language_omitted_when_none(fake_openai, tmp_path: Path):
    OpenAITranscriptionProvider().transcribe(_wav(tmp_path))
    assert "language" not in fake_openai.calls[-1]


def test_openai_missing_api_key_raises(fake_openai, monkeypatch, tmp_path: Path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        OpenAITranscriptionProvider().transcribe(_wav(tmp_path))


def test_openai_missing_dependency_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(RuntimeError, match="install the 'transcribe' extra"):
        OpenAITranscriptionProvider().transcribe(_wav(tmp_path))


def test_openai_api_error_wrapped_as_runtime_error(fake_openai, tmp_path: Path):
    fake_openai.next_error = Exception("boom: rate limit exceeded")
    with pytest.raises(RuntimeError, match="openai transcription failed"):
        OpenAITranscriptionProvider().transcribe(_wav(tmp_path))


def test_openai_empty_segments_returns_empty_list(fake_openai, tmp_path: Path):
    fake_openai.next_segments = []
    assert OpenAITranscriptionProvider().transcribe(_wav(tmp_path)) == []
