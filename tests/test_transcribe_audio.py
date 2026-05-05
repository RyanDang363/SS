from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest

from video_rag.index.transcribe_audio import main, transcribe_audio
from video_rag.index.transcription_providers import ProviderSegment
from video_rag.io_utils import read_jsonl
from video_rag.schemas import TranscriptSegment


def _make_audio(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake wav")
    return path


class _FixedProvider:
    """Inline test double — returns the segments handed to it."""

    name = "fixed"

    def __init__(self, segments: Iterable[ProviderSegment]):
        self._segments = list(segments)

    def transcribe(self, audio_path, *, language=None):
        return list(self._segments)


class _RaisingProvider:
    name = "raising"

    def transcribe(self, audio_path, *, language=None):
        raise RuntimeError("simulated provider failure")


# --- happy path -------------------------------------------------------------


def test_happy_path_with_mock(tmp_path: Path):
    data = tmp_path / "data"
    _make_audio(data / "audio" / "lec.wav")

    segments = transcribe_audio(
        video_id="lec",
        provider="mock",
        data_dir=data,
    )

    assert len(segments) >= 1
    transcript = data / "transcripts" / "lec.jsonl"
    assert transcript.is_file()
    on_disk = list(read_jsonl(transcript, TranscriptSegment))
    assert on_disk == segments
    assert all(s.video_id == "lec" for s in on_disk)


def test_cli_happy_path(tmp_path: Path):
    data = tmp_path / "data"
    _make_audio(data / "audio" / "lec.wav")

    rc = main([
        "--video-id", "lec",
        "--provider", "mock",
        "--data-dir", str(data),
    ])

    assert rc == 0
    assert (data / "transcripts" / "lec.jsonl").is_file()


# --- error mapping ----------------------------------------------------------


def test_missing_audio_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        transcribe_audio(video_id="lec", provider="mock", data_dir=tmp_path / "data")


def test_missing_audio_cli_exits_nonzero(tmp_path: Path):
    rc = main([
        "--video-id", "lec",
        "--provider", "mock",
        "--data-dir", str(tmp_path / "data"),
    ])
    assert rc != 0


def test_existing_transcript_no_overwrite_raises(tmp_path: Path):
    data = tmp_path / "data"
    _make_audio(data / "audio" / "lec.wav")
    transcribe_audio(video_id="lec", provider="mock", data_dir=data)

    with pytest.raises(FileExistsError):
        transcribe_audio(video_id="lec", provider="mock", data_dir=data)


def test_existing_transcript_with_overwrite_replaces(tmp_path: Path):
    data = tmp_path / "data"
    _make_audio(data / "audio" / "lec.wav")
    transcribe_audio(video_id="lec", provider="mock", data_dir=data)

    custom = _FixedProvider(
        [ProviderSegment(start=10.0, end=12.0, text="overwritten.")]
    )
    segments = transcribe_audio(
        video_id="lec",
        provider=custom,
        data_dir=data,
        overwrite=True,
    )

    assert len(segments) == 1
    on_disk = list(
        read_jsonl(data / "transcripts" / "lec.jsonl", TranscriptSegment)
    )
    assert [s.text for s in on_disk] == ["overwritten."]


def test_overwrite_does_not_touch_sibling_transcripts(tmp_path: Path):
    data = tmp_path / "data"
    _make_audio(data / "audio" / "lec.wav")
    transcribe_audio(video_id="lec", provider="mock", data_dir=data)

    sibling = data / "transcripts" / "other.jsonl"
    sibling.write_text(
        '{"video_id": "other", "start_time": 0.0, "end_time": 1.0, "text": "x"}\n',
        encoding="utf-8",
    )
    sibling_before = sibling.read_bytes()

    custom = _FixedProvider(
        [ProviderSegment(start=0.0, end=1.0, text="new")]
    )
    transcribe_audio(
        video_id="lec",
        provider=custom,
        data_dir=data,
        overwrite=True,
    )

    assert sibling.read_bytes() == sibling_before


def test_empty_provider_response_raises(tmp_path: Path):
    data = tmp_path / "data"
    _make_audio(data / "audio" / "lec.wav")
    with pytest.raises(ValueError, match="empty transcript"):
        transcribe_audio(video_id="lec", provider=_FixedProvider([]), data_dir=data)


def test_provider_segment_with_invalid_end_time_raises(tmp_path: Path):
    data = tmp_path / "data"
    _make_audio(data / "audio" / "lec.wav")
    custom = _FixedProvider([ProviderSegment(start=2.0, end=1.0, text="oops")])
    with pytest.raises(ValueError):
        transcribe_audio(video_id="lec", provider=custom, data_dir=data)


def test_provider_segment_with_whitespace_text_raises(tmp_path: Path):
    data = tmp_path / "data"
    _make_audio(data / "audio" / "lec.wav")
    custom = _FixedProvider([ProviderSegment(start=0.0, end=1.0, text="   ")])
    with pytest.raises(ValueError):
        transcribe_audio(video_id="lec", provider=custom, data_dir=data)


def test_unknown_provider_string_raises(tmp_path: Path):
    data = tmp_path / "data"
    _make_audio(data / "audio" / "lec.wav")
    with pytest.raises(ValueError, match="unknown provider"):
        transcribe_audio(
            video_id="lec", provider="not-a-real-provider", data_dir=data
        )


# --- ordering --------------------------------------------------------------


def test_out_of_order_provider_output_is_sorted(tmp_path: Path):
    data = tmp_path / "data"
    _make_audio(data / "audio" / "lec.wav")
    custom = _FixedProvider([
        ProviderSegment(start=10.0, end=12.0, text="third"),
        ProviderSegment(start=0.0, end=1.0, text="first"),
        ProviderSegment(start=5.0, end=6.0, text="second"),
    ])

    segments = transcribe_audio(video_id="lec", provider=custom, data_dir=data)

    assert [s.text for s in segments] == ["first", "second", "third"]
    on_disk = list(
        read_jsonl(data / "transcripts" / "lec.jsonl", TranscriptSegment)
    )
    assert [s.start_time for s in on_disk] == [0.0, 5.0, 10.0]


def test_adjacent_overlap_is_preserved(tmp_path: Path):
    data = tmp_path / "data"
    _make_audio(data / "audio" / "lec.wav")
    custom = _FixedProvider([
        ProviderSegment(start=0.0, end=2.0, text="a"),
        ProviderSegment(start=1.5, end=3.0, text="b"),
    ])

    segments = transcribe_audio(video_id="lec", provider=custom, data_dir=data)

    assert len(segments) == 2
    assert segments[1].start_time < segments[0].end_time


# --- atomic write ----------------------------------------------------------


def test_no_partial_file_on_provider_failure(tmp_path: Path):
    data = tmp_path / "data"
    _make_audio(data / "audio" / "lec.wav")

    with pytest.raises(RuntimeError, match="simulated provider failure"):
        transcribe_audio(
            video_id="lec", provider=_RaisingProvider(), data_dir=data
        )

    transcripts_dir = data / "transcripts"
    assert not (transcripts_dir / "lec.jsonl").exists()
    assert not (transcripts_dir / "lec.jsonl.tmp").exists()


# --- CLI parsing -----------------------------------------------------------


def test_cli_missing_provider_exits_nonzero(tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        main(["--video-id", "lec", "--data-dir", str(tmp_path / "data")])
    assert exc.value.code != 0
