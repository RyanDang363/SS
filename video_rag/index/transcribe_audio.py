"""Stage 4: transcribe an extracted audio file into timestamped segments.

Reads the WAV produced by Stage 3 from ``{data_dir}/audio/{video_id}.wav``
and writes a ``TranscriptSegment`` JSONL file to
``{data_dir}/transcripts/{video_id}.jsonl``.

Transcription itself is delegated to a :class:`TranscriptionProvider`
(see :mod:`video_rag.index.transcription_providers`); this stage attaches
``video_id``, sorts segments by ``start_time``, validates them as
``TranscriptSegment``, and writes the result atomically.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from video_rag.index.transcription_providers import (
    PROVIDERS,
    ProviderSegment,
    TranscriptionProvider,
    get_provider,
)
from video_rag.io_utils import write_jsonl
from video_rag.schemas import TranscriptSegment

PathLike = str | Path


def _default_audio_path(data_dir: Path, video_id: str) -> Path:
    return data_dir / "audio" / f"{video_id}.wav"


def _default_transcript_path(data_dir: Path, video_id: str) -> Path:
    return data_dir / "transcripts" / f"{video_id}.jsonl"


def _resolve_provider(provider: TranscriptionProvider | str) -> TranscriptionProvider:
    if isinstance(provider, str):
        return get_provider(provider)
    return provider


def _adapt_segments(
    raw: list[ProviderSegment],
    video_id: str,
) -> list[TranscriptSegment]:
    raw_sorted = sorted(raw, key=lambda s: s.start)
    segments: list[TranscriptSegment] = []
    for i, ps in enumerate(raw_sorted):
        try:
            seg = TranscriptSegment(
                video_id=video_id,
                start_time=ps.start,
                end_time=ps.end,
                text=ps.text,
            )
        except ValidationError as e:
            raise ValueError(
                f"provider segment {i} failed schema validation: {e}"
            ) from e
        segments.append(seg)
    return segments


def _atomic_write_jsonl(path: Path, records: list[TranscriptSegment]) -> None:
    tmp = path.parent / (path.name + ".tmp")
    try:
        write_jsonl(tmp, records)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def transcribe_audio(
    video_id: str,
    audio_path: PathLike | None = None,
    transcript_path: PathLike | None = None,
    provider: TranscriptionProvider | str = "mock",
    language: str | None = None,
    overwrite: bool = False,
    data_dir: PathLike = "data",
) -> list[TranscriptSegment]:
    """Transcribe a Stage-3 audio file. Returns persisted ``TranscriptSegment``s."""
    if not video_id or not video_id.strip():
        raise ValueError("video_id must be non-empty")

    data_root = Path(data_dir)
    audio = (
        Path(audio_path)
        if audio_path is not None
        else _default_audio_path(data_root, video_id)
    )
    transcript = (
        Path(transcript_path)
        if transcript_path is not None
        else _default_transcript_path(data_root, video_id)
    )

    if not audio.exists() or not audio.is_file():
        raise FileNotFoundError(f"audio file not found or not a file: {audio}")

    if transcript.exists() and not overwrite:
        raise FileExistsError(
            f"transcript already exists: {transcript} (pass overwrite=True)"
        )

    resolved = _resolve_provider(provider)
    raw = resolved.transcribe(audio, language=language)

    if not raw:
        raise ValueError(
            f"empty transcript: provider {resolved.name!r} returned no segments"
        )

    segments = _adapt_segments(raw, video_id)
    _atomic_write_jsonl(transcript, segments)
    return segments


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_rag.index.transcribe_audio",
        description="Transcribe an extracted audio file into timestamped segments (Stage 4).",
    )
    p.add_argument("--video-id", dest="video_id", required=True, help="Video identifier.")
    p.add_argument(
        "--provider",
        required=True,
        choices=sorted(PROVIDERS.keys()),
        help="Transcription provider (mock is for testing only).",
    )
    p.add_argument("--audio", default=None, type=Path, help="Override audio path.")
    p.add_argument(
        "--transcript",
        default=None,
        type=Path,
        help="Override transcript output path.",
    )
    p.add_argument("--language", default=None, help="Optional language hint passed to the provider.")
    p.add_argument("--overwrite", action="store_true", help="Replace existing transcript file.")
    p.add_argument(
        "--data-dir",
        dest="data_dir",
        type=Path,
        default=Path("data"),
        help="Artifact root (default: data).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        segments = transcribe_audio(
            video_id=args.video_id,
            audio_path=args.audio,
            transcript_path=args.transcript,
            provider=args.provider,
            language=args.language,
            overwrite=args.overwrite,
            data_dir=args.data_dir,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError) as e:
        print(f"FAIL  {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1

    transcript_path = (
        Path(args.transcript)
        if args.transcript is not None
        else _default_transcript_path(Path(args.data_dir), args.video_id)
    )
    print("Wrote transcript:")
    print(f"  video_id: {args.video_id}")
    print(f"  provider: {args.provider}")
    print(f"  segments: {len(segments)}")
    print(f"  output: {transcript_path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
