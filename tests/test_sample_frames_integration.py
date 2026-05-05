"""Integration smoke test for the real C++ frame extractor.

Skipped automatically unless:
    1. ``raggers_frame_extract`` has been built (default location or
       ``RAGGERS_FRAME_EXTRACT_BIN`` env var), and
    2. ``ffmpeg`` is on ``PATH`` (used to synthesize a tiny fixture video).

To enable locally:
    brew install ffmpeg
    cmake -S cpp/frame_extract -B cpp/frame_extract/build -DCMAKE_BUILD_TYPE=Release
    cmake --build cpp/frame_extract/build -j
    pytest tests/test_sample_frames_integration.py -q

This test exercises the full Stage 5 path end-to-end: real video file, real
C++ binary, real OpenCV decode, real JPEG writes, real Pydantic validation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from video_rag.index import sample_frames as sf
from video_rag.io_utils import read_jsonl, write_json
from video_rag.schemas import FrameSample, MediaMetadata, VideoManifest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BINARY = REPO_ROOT / "cpp" / "frame_extract" / "build" / "raggers_frame_extract"

FIXTURE_DURATION_SECONDS = 3
FIXTURE_FPS = 30
FIXTURE_WIDTH = 64
FIXTURE_HEIGHT = 48
FIXTURE_INTERVAL_SECONDS = 1


def _resolve_real_binary() -> Path | None:
    candidates: list[Path] = []
    env = os.environ.get(sf.BINARY_ENV_VAR)
    if env:
        candidates.append(Path(env))
    candidates.append(DEFAULT_BINARY)
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    return None


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


pytestmark = pytest.mark.skipif(
    _resolve_real_binary() is None or not _have_ffmpeg(),
    reason="raggers_frame_extract binary or ffmpeg not available",
)


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
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def test_real_extractor_produces_expected_frames(tmp_path: Path):
    binary = _resolve_real_binary()
    assert binary is not None

    data_dir = tmp_path / "data"
    video_id = "fixture_int"
    video_path = data_dir / "videos" / f"{video_id}.mp4"
    _synthesize_fixture_video(video_path)

    write_json(
        data_dir / "manifests" / video_id / "video_manifest.json",
        VideoManifest(
            video_id=video_id,
            title="Integration fixture",
            source_path=f"{data_dir.name}/videos/{video_id}.mp4",
        ),
    )
    write_json(
        data_dir / "manifests" / video_id / "media_metadata.json",
        MediaMetadata(
            video_id=video_id,
            duration_seconds=float(FIXTURE_DURATION_SECONDS),
            fps=float(FIXTURE_FPS),
            width=FIXTURE_WIDTH,
            height=FIXTURE_HEIGHT,
            has_audio=False,
        ),
    )

    samples = sf.sample_frames(
        video_id=video_id,
        data_dir=data_dir,
        interval_seconds=FIXTURE_INTERVAL_SECONDS,
        binary_path=binary,
    )

    expected_timestamps = [
        float(t) for t in range(0, FIXTURE_DURATION_SECONDS, FIXTURE_INTERVAL_SECONDS)
    ]
    assert [s.timestamp for s in samples] == expected_timestamps
    assert all(s.video_id == video_id for s in samples)
    assert all(s.sampling_method == "fixed_interval" for s in samples)
    assert all(s.thumbnail_path is None for s in samples)
    assert all(s.width == FIXTURE_WIDTH and s.height == FIXTURE_HEIGHT for s in samples)

    for sample in samples:
        absolute = tmp_path / sample.frame_path
        assert absolute.is_file()
        assert absolute.stat().st_size > 0

    manifest_out = data_dir / "frames" / video_id / "frame_manifest.jsonl"
    loaded = list(read_jsonl(manifest_out, FrameSample))
    assert loaded == samples
