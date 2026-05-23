# RAGGERS

Staged video indexing pipeline: register videos, extract artifacts (metadata,
audio, frames, transcripts, …), and build toward retrieval. Shared contracts
live in [`docs/contracts.md`](docs/contracts.md).

This README covers **local setup** so you can run tests and **Stage 5** (frame
sampling). Other stages are documented in contracts as they land.

## Prerequisites

| Tool | Used for |
|------|----------|
| Python 3.10+ | Orchestration, schemas, CLI, tests |
| ffmpeg (optional) | Integration test fixture video only |

No C++ compiler or CMake required to run Stage 5. A standalone C++ extractor
(`cpp/frame_extract/`) is kept in the repo as an optional high-throughput
alternative — see [`cpp/frame_extract/README.md`](cpp/frame_extract/README.md)
if you want to build it.

## 1. Python environment

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

This installs all runtime deps (including `opencv-python-headless` for Stage 5
frame decoding) and `pytest` for tests.

Installed console scripts:

- `raggers-validate` — validate JSON/JSONL artifacts
- `raggers-sample-frames` — Stage 5 frame sampling (alias for the module CLI)

## 2. Run tests

All tests (integration test auto-skips if `ffmpeg` is not on `PATH`):

```bash
pytest -q
```

Stage 5 only:

```bash
pytest -k "sample_frames or frame_sample" -q
```

Full end-to-end integration test (needs `ffmpeg` to synthesize a tiny fixture
video, then exercises real OpenCV decode + JPEG writes):

```bash
# macOS
brew install ffmpeg

# Linux
apt install ffmpeg   # or dnf install ffmpeg

pytest tests/test_sample_frames_integration.py -v
```

Expect `106 passed` when `ffmpeg` is available (otherwise `105 passed,
1 skipped`).

## 3. Run Stage 5 on a video

Stage 5 depends on **Stage 1** (register) and **Stage 2** (probe metadata).

```bash
# Stage 1 — register a local video (example)
python -m video_rag.index.register_video \
  --video path/to/lecture.mp4 \
  --title "Bayes Lecture" \
  --video-id lecture_001

# Stage 2 — not in this repo yet; you need media_metadata.json from that stage.
# See docs/contracts.md for the expected shape (examples/media_metadata.example.json).

# Stage 5 — sample frames every 5 seconds
python -m video_rag.index.sample_frames \
  --video-id lecture_001 \
  --interval-seconds 5
```

Outputs:

```text
data/frames/lecture_001/frame_000000.jpg
data/frames/lecture_001/frame_000005.jpg
...
data/frames/lecture_001/frame_manifest.jsonl
```

Validate the manifest:

```bash
raggers-validate data/frames/lecture_001/frame_manifest.jsonl --type frame_sample
```

Use `--overwrite` to replace existing frame outputs for that video.

## Project layout (quick reference)

```text
video_rag/              Python package (schemas, stages, validate CLI)
cpp/frame_extract/      C++ OpenCV extractor (optional, not required for V0)
docs/contracts.md       Artifact schemas and stage contracts
data/                   Pipeline artifacts (gitignored contents)
tests/                  pytest suite
examples/               Sample JSON/JSONL for validate CLI
```

## Further reading

- [Data contracts & all stages](docs/contracts.md)
- [C++ extractor build details](cpp/frame_extract/README.md)
- [Artifact folder layout](data/README.md)
