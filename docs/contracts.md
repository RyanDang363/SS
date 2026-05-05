# RAGGERS data contracts (Stage 0-lite)

This document describes the **shared foundation** every RAGGERS module agrees
on. It is intentionally minimal: each downstream module (transcription, frame
extraction, OCR, captions, chunking, embeddings, retrieval, answers) will add
its own schema as it is built.

## What is Stage 0-lite?

Stage 0-lite is the bare minimum needed for the team to collaborate without
stepping on each other:

- A package layout (`video_rag/`) and an artifact folder layout (`data/`).
- JSON / JSONL read/write helpers with Pydantic validation.
- Two starter schemas:
  - `VideoManifest` — registers a source video.
  - `MediaMetadata` — probed media facts for that video.
- A `python -m video_rag.validate` CLI for smoke-checking artifacts.

> Naming note: the project / demo is **RAGGERS**. The Python package is
> deliberately named `video_rag` so imports stay neutral
> (`from video_rag.schemas import VideoManifest`) while branding lives in the
> README and docs.

No transcription, no frames, no OCR, no embeddings, no retrieval — those come
later, each behind their own schema.

## Why bother with contracts?

Different teammates will write different stages. If we don't agree on field
names and types up front, every consumer ends up writing one-off adapters and
the data on disk is impossible to reason about. A small, enforced schema keeps
artifacts interchangeable and makes bugs obvious at the boundary instead of
deep inside a module.

## Shared conventions

These rules apply to **every** artifact, current or future:

- **`video_id`**: every record that refers to a video must include it. It's the
  join key across all artifacts.
- **Timestamps are seconds, as `float`.** No milliseconds, no `HH:MM:SS`
  strings, no frame indices in time fields.
- **Time ranges** use `start_time` and `end_time`.
- **Single points in time** use `timestamp`.
- **Artifacts are JSON or JSONL.** Use JSON for one-record files
  (`VideoManifest`, `MediaMetadata`) and JSONL for streams of records
  (transcript segments, frame samples, chunks, ...).
- **File paths are stored as strings, repo-root-relative.** Paths inside
  manifests and other artifacts (e.g. `source_path`, frame paths, chunk
  paths) are written relative to the repository root and use forward slashes
  — e.g. `data/videos/lecture_001.mp4`, never `C:\Users\...` and never
  `./data/...`. Pipeline commands should be run from the repo root, or
  should resolve these paths against an explicitly configured project root.
  This keeps artifacts portable between machines and CI.
- **Schemas stay minimal.** Add the field you need now; do not pre-design the
  full pipeline.

## Current schemas

### `VideoManifest`

A registered source video. One JSON file per video, typically under
`data/manifests/`.

| Field               | Type        | Required | Notes                              |
| ------------------- | ----------- | -------- | ---------------------------------- |
| `video_id`          | `str`       | yes      | Non-empty.                         |
| `title`             | `str`       | yes      | Non-empty, human-readable.         |
| `source_path`       | `str`       | yes      | Non-empty path to the video file.  |
| `original_filename` | `str?`      | no       | Original on-disk filename, if any. |
| `created_at`        | `str?`      | no       | ISO-8601 timestamp string.         |

Example: [`examples/video_manifest.example.json`](../examples/video_manifest.example.json).

### `MediaMetadata`

Probed media-level facts for a video. One JSON file per video, typically
alongside the manifest.

| Field              | Type     | Required | Notes                         |
| ------------------ | -------- | -------- | ----------------------------- |
| `video_id`         | `str`    | yes      | Non-empty; joins to manifest. |
| `duration_seconds` | `float`  | yes      | Strictly greater than 0.      |
| `has_audio`        | `bool`   | yes      |                               |
| `fps`              | `float?` | no       | If present, > 0.              |
| `width`            | `int?`   | no       | If present, > 0.              |
| `height`           | `int?`   | no       | If present, > 0.              |

Example: [`examples/media_metadata.example.json`](../examples/media_metadata.example.json).

### `FrameSample`

A single sampled frame from a video. One JSONL record per JPEG, written by
Stage 5 to `data/frames/{video_id}/frame_manifest.jsonl`.

| Field             | Type                          | Required | Notes                                                         |
| ----------------- | ----------------------------- | -------- | ------------------------------------------------------------- |
| `video_id`        | `str`                         | yes      | Non-empty; joins to manifest.                                 |
| `timestamp`       | `float`                       | yes      | Seconds, `>= 0`. Single point in time.                        |
| `frame_path`      | `str`                         | yes      | Repo-root-relative POSIX path to the JPEG.                    |
| `thumbnail_path`  | `str?`                        | no       | Reserved for Stage 6; `null` in V0.                           |
| `width`           | `int`                         | yes      | `> 0`. Pixel width of the JPEG.                               |
| `height`          | `int`                         | yes      | `> 0`. Pixel height of the JPEG.                              |
| `sampling_method` | `Literal["fixed_interval"]`   | yes      | Defaults to `"fixed_interval"`. Reserved for future modes.    |

Example: [`examples/frame_sample.example.jsonl`](../examples/frame_sample.example.jsonl).

## Stage 1: Video Registration

**Implemented.** Module: [`video_rag/index/register_video.py`](../video_rag/index/register_video.py).

Input:

- local video path
- optional title
- optional `video_id`

Output:

- registered video file under `data/videos/`
- `VideoManifest` under `data/manifests/{video_id}/video_manifest.json`

`video_id` is sanitized (lowercase, spaces to `_`, only `[a-z0-9_-]` allowed)
and de-duplicated with a numeric suffix when generated from a title or
filename. Modes: `--mode copy` (default) or `--mode symlink`. Existing
artifacts are preserved unless `--overwrite` is passed.

CLI:

```bash
python -m video_rag.index.register_video \
  --video path/to/lecture.mp4 \
  --title "Bayes Lecture" \
  --video-id lecture_001
```

This stage does not inspect codecs, duration, FPS, or audio. Media probing
happens in Stage 2.

## Stage 5: Frame Sampling

**Implemented.** Module: [`video_rag/index/sample_frames.py`](../video_rag/index/sample_frames.py).
Compute kernel: [`cpp/frame_extract/`](../cpp/frame_extract/).

Inputs:

- `data/manifests/{video_id}/video_manifest.json` (Stage 1 output)
- `data/manifests/{video_id}/media_metadata.json` (Stage 2 output)
- the registered video file referenced by `VideoManifest.source_path`

Outputs:

- JPEG frames at `data/frames/{video_id}/frame_NNNNNN.jpg` (zero-padded
  integer seconds)
- `data/frames/{video_id}/frame_manifest.jsonl` containing one `FrameSample`
  per JPEG

Behavior:

- Fixed-interval sampling. Schedule is `[0, interval, 2*interval, ...]`
  strictly less than `duration_seconds`.
- V0 only accepts integer-second `interval_seconds` (default `5`).
- Existing frame outputs are preserved unless `--overwrite` is passed; on
  overwrite, only this video's `data/frames/{video_id}/` is replaced.
- Decoding and JPEG encoding happen in a small C++/OpenCV binary
  (`raggers_frame_extract`) invoked via `subprocess`. Python owns
  manifest IO, schema validation, and overwrite policy.

Build the C++ binary once before running Stage 5:

```bash
cmake -S cpp/frame_extract -B cpp/frame_extract/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/frame_extract/build -j
```

CLI:

```bash
python -m video_rag.index.sample_frames \
  --video-id lecture_001 \
  --interval-seconds 5
```

Optional: override the binary location with `--binary-path`, or set
`RAGGERS_FRAME_EXTRACT_BIN` in the environment.

This stage does not generate thumbnails (Stage 6), OCR text (Stage 7), or
VLM captions (Stage 8). Those stages each consume `frame_manifest.jsonl`
plus the JPEGs as their inputs.

## Future modules

Each module adds its own schema in `video_rag/schemas.py` (or a sibling
module) when it lands. Anticipated additions — **not implemented yet** —
include:

- `TranscriptSegment` (range: `start_time`, `end_time`, `text`).
- `FrameSample` (point: `timestamp`, `frame_path`).
- `OCRRecord`, `CaptionRecord` keyed by `timestamp`.
- `Chunk`, `Embedding`, retrieval results, answer payloads.

Each module owner defines the contract for their stage. Don't pre-spec them
here.

## Folder layout

The artifact folder layout is documented in [`../data/README.md`](../data/README.md).
Stage 0-lite only writes to `data/manifests/` and `data/validation/`; other
folders are placeholders.

## Validating artifacts

```bash
python -m video_rag.validate examples/video_manifest.example.json --type video_manifest
python -m video_rag.validate examples/media_metadata.example.json --type media_metadata
```

A console script `raggers-validate` is also installed as an alias.

Exit code `0` means valid; non-zero means the file failed schema validation.
