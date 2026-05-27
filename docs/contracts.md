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
- Starter schemas:
  - `VideoManifest` — registers a source video.
  - `MediaMetadata` — probed media facts for that video.
  - `FrameSample` — a sampled frame from a video.
  - `TranscriptSegment` — a timestamped transcript segment.
  - `OCRResult` — OCR text detected for a sampled frame.
  - `VLMCaption` — a generic caption for sampled frame windows.
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

### `VLMCaption`

A generic visual caption for a group of sampled frames. Records are written as
JSONL by Stage 8.

| Field          | Type        | Required | Notes                         |
| -------------- | ----------- | -------- | ----------------------------- |
| `video_id`     | `str`       | yes      | Non-empty; joins to manifest. |
| `start_time`   | `float`     | yes      | Seconds, must be >= 0.        |
| `end_time`     | `float`     | yes      | Seconds, must be >= 0.        |
| `frame_paths`  | `list[str]` | yes      | Non-empty frame path list.    |
| `caption`      | `str`       | yes      | Non-empty caption text.       |
| `caption_type` | `str`       | yes      | `generic` for Stage 8.        |
| `model`        | `str`       | yes      | VLM model name.               |

### `OCRResult`

OCR text detected for a sampled frame. Records are written as JSONL by Stage 7.

| Field        | Type      | Required | Notes                         |
| ------------ | --------- | -------- | ----------------------------- |
| `video_id`   | `str`     | yes      | Non-empty; joins to manifest. |
| `timestamp`  | `float`   | yes      | Seconds, must be >= 0.        |
| `frame_path` | `str`     | yes      | Non-empty path to the frame.  |
| `ocr_text`   | `str`     | yes      | Empty string when no text.    |
| `confidence` | `float?`  | no       | If present, 0 through 1.      |

### `TranscriptSegment`

One timestamped transcript segment for a video. Stored as JSONL — one
record per line — under `data/transcripts/{video_id}.jsonl`.

| Field        | Type    | Required | Notes                                                  |
| ------------ | ------- | -------- | ------------------------------------------------------ |
| `video_id`   | `str`   | yes      | Non-empty; joins to manifest.                          |
| `start_time` | `float` | yes      | Seconds; `>= 0`.                                       |
| `end_time`   | `float` | yes      | Seconds; `> start_time`.                               |
| `text`       | `str`   | yes      | Non-empty after `strip()`.                             |

File-level rules enforced by `python -m video_rag.validate`:

- All records in a file share the same `video_id`.
- Records are ordered by non-decreasing `start_time`. Small overlaps
  between adjacent segments are permitted because real ASR output
  often produces them.

Example: [`examples/transcript_segment.example.jsonl`](../examples/transcript_segment.example.jsonl).

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

## Stage 2: Media Probe

**Implemented.** Module: [`video_rag/index/probe_media.py`](../video_rag/index/probe_media.py).

Input:

- `data/manifests/{video_id}/video_manifest.json`

Output:

- `data/manifests/{video_id}/media_metadata.json`

This stage reads `VideoManifest.source_path` and resolves manifest source paths
as repo-root-relative values. It uses `ffprobe` to inspect duration, FPS,
resolution, and audio presence, then writes validated `MediaMetadata`. It does
not modify `video_manifest.json`, extract audio, or perform later indexing
steps.

## Stage 3: Audio Extraction

**Implemented.** Module: [`video_rag/index/extract_audio.py`](../video_rag/index/extract_audio.py).

Inputs:

- `data/manifests/{video_id}/video_manifest.json`
- `data/manifests/{video_id}/media_metadata.json`

Output:

- `data/audio/{video_id}.wav`

This stage requires `has_audio=true` in `media_metadata.json`. It extracts mono
16 kHz WAV audio for transcription, but does not transcribe; transcription
happens in Stage 4. It does not modify Stage 1 or Stage 2 artifacts. Manifest
source paths are repo-root-relative, so `source_path` values such as
`data/videos/lecture_001.mp4` are resolved from the repository root/current
working directory, not from the manifest folder.

## Stage 4: Transcription

**Implemented.** Module: [`video_rag/index/transcribe_audio.py`](../video_rag/index/transcribe_audio.py).

Input:

- audio file at `data/audio/{video_id}.wav` (produced by Stage 3)

Output:

- JSONL of `TranscriptSegment` records at
  `data/transcripts/{video_id}.jsonl`

Transcription is delegated to a provider adapter (see
[`video_rag/index/transcription_providers.py`](../video_rag/index/transcription_providers.py)).
The stage attaches `video_id` to each provider segment, sorts by
`start_time` (stable), validates each record against the schema, and
writes the result atomically (`*.jsonl.tmp` + `os.replace`).

Built-in providers:

- **`mock`** — deterministic, offline. **Tests and smoke checks only** —
  do not run on real artifacts.
- **`openai`** — real provider via OpenAI `whisper-1`. Lazy-imports
  `openai` and reads `OPENAI_API_KEY` from the environment. Install
  with `pip install -e .[transcribe]`.

CLI:

```bash
python -m video_rag.index.transcribe_audio \
  --video-id lecture_001 \
  --provider openai \
  [--language en] \
  [--overwrite]
```

`--provider` is **required**: there is no default, so a mock transcript
can never be produced by accident. A console-script alias
`raggers-transcribe` is installed.

Existing transcripts are preserved unless `--overwrite` is passed, and
overwrite replaces only the single target file in `data/transcripts/` —
sibling transcripts for other videos are never touched.

## Stage 5: Frame Sampling

**Implemented.** Module: [`video_rag/index/sample_frames.py`](../video_rag/index/sample_frames.py).
Compute: `opencv-python-headless` (Python). Optional C++ kernel:
[`cpp/frame_extract/`](../cpp/frame_extract/).

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
- Decoding and JPEG encoding are performed by `opencv-python-headless` (a
  regular Python dep — no build step required). Python owns all of manifest
  IO, schema validation, and overwrite policy.

No build step required for V0. Simply install the Python package:

```bash
pip install -e ".[dev]"
```

CLI:

```bash
python -m video_rag.index.sample_frames \
  --video-id lecture_001 \
  --interval-seconds 5
```

Optional flags: `--quality` (JPEG quality 1-100, default 85),
`--overwrite`, `--data-dir`.

An optional high-throughput C++ extractor (`cpp/frame_extract/`) is kept in
the repo and shares the same output contract. See
[`cpp/frame_extract/README.md`](../cpp/frame_extract/README.md) if you want
to build and use it instead.

This stage does not generate thumbnails (Stage 6), OCR text (Stage 7), or
VLM captions (Stage 8). Those stages each consume `frame_manifest.jsonl`
plus the JPEGs as their inputs.

## Stage 7: OCR Extraction

**Implemented.** Module: [`video_rag/index/run_ocr.py`](../video_rag/index/run_ocr.py).

Input:

- `data/frames/{video_id}/frame_manifest.jsonl`

Output:

- `data/ocr/{video_id}.jsonl`

This stage reads sampled frame records from Stage 5, runs OCR on each
referenced frame, and writes timestamped OCR results. If no text is found, it
writes a valid OCR record with empty `ocr_text` and `confidence: null`. It does
not generate VLM captions, chunk, embed, retrieve, answer questions, or modify
frame manifests or video manifests.

## Stage 8: VLM Frame Captioning

**Implemented.** Module: [`video_rag/index/caption_frames.py`](../video_rag/index/caption_frames.py).

Input:

- `data/frames/{video_id}/frame_manifest.jsonl`

Output:

- `data/captions/{video_id}.jsonl`

This stage reads sampled frame records from Stage 5, groups frames into caption
windows, and uses a VLM to generate generic indexing-time visual captions. The
default model is `gpt-4o-mini`. It does not OCR, chunk, embed, retrieve, answer
questions, or modify frame manifests, OCR outputs, or video manifests.
Query-aware captioning is a future retrieval-time enhancement.

## Stage 10: Search Text Construction

**Implemented.** Module: [`video_rag/index/build_search_text.py`](../video_rag/index/build_search_text.py).

Input:

- `data/chunks/{video_id}_{chunk_seconds}s.jsonl`

Output:

- `data/chunks/{video_id}_{chunk_seconds}s_enriched.jsonl`

This stage reads Stage 9 timestamped chunks and adds search-text variants for
retrieval experiments. Variants are transcript_only, transcript_ocr,
transcript_vlm, and transcript_ocr_vlm. It preserves chunk timing, frame
paths, and original metadata. It does not embed, store vectors, retrieve,
rerank, answer, or modify Stage 9 chunk timing or artifact paths. Embedding
happens in Stage 11.

CLI:

```bash
python -m video_rag.index.build_search_text \
  --video-id lecture_001 \
  --chunk-seconds 30
```

## Future modules

Each module adds its own schema in `video_rag/schemas.py` (or a sibling
module) when it lands. Anticipated additions — **not implemented yet** —
include:

- `Embedding`, retrieval results, answer payloads.

Each module owner defines the contract for their stage. Don't pre-spec them
here.

## Folder layout

The artifact folder layout is documented in [`../data/README.md`](../data/README.md).
Implemented stages write to `data/videos/`, `data/manifests/`, `data/audio/`,
`data/transcripts/`, `data/frames/`, `data/ocr/`, `data/captions/`, and
`data/validation/`; other folders are placeholders.

## Validating artifacts

```bash
python -m video_rag.validate examples/video_manifest.example.json --type video_manifest
python -m video_rag.validate examples/media_metadata.example.json --type media_metadata
python -m video_rag.validate examples/frame_sample.example.jsonl --type frame_sample
python -m video_rag.validate examples/transcript_segment.example.jsonl --type transcript_segments
```

A console script `raggers-validate` is also installed as an alias.

Exit code `0` means valid; non-zero means the file failed schema validation.
