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

### `Chunk`

A fixed-time retrieval unit aligning all modalities on a shared timeline.
Written by Stage 9 as JSONL — one record per line — under
`data/chunks/{video_id}_{chunk_seconds}s.jsonl`. This is custom timestamp
alignment across modalities, not text-splitter document chunking.

| Field               | Type                | Required | Notes                                                       |
| ------------------- | ------------------- | -------- | ----------------------------------------------------------- |
| `chunk_id`          | `str`               | yes      | Non-empty; e.g. `lecture_001_chunk_0007`.                   |
| `video_id`          | `str`               | yes      | Non-empty; joins to manifest.                               |
| `chunk_index`       | `int`               | yes      | `>= 0`. Position of the window in the video.                |
| `start_time`        | `float`             | yes      | Seconds; `>= 0`.                                            |
| `end_time`          | `float`             | yes      | Seconds; `> start_time`. Last chunk ends at duration.       |
| `transcript_text`   | `str`               | yes      | May be empty. Overlapping segments joined by spaces.        |
| `ocr_text`          | `str`               | yes      | May be empty. In-window OCR text joined by spaces.          |
| `vlm_caption`       | `str`               | yes      | May be empty. Overlapping captions joined by spaces.        |
| `frame_paths`       | `list[str]`         | yes      | May be empty. In-window frame paths, ordered by time.       |
| `chunk_seconds`     | `float`             | yes      | `> 0`. Fixed window length used for chunking.               |
| `overlap_seconds`   | `float`             | yes      | `>= 0` and `< chunk_seconds`. Defaults to `0`.              |
| `chunking_strategy` | `Literal["fixed"]`  | yes      | `"fixed"` for the MVP.                                       |

### `EmbeddingRecord`

One vector embedding for an enriched chunk, written by Stage 11 to
`data/embeddings/{video_id}_{chunk_seconds}s_{variant}.jsonl`.

| Field                | Type          | Required | Notes                                   |
| -------------------- | ------------- | -------- | --------------------------------------- |
| `chunk_id`           | `str`         | yes      | Joins to enriched chunk records.        |
| `video_id`           | `str`         | yes      | Non-empty; joins to manifest.           |
| `start_time`         | `float`       | yes      | Seconds; `>= 0`.                        |
| `end_time`           | `float`       | yes      | Seconds; `> start_time`.                |
| `embedding_model`    | `str`         | yes      | Model used to produce the vector.       |
| `embedding_provider` | `str`         | yes      | Provider used to produce the vector.    |
| `embedding_variant`  | `str`         | yes      | Search-text variant embedded.           |
| `vector`             | `list[float]` | yes      | Non-empty embedding vector.             |
| `vector_dim`         | `int`         | yes      | `> 0` and must equal `len(vector)`.     |

### `VectorStoreManifest`

Manifest for a persisted local vector index, written by Stage 12 to
`data/indexes/{video_id}_{chunk_seconds}s_{variant}/vector_store_manifest.json`.

| Field               | Type    | Required | Notes                                      |
| ------------------- | ------- | -------- | ------------------------------------------ |
| `video_id`          | `str`   | yes      | Non-empty; joins to manifest.              |
| `chunk_seconds`     | `float` | yes      | Chunk window size; must be `> 0`.          |
| `embedding_variant` | `str`   | yes      | Variant stored in the index.               |
| `backend`           | `str?`  | no       | Vector store backend (e.g. `chroma`).      |
| `index_path`        | `str?`  | no       | Path to the persisted index directory.     |
| `num_vectors`       | `int`   | yes      | Number of vectors stored; must be `>= 0`.  |
| `vector_dim`        | `int?`  | no       | Embedding dimension if known.              |

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

## Stage 9: Chunking

**Implemented.** Module: [`video_rag/index/build_chunks.py`](../video_rag/index/build_chunks.py).

Inputs:

- `data/manifests/{video_id}/media_metadata.json` (Stage 2 output, **required**)
- `data/transcripts/{video_id}.jsonl` (Stage 4 output, **required**)
- `data/frames/{video_id}/frame_manifest.jsonl` (Stage 5 output, optional)
- `data/ocr/{video_id}.jsonl` (Stage 7 output, optional)
- `data/captions/{video_id}.jsonl` (Stage 8 output, optional)

Output:

- `data/chunks/{video_id}_{chunk_seconds}s.jsonl` containing one `Chunk` per
  line. When `overlap_seconds > 0`, the filename gains a suffix to avoid
  collisions: `data/chunks/{video_id}_{chunk_seconds}s_overlap{overlap_seconds}s.jsonl`.

Fixed-time chunking strategy:

- Windows are derived from `MediaMetadata.duration_seconds`. The default chunk
  size is 30 seconds with `overlap_seconds = 0`.
- Window starts are `0, stride, 2*stride, ...` (where
  `stride = chunk_seconds - overlap_seconds`) while the start is still inside
  the video; each window ends at `min(start + chunk_seconds, duration)`, so the
  final chunk ends exactly at `duration_seconds`, never beyond it.
- A transcript segment or VLM caption is attached when its time range overlaps
  the window (`record.start_time < chunk.end_time and record.end_time >
  chunk.start_time`). An OCR record or frame is attached when its `timestamp`
  falls in `[chunk.start_time, chunk.end_time)`; the final chunk also includes
  records sitting exactly at `duration_seconds`.
- Text fields are joined from the matched records (`transcript_text`,
  `ocr_text`, `vlm_caption`) and `frame_paths` lists the matched frames. Missing
  optional inputs simply yield empty text fields / an empty `frame_paths`; only
  metadata and the transcript are required. No values are hallucinated.

This is **custom timestamp alignment across modalities, not LangChain or
generic text-splitter chunking** — the point of the stage is to line up
transcript, OCR, captions, and frames on the same fixed time grid. It does not
build searchable text, embed, store vectors, retrieve, rerank, answer
questions, or evaluate. Stage 10 turns these chunks into searchable text.

CLI:

```bash
python -m video_rag.index.build_chunks \
  --video-id lecture_001 \
  --chunk-seconds 30
```

Optional flags: `--data-dir` (default `data`), `--overlap-seconds` (default
`0`), `--overwrite` (replaces only the target chunk file).

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

## Stage 11: Embedding

**Implemented.** Module: [`video_rag/index/embed_chunks.py`](../video_rag/index/embed_chunks.py).

Input:

- `data/chunks/{video_id}_{chunk_seconds}s_enriched.jsonl` (Stage 10 enriched
  chunks; Stage 9 chunking + Stage 10 enrichment produce it).

Output:

- `data/embeddings/{video_id}_{chunk_seconds}s_{variant}.jsonl` containing one
  `EmbeddingRecord` per embedded chunk — e.g.
  `data/embeddings/lecture_001_30s_transcript_ocr_vlm.jsonl`.

This stage selects one search-text variant from each enriched chunk, embeds that
text through a provider, and writes embedding records. It does **not** build
searchable text (Stage 10), store vectors in a vector DB / Chroma (Stage 12),
retrieve, rerank, answer questions, or evaluate.

### `EmbeddingRecord`

| Field                | Type             | Required | Notes                                                        |
| -------------------- | ---------------- | -------- | ------------------------------------------------------------ |
| `chunk_id`           | `str`            | yes      | Non-empty; joins back to the source chunk.                   |
| `video_id`           | `str`            | yes      | Non-empty; joins to manifest.                                |
| `start_time`         | `float`          | yes      | Seconds; `>= 0`.                                             |
| `end_time`           | `float`          | yes      | Seconds; `> start_time`.                                     |
| `embedding_model`    | `str`            | yes      | Non-empty; model used (e.g. `text-embedding-3-small`).       |
| `embedding_provider` | `str`            | yes      | Non-empty; provider name (`mock` / `openai`).                |
| `embedding_variant`  | `str`            | yes      | One of the four supported variants below.                    |
| `vector`             | `list[float]`    | yes      | Non-empty embedding vector.                                  |
| `vector_dim`         | `int`            | yes      | `> 0` and must equal `len(vector)`.                          |

### Supported variants

Each variant reads one pre-joined field from the enriched chunk:

| Variant              | Enriched source field             |
| -------------------- | --------------------------------- |
| `transcript_only`    | `combined_text_transcript_only`   |
| `transcript_ocr`     | `combined_text_transcript_ocr`    |
| `transcript_vlm`     | `combined_text_transcript_vlm`    |
| `transcript_ocr_vlm` | `combined_text_all`               |

An unknown variant fails clearly. If the selected field is **absent** from the
records, the stage fails clearly. If the field is **present but empty/whitespace**
for a chunk, that chunk is skipped with a warning and simply produces no
embedding record; Stage 13 validation reports skipped/missing embeddings later.

### Provider strategy

Embedding is delegated to a provider adapter (see
[`video_rag/index/embedding_providers.py`](../video_rag/index/embedding_providers.py)),
mirroring Stage 4's transcription providers:

- **`mock`** — deterministic, offline, configurable small dimension. **Tests and
  smoke checks only** — never use it on real artifacts.
- **`openai`** — real provider via the OpenAI embeddings API (default
  `text-embedding-3-small`, 1536-dim). Lazy-imports `openai` and reads
  `OPENAI_API_KEY` from the environment, so importing the stage and running the
  mock tests never require the dependency or a key. Install with
  `pip install -e .[embed]`.

CLI:

```bash
python -m video_rag.index.embed_chunks \
  --video-id lecture_001 \
  --chunk-seconds 30 \
  --variant transcript_ocr_vlm \
  --provider openai
```

`--provider` is **required**: there is no default, so a mock embedding can never
be produced by accident. Optional flags: `--data-dir` (default `data`),
`--model` (default `text-embedding-3-small`, openai only), `--batch-size`
(default `64`), `--overwrite` (replaces only the target embedding file). A console
script `raggers-embed` is installed as an alias. Vectors are written atomically
(`*.jsonl.tmp` + `os.replace`) in deterministic input-chunk order. **Vector
storage into a vector index happens in Stage 12.**

## Stage 12: Vector Storage

**Implemented.** Module: [`video_rag/index/store_vectors.py`](../video_rag/index/store_vectors.py).

Input:

- `data/embeddings/{video_id}_{chunk_seconds}s_{variant}.jsonl` (Stage 11)
- optional `data/chunks/{video_id}_{chunk_seconds}s_enriched.jsonl` (Stage 9/10) for document text

Output:

- persisted local vector index at `data/indexes/{video_id}_{chunk_seconds}s_{variant}/`
- optional manifest at `data/indexes/{video_id}_{chunk_seconds}s_{variant}/vector_store_manifest.json`

This stage reads embedding records from Stage 11, validates consistent
`video_id`, `embedding_variant`, and vector dimensions, and writes vectors plus
retrieval metadata into a local Chroma collection. Each vector is keyed by
`chunk_id` and stores metadata needed for timestamped evidence:
`chunk_id`, `video_id`, `start_time`, `end_time`, `embedding_variant`, and
`embedding_model`. If enriched chunks are available, the selected variant's
combined search text is attached as the Chroma document.

If the index output already exists, the stage fails unless `overwrite=True`
(or `--overwrite` on the CLI). It does not create embeddings, rerank, answer
questions, or run evaluation. **Index validation happens in Stage 13.**

Install the vector-store dependency with:

```bash
pip install -e ".[vectorstore]"
```

CLI:

```bash
python -m video_rag.index.store_vectors \
    --video-id lecture_001 \
    --chunk-seconds 30 \
    --variant transcript_ocr_vlm
```

## Future modules

Each module adds its own schema in `video_rag/schemas.py` (or a sibling
module) when it lands. Anticipated additions — **not implemented yet** —
include:

- retrieval results, answer payloads.

Each module owner defines the contract for their stage. Don't pre-spec them
here.

## Folder layout

The artifact folder layout is documented in [`../data/README.md`](../data/README.md).
Implemented stages write to `data/videos/`, `data/manifests/`, `data/audio/`,
`data/transcripts/`, `data/frames/`, `data/ocr/`, `data/captions/`,
`data/chunks/`, `data/embeddings/`, `data/indexes/`, and `data/validation/`;
other folders are placeholders.

## Validating artifacts

```bash
python -m video_rag.validate examples/video_manifest.example.json --type video_manifest
python -m video_rag.validate examples/media_metadata.example.json --type media_metadata
python -m video_rag.validate examples/frame_sample.example.jsonl --type frame_sample
python -m video_rag.validate examples/transcript_segment.example.jsonl --type transcript_segments
```

A console script `raggers-validate` is also installed as an alias.

Exit code `0` means valid; non-zero means the file failed schema validation.
