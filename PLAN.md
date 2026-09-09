# RAGGERS Completion Plan

## Current State

`main` now contains a staged video indexing pipeline through Stage 13:

1. Register a video into `data/videos/` and write a `VideoManifest`.
2. Probe media metadata with `ffprobe`.
3. Extract mono 16 kHz WAV audio with `ffmpeg`.
4. Transcribe audio into timestamped `TranscriptSegment` JSONL.
5. Sample video frames into JPEGs and a frame manifest.
6. Run OCR over sampled frames.
7. Generate VLM captions for groups of sampled frames.
8. Build fixed-time multimodal chunks.
9. Build enriched searchable text variants.
10. Embed enriched chunks.
11. Store embeddings in a local Chroma vector index.
12. Validate the built index.

The core architecture is artifact-first. Each stage reads JSON or JSONL from
`data/`, validates records with Pydantic schemas, writes the next artifact, and
keeps stage boundaries explicit.

Current important modules:

- `video_rag/index/register_video.py`
- `video_rag/index/probe_media.py`
- `video_rag/index/extract_audio.py`
- `video_rag/index/transcribe_audio.py`
- `video_rag/index/sample_frames.py`
- `video_rag/index/run_ocr.py`
- `video_rag/index/caption_frames.py`
- `video_rag/index/build_chunks.py`
- `video_rag/index/build_search_text.py`
- `video_rag/index/embed_chunks.py`
- `video_rag/index/store_vectors.py`
- `video_rag/index/validate_index.py`

The main gap is productization: the current repo can build index artifacts, but
it does not yet provide a hosted app, upload flow, retrieval API, answer
generation, or a full pipeline runner that includes Stages 9-13.

## Target Product

Build a hosted video RAG app where a user can:

1. Upload or register a video.
2. Run transcription and multimodal indexing.
3. See indexing progress and stage failures.
4. Ask questions about the video.
5. Receive grounded answers with timestamps, evidence text, and frame paths.
6. Reuse existing artifacts without rerunning expensive stages unless requested.

## Recommended Architecture

Use the existing Python package as the backend indexing engine and add a thin
application layer around it.

Backend responsibilities:

- Accept video uploads.
- Create a job record.
- Run the indexing pipeline stage by stage.
- Persist artifacts under `data/`.
- Expose job status, artifacts, retrieval, and answer endpoints.

Frontend responsibilities:

- Upload video.
- Show indexing progress.
- Display transcript/chunk/index status.
- Ask questions.
- Render answers with timestamp citations and frame evidence.

Initial hosting target should be a single-process app for simplicity:

- FastAPI backend.
- Local `data/` artifact storage.
- Local Chroma index.
- Simple frontend served by the app or a lightweight separate web UI.

Later production hosting can move artifacts to object storage and jobs to a
worker queue.

## Stage A: Make The Pipeline Truly End-To-End

Branch name:

```bash
stage-14-full-index-pipeline
```

Status:

```text
Complete and merged into main.
```

Goal:

Update `video_rag/index/run_pipeline.py` so the CLI can build a complete
searchable index, not just extraction artifacts.

Current `run_pipeline.py` stops after:

```text
register -> probe -> audio -> transcribe -> frames -> OCR -> captions
```

It should become:

```text
register -> probe -> audio -> transcribe -> frames -> OCR -> captions
-> build_chunks -> build_search_text -> embed_chunks -> store_vectors
-> validate_index
```

Implementation tasks:

- Import `build_chunks`, `build_search_text`, `embed_chunks`, `store_vectors`,
  and `validate_index`.
- Add CLI options:
  - `--chunk-seconds`
  - `--overlap-seconds`
  - `--embedding-provider`
  - `--embedding-model`
  - `--embedding-variant`
  - `--embedding-batch-size`
  - `--vector-backend`
  - `--skip-ocr`
  - `--skip-captions`
  - `--skip-vector-store`
  - `--validate-only` or a separate validation command path
- Preserve the current `--provider` for transcription, or rename it to
  `--transcription-provider` while keeping backwards compatibility.
- Return or print the final validation report path.
- Keep `overwrite=True` for the current smoke runner behavior, but add a later
  issue to support resumable/idempotent runs.

Definition of done:

- `python -m video_rag.index.run_pipeline` can produce chunks, enriched chunks,
  embeddings, a vector index, and a validation report.
- Tests cover stage call order and argument forwarding.
- Existing extraction tests still pass.

## Stage B: Retrieval API

Branch name:

```bash
stage-15-retrieval
```

Status:

```text
In progress on the current branch.
```

Goal:

Add query-time retrieval over the local vector index.

New modules:

- `video_rag/search/__init__.py`
- `video_rag/search/retrieve.py`

Core flow:

1. Embed the user question with the same provider/model family used for chunks.
2. Open the Chroma collection for `{video_id}_{chunk_seconds}s_{variant}`.
3. Query top-k nearest chunks.
4. Load enriched chunk records from `data/chunks/`.
5. Return structured retrieval results with score, timestamps, text fields, and
   frame paths.

Suggested result shape:

```json
{
  "chunk_id": "lecture_001_chunk_0007",
  "video_id": "lecture_001",
  "score": 0.82,
  "start_time": 210.0,
  "end_time": 240.0,
  "transcript_text": "...",
  "ocr_text": "...",
  "vlm_caption": "...",
  "frame_paths": ["data/frames/lecture_001/frame_000210.jpg"]
}
```

Definition of done:

- Querying a built local index returns top-k chunk evidence.
- Retrieval works with a mock embedding provider in tests.
- Retrieval validates missing index, missing chunks, and variant mismatch
  clearly.

## Stage C: Answer Generation

Branch name:

```bash
stage-16-grounded-answering
```

Status:

```text
In progress on the current branch.
```

Goal:

Generate answers from retrieved chunks with timestamp citations.

New modules:

- `video_rag/search/answer.py`
- `video_rag/search/prompts.py`

Answer rules:

- Use only retrieved video evidence.
- Cite timestamps in every factual claim when possible.
- Distinguish spoken transcript, OCR text, and visual captions when relevant.
- If evidence is insufficient, say the answer was not found in the indexed
  video.

Suggested answer payload:

```json
{
  "video_id": "lecture_001",
  "question": "What formula was shown for Bayes theorem?",
  "answer": "The slide shows P(A|B) = P(B|A)P(A)/P(B) [03:30-04:00].",
  "citations": [
    {
      "chunk_id": "lecture_001_chunk_0007",
      "start_time": 210.0,
      "end_time": 240.0,
      "frame_paths": ["data/frames/lecture_001/frame_000210.jpg"]
    }
  ]
}
```

Definition of done:

- CLI can answer a question from an indexed video.
- Tests use a fake LLM provider.
- Prompt rejects unsupported claims.

## Stage D: Backend Service

Branch name:

```bash
stage-17-api-server
```

Goal:

Expose upload, indexing, status, retrieval, and answer generation over HTTP.

Recommended framework:

- FastAPI
- Uvicorn

Endpoints:

- `POST /videos`
  - Upload video.
  - Register it.
  - Return `video_id`.
- `POST /videos/{video_id}/index`
  - Start indexing.
  - Return job id.
- `GET /jobs/{job_id}`
  - Return stage, status, error, timestamps.
- `GET /videos/{video_id}`
  - Return manifest, metadata, available artifacts, validation status.
- `POST /videos/{video_id}/search`
  - Return top-k retrieval chunks.
- `POST /videos/{video_id}/answer`
  - Return final grounded answer.
- `GET /artifacts/{path}`
  - Serve frames or thumbnails with path safety checks.

Job storage for V0:

- JSON job records under `data/jobs/`.
- In-process background tasks are acceptable for local demo.

Production follow-up:

- Move job execution to RQ, Celery, Dramatiq, or a hosted worker.
- Move videos and frames to object storage.
- Move metadata to Postgres.

Definition of done:

- Local server accepts a video and runs indexing.
- Server can report progress and return answer results.
- Tests cover endpoint behavior with mocked expensive stages.

## Stage E: Frontend

Branch name:

```bash
stage-18-hosted-ui
```

Goal:

Build a usable web UI for upload, indexing progress, and question answering.

Core screens:

- Video upload screen.
- Video detail screen.
- Indexing progress/status.
- Transcript/chunk summary.
- Ask panel.
- Answer panel with citations.
- Evidence panel showing timestamps, OCR text, transcript text, captions, and
  frame previews.

Minimum UX:

- User uploads a video.
- User clicks index.
- UI shows current stage.
- User asks a question after validation passes.
- UI shows answer and clickable timestamp evidence.

Definition of done:

- App can be run locally against the FastAPI backend.
- UI handles loading, failed jobs, empty answers, and missing evidence.
- No manual artifact inspection is required for the basic demo.

## Stage F: Hosting

Branch name:

```bash
stage-19-deploy
```

Goal:

Deploy the app so it is usable from a browser.

V0 hosting options:

- Render, Railway, Fly.io, or a small VM for the FastAPI app.
- Persistent disk for `data/` if using local artifacts.
- Environment variables for OpenAI API keys.

Important hosting constraints:

- Video uploads can be large.
- `ffmpeg` and `ffprobe` must be installed in the runtime image.
- OCR and Chroma dependencies are heavy.
- Long-running indexing should not block a request.
- Local disk persistence must be explicit.

Recommended deployment path:

1. Add a Dockerfile with Python dependencies and `ffmpeg`.
2. Add runtime env var documentation.
3. Add health check endpoint.
4. Add persistent volume mount for `data/`.
5. Deploy backend.
6. Deploy frontend or serve static frontend from backend.

Definition of done:

- A user can open a hosted URL.
- Upload a video.
- Trigger indexing.
- Ask questions after indexing completes.
- Receive cited answers.

## Stage G: Production Hardening

After the hosted demo works, harden the system:

- Resumable pipeline runs.
- Per-stage artifact existence checks.
- Job cancellation.
- File size limits.
- Video type validation.
- Secure artifact serving.
- Rate limiting.
- Model/provider configuration.
- Better chunking experiments.
- Retrieval evaluation set.
- Answer quality evals.
- Thumbnail generation.
- User/project isolation.
- Cloud object storage.
- Queue-backed workers.
- Database-backed job and video metadata.

## Immediate Next Step

After merging `stage-16-grounded-answering`, start:

```bash
git checkout -b stage-17-api-server
```

Then implement Stage D.

This is the right next branch because the core local RAG flow will be complete:
indexing, retrieval, and grounded answers. The next product capability is
exposing that flow through upload, job status, search, and answer endpoints.
