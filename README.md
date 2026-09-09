# RAGGERS

RAGGERS is a staged video RAG project for uploading a video, transcribing it,
building multimodal retrieval artifacts, searching timestamped evidence, and
asking grounded questions over the indexed video.

The current app includes:

- artifact-first Python indexing stages,
- local Chroma vector storage,
- retrieval and grounded answer generation,
- a FastAPI backend,
- a bundled browser UI,
- Docker and Render deployment packaging.

Shared schemas and stage contracts live in [`docs/contracts.md`](docs/contracts.md).
Deployment details live in [`docs/deploy.md`](docs/deploy.md).

## Prerequisites

| Tool | Used for |
| ---- | -------- |
| Python 3.10+ | Package, API, CLI, tests |
| ffmpeg / ffprobe | Audio extraction, media probing, frame sampling |
| OpenAI API key | Real transcription, embeddings, captions, and answers |
| Docker | Containerized hosting path |

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api,transcribe,embed,answer,vectorstore]"
```

Optional OCR support:

```bash
pip install -e ".[ocr]"
```

Set provider credentials for real runs:

```bash
cp .env.example .env
# edit OPENAI_API_KEY
```

## Run The App

```bash
python -m video_rag.api.app --host 127.0.0.1 --port 8000 --data-dir data
```

Open:

```text
http://127.0.0.1:8000
```

The UI supports upload, indexing, job polling, artifact status, retrieval-only
search, grounded answers, and frame evidence previews.

## CLI Flow

Full local index:

```bash
python -m video_rag.index.run_pipeline \
  --video path/to/lecture.mp4 \
  --title "Bayes Lecture" \
  --video-id lecture_001 \
  --transcription-provider openai \
  --embedding-provider openai \
  --embedding-variant transcript_ocr_vlm
```

Search:

```bash
python -m video_rag.search.retrieve \
  --video-id lecture_001 \
  --question "What formula was shown?" \
  --provider openai
```

Answer:

```bash
python -m video_rag.search.answer \
  --video-id lecture_001 \
  --question "What formula was shown?" \
  --retrieval-provider openai \
  --answer-provider openai
```

## Tests

```bash
pytest -q
```

Integration tests that need `ffmpeg` auto-skip when the binary is missing.

## Docker

```bash
docker build -t raggers .
docker run --rm \
  -p 8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -v "$PWD/data:/app/data" \
  raggers
```

Build with OCR dependencies:

```bash
docker build --build-arg INSTALL_OCR=true -t raggers:ocr .
```

## Project Layout

```text
video_rag/              Python package, API, UI assets, schemas, stages
docs/contracts.md       Artifact schemas and stage contracts
docs/deploy.md          Hosting and Docker notes
data/                   Pipeline artifacts, gitignored except layout files
tests/                  pytest suite
examples/               Sample JSON/JSONL for validate CLI
cpp/frame_extract/      Optional C++ OpenCV extractor
```
