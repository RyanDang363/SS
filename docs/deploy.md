# Deploying RAGGERS

Stage 19 packages the FastAPI app and bundled UI for a hosted runtime. The
deployment target needs Python, `ffmpeg`/`ffprobe`, persistent storage for
`data/`, and provider credentials.

## Runtime

Required environment variables:

| Name | Purpose |
| ---- | ------- |
| `OPENAI_API_KEY` | Required for OpenAI transcription, embeddings, captions, and answers. |
| `DATA_DIR` | Persistent artifact root. Defaults to `data` locally and `/app/data` in Docker. |
| `HOST` | Bind host. Defaults to `127.0.0.1` locally and `0.0.0.0` in Docker. |
| `PORT` | Bind port. Defaults to `8000`. Most hosts inject this automatically. |

The V0 service stores uploads, manifests, transcripts, frames, chunks,
embeddings, indexes, validation reports, and job records under `DATA_DIR`.
Mount this path on persistent disk. Without persistence, uploaded videos and
indexes disappear when the container restarts.

## Docker

Build the default image:

```bash
docker build -t raggers .
```

Run locally:

```bash
docker run --rm \
  -p 8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -v "$PWD/data:/app/data" \
  raggers
```

Open:

```text
http://127.0.0.1:8000
```

The default Docker image installs API, transcription, embedding, answering, and
vector-store dependencies. OCR is intentionally opt-in because `easyocr` brings
large model/runtime dependencies:

```bash
docker build --build-arg INSTALL_OCR=true -t raggers:ocr .
```

If OCR is not installed, use the UI's "Skip OCR" indexing option or deploy the
OCR image.

## Render

The repo includes `render.yaml` for a Docker-backed Render web service with a
persistent disk mounted at `/app/data`.

1. Create a new Render Blueprint from this repository.
2. Set `OPENAI_API_KEY` as a secret environment variable.
3. Keep the persistent disk mounted at `/app/data`.
4. Deploy and wait for `/health` to pass.

The same Docker image can be used on Railway, Fly.io, or a VM. The essential
requirements are the same: expose the injected `PORT`, mount `DATA_DIR`
persistently, and provide `OPENAI_API_KEY`.

## Smoke Checks

After deploy:

```bash
curl "$APP_URL/health"
```

Expected:

```json
{"status":"ok"}
```

Then open `$APP_URL`, upload a short video, start indexing, and ask a question
after validation succeeds. For a low-cost smoke run, choose mock providers and
skip OCR/captions; for real transcription and answering, use OpenAI providers.
