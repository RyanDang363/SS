FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DATA_DIR=/app/data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY video_rag ./video_rag

ARG INSTALL_OCR=false
RUN if [ "$INSTALL_OCR" = "true" ]; then \
      pip install --no-cache-dir -e ".[api,transcribe,embed,answer,vectorstore,ocr]"; \
    else \
      pip install --no-cache-dir -e ".[api,transcribe,embed,answer,vectorstore]"; \
    fi

RUN mkdir -p /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health').read()"

CMD ["python", "-m", "video_rag.api.app"]
