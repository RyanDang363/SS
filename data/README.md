# `data/` artifact layout

Each subfolder owns one type of pipeline artifact. Folders are tracked via
`.gitkeep` so the layout is reproducible; actual artifact files should not be
committed unless they are intentionally small fixtures.

| Folder           | Contents                                                  |
| ---------------- | --------------------------------------------------------- |
| `videos/`        | Source video files (`.mp4`, etc.).                        |
| `audio/`         | Audio extracted from videos.                              |
| `frames/`        | Sampled frames (per video, per timestamp).                |
| `thumbnails/`    | Per-video preview images.                                 |
| `transcripts/`   | Transcripts (future module).                              |
| `ocr/`           | OCR text per frame (future module).                       |
| `captions/`      | VLM frame captions (future module).                       |
| `chunks/`        | Retrieval chunks (future module).                         |
| `embeddings/`    | Embedding vectors / sidecars (future module).             |
| `indexes/`       | Vector DB / index payloads (future module).               |
| `manifests/`     | `VideoManifest` and `MediaMetadata` JSON artifacts.       |
| `validation/`    | Validation reports and logs.                              |

Stage 0-lite only writes to `manifests/` and `validation/`. Other folders are
placeholders for downstream modules.
