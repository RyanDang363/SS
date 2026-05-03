## Indexing-stage architecture

```text
Raw video
↓
1. Register video
↓
2. Probe media metadata
↓
3. Extract audio
↓
4. Transcribe audio
↓
5. Sample frames / thumbnails
↓
6. OCR frames
↓
7. Generate VLM captions
↓
8. Build chunks
↓
9. Build searchable text
↓
10. Embed chunks
↓
11. Store vectors + metadata
↓
12. Validate index
```

## Recommended atomic tasks

| Stage                            | Atomic task                                              | Input               | Output                              | Best language                 |
| -------------------------------- | -------------------------------------------------------- | ------------------- | ----------------------------------- | ----------------------------- |
| **0. Project schema**            | Define shared JSON schemas                               | none                | `schemas.py`, sample JSON           | Python                        |
| **1. Video registration**        | Assign stable `video_id`, copy video into `data/videos/` | raw `.mp4`          | `video_manifest.json`               | Python                        |
| **2. Media probe**               | Get duration, FPS, resolution, audio streams             | video path          | `media_metadata.json`               | Python calling FFmpeg/ffprobe |
| **3. Audio extraction**          | Extract `.wav` for transcription                         | video path          | `audio.wav`                         | FFmpeg CLI                    |
| **4. Transcription**             | Speech-to-text with timestamps                           | `audio.wav`         | `transcript_segments.json`          | Python                        |
| **5. Frame sampling**            | Sample frames every N seconds or per chunk               | video path          | JPEG frames + `frame_manifest.json` | **C++ best fit**              |
| **6. Thumbnail generation**      | Resize frames for UI evidence cards                      | sampled frames      | thumbnails                          | **C++ or Python**             |
| **7. OCR extraction**            | Read visible slide/board text                            | sampled frames      | `ocr_segments.json`                 | Python                        |
| **8. VLM captions**              | Caption sampled frames/chunks                            | sampled frames      | `vlm_captions.json`                 | Python                        |
| **9. Chunking**                  | Combine transcript/OCR/captions into 15s/30s chunks      | all artifacts       | `chunks.jsonl`                      | Python                        |
| **10. Search text construction** | Build `combined_text` per chunk                          | `chunks.jsonl`      | enriched `chunks.jsonl`             | Python                        |
| **11. Embedding**                | Embed `combined_text`                                    | enriched chunks     | `embeddings.npy` / vector records   | Python                        |
| **12. Vector storage**           | Store embeddings + metadata                              | chunks + embeddings | Chroma/FAISS index                  | Python                        |
| **13. Index validation**         | Check timestamps, empty chunks, missing metadata         | full index          | validation report                   | Python                        |

OpenCV is a good C++ fit for reading video files and frame sequences through `VideoCapture`, while FFmpeg is the right tool for audio/video extraction and format handling. Chroma is a good V0 vector store because it supports storing metadata and filtering query results by metadata fields. ([OpenCV Documentation][1])

---

# Stage-by-stage breakdown

## 0. Define shared data contracts first

Before coding extraction, define the JSON objects all modules produce and consume.

### `video_manifest.json`

```json
{
  "video_id": "lecture_001",
  "title": "Bayes Theorem Lecture",
  "source_path": "data/videos/lecture_001.mp4",
  "created_at": "2026-05-03T12:00:00Z"
}
```

### `chunk.jsonl` record

```json
{
  "chunk_id": "lecture_001_chunk_0007",
  "video_id": "lecture_001",
  "chunk_index": 7,
  "start_time": 210.0,
  "end_time": 240.0,
  "transcript_text": "",
  "ocr_text": "",
  "vlm_caption": "",
  "combined_text": "",
  "keyframe_paths": [],
  "thumbnail_paths": [],
  "metadata": {
    "chunk_seconds": 30,
    "has_transcript": true,
    "has_ocr": false,
    "has_vlm_caption": true
  }
}
```

**Definition of done:** every later module reads/writes these schemas without needing to know the whole pipeline.

---

## 1. Video registration

Goal: take a raw video and make it a stable project asset.

Atomic tasks:

```text
- Accept local video path
- Generate video_id
- Copy or symlink file into data/videos/
- Create video_manifest.json
```

Suggested command:

```bash
python -m raggers.index.register_video \
  --video ~/Downloads/lecture.mp4 \
  --title "Bayes Theorem Lecture"
```

Output:

```text
data/manifests/lecture_001/video_manifest.json
```

C++ is not needed here.

---

## 2. Media probe

Goal: inspect the video before doing expensive work.

Atomic tasks:

```text
- Get duration
- Get FPS
- Get width/height
- Check whether audio exists
- Store metadata
```

Suggested output:

```json
{
  "video_id": "lecture_001",
  "duration_seconds": 1840.5,
  "fps": 30.0,
  "width": 1920,
  "height": 1080,
  "has_audio": true
}
```

Use Python calling `ffprobe`. Keep this simple.

---

## 3. Audio extraction

Goal: produce a clean audio file for transcription.

Atomic tasks:

```text
- Take video path
- Extract mono WAV
- Save to data/audio/
```

Suggested command:

```bash
python -m raggers.index.extract_audio \
  --video-id lecture_001
```

Internally:

```bash
ffmpeg -i data/videos/lecture_001.mp4 -vn -ac 1 -ar 16000 data/audio/lecture_001.wav
```

C++ is not worth it here. FFmpeg CLI is perfect.

---

## 4. Transcription

Goal: turn audio into timestamped text.

Atomic tasks:

```text
- Send audio to transcription model
- Normalize transcript segments
- Store start/end timestamps
```

Output:

```json
[
  {
    "start_time": 211.2,
    "end_time": 225.8,
    "text": "Bayes theorem tells us how to update a probability after seeing evidence."
  }
]
```

For V0, use an API or Python Whisper. Later, `whisper.cpp` can be a strong local/offline C++-adjacent module.

---

## 5. Frame sampling

Goal: sample representative frames from the video.

This is the **best first C++ module**.

Atomic tasks:

```text
- Open video
- Sample frame every N seconds
- Optionally sample midpoint of each future chunk
- Save JPEGs
- Emit frame_manifest.json
```

C++ command:

```bash
raggers_frame_extract \
  --video data/videos/lecture_001.mp4 \
  --video-id lecture_001 \
  --interval-seconds 5 \
  --out-dir data/frames/lecture_001 \
  --manifest data/frames/lecture_001/frame_manifest.json
```

Output:

```json
[
  {
    "video_id": "lecture_001",
    "timestamp": 210.0,
    "frame_path": "data/frames/lecture_001/frame_000210.jpg",
    "width": 1920,
    "height": 1080
  }
]
```

Why C++ here: this is repetitive frame I/O, decode, resize, write. It is cleanly separable from the rest of the RAG code.

---

## 6. Thumbnail generation

Goal: create small images for UI evidence cards.

Atomic tasks:

```text
- Read sampled frames
- Resize to thumbnail dimensions
- Save to data/thumbnails/
- Add thumbnail path to frame manifest
```

This can live in the same C++ binary as frame extraction.

Suggested output:

```json
{
  "frame_path": "data/frames/lecture_001/frame_000210.jpg",
  "thumbnail_path": "data/thumbnails/lecture_001/thumb_000210.jpg"
}
```

---

## 7. OCR extraction

Goal: read visible text from slides, whiteboards, screens, diagrams.

Atomic tasks:

```text
- Load sampled frames
- Run OCR
- Normalize text
- Attach OCR text to frame timestamp
```

Output:

```json
[
  {
    "video_id": "lecture_001",
    "timestamp": 210.0,
    "frame_path": "data/frames/lecture_001/frame_000210.jpg",
    "ocr_text": "Bayes Theorem: P(A|B) = P(B|A)P(A)/P(B)"
  }
]
```

Keep this in Python. OCR libraries and image preprocessing are easier to iterate there.

---

## 8. VLM captioning

Goal: create visual descriptions that transcript/OCR may miss.

Atomic tasks:

```text
- Group frames by chunk or timestamp
- Send 1–3 frames to VLM
- Ask for concise factual caption
- Store caption with timestamp range
```

Output:

```json
[
  {
    "video_id": "lecture_001",
    "start_time": 210.0,
    "end_time": 240.0,
    "frame_paths": [
      "data/frames/lecture_001/frame_000210.jpg",
      "data/frames/lecture_001/frame_000225.jpg"
    ],
    "vlm_caption": "The instructor stands beside a slide showing Bayes theorem and a probability formula."
  }
]
```

Suggested caption prompt:

```text
Describe only what is visually present in these lecture frames.
Mention visible equations, diagrams, slide titles, board writing, people, and actions.
Do not infer beyond the images.
Return 1–3 concise sentences.
```

Keep this in Python. The bottleneck is the VLM call, not local compute.

---

## 9. Chunking

Goal: create the actual retrieval units.

Start with fixed chunks:

```text
15 seconds
30 seconds
```

Atomic tasks:

```text
- Create time windows
- Pull transcript segments overlapping each window
- Pull OCR frames inside each window
- Pull VLM captions for frames/chunk
- Create chunk records
```

Example:

```json
{
  "chunk_id": "lecture_001_chunk_0007",
  "start_time": 210.0,
  "end_time": 240.0,
  "transcript_text": "Bayes theorem tells us how to update probability...",
  "ocr_text": "P(A|B) = P(B|A)P(A)/P(B)",
  "vlm_caption": "The instructor stands beside a slide showing Bayes theorem.",
  "keyframe_paths": ["data/frames/lecture_001/frame_000225.jpg"]
}
```

Make chunking independent:

```bash
python -m raggers.index.build_chunks \
  --video-id lecture_001 \
  --chunk-seconds 30
```

Later experiments:

```text
fixed 15s vs fixed 30s
overlapping vs non-overlapping
scene-based chunks
slide-change chunks
```

---

## 10. Search text construction

Goal: decide what text gets embedded.

Atomic tasks:

```text
- Take each chunk
- Build combined_text
- Keep modality sections clearly labeled
```

Example:

```text
Transcript:
Bayes theorem tells us how to update probability after seeing evidence.

On-screen text:
P(A|B) = P(B|A)P(A)/P(B)

Visual caption:
The instructor stands beside a slide showing Bayes theorem and a probability formula.
```

This stage matters because you want to test:

```text
transcript only
transcript + OCR
transcript + VLM
transcript + OCR + VLM
```

So build multiple `combined_text` variants.

Suggested output fields:

```json
{
  "combined_text_transcript_only": "...",
  "combined_text_transcript_ocr": "...",
  "combined_text_transcript_vlm": "...",
  "combined_text_all": "..."
}
```

---

## 11. Embedding

Goal: turn chunks into vectors.

Atomic tasks:

```text
- Select text variant
- Batch embed chunks
- Store embedding vectors
- Store embedding model name
```

Output:

```json
{
  "chunk_id": "lecture_001_chunk_0007",
  "embedding_model": "text-embedding-3-small",
  "embedding_variant": "transcript_ocr_vlm",
  "embedding": [0.012, -0.034, "..."]
}
```

Keep this in Python. Embedding calls are API/model-bound.

---

## 12. Vector storage

Goal: write records into Chroma or FAISS.

Atomic tasks:

```text
- Create collection/index
- Insert chunk vectors
- Attach metadata
- Persist index
```

Metadata to store:

```json
{
  "chunk_id": "lecture_001_chunk_0007",
  "video_id": "lecture_001",
  "start_time": 210.0,
  "end_time": 240.0,
  "has_transcript": true,
  "has_ocr": true,
  "has_vlm_caption": true,
  "embedding_variant": "transcript_ocr_vlm"
}
```

For V0: Chroma.
For performance later: FAISS, possibly C++.

---

## 13. Index validation

Goal: make sure indexing did not silently fail.

Atomic checks:

```text
- Every chunk has video_id/start/end
- No negative timestamps
- No overlapping bugs unless intentionally overlapping
- Chunks cover full video
- Transcript text is attached to correct time ranges
- Frame timestamps fall inside chunk ranges
- VLM captions are not empty for chunks with frames
- Vector count equals chunk count
```

Output:

```json
{
  "video_id": "lecture_001",
  "num_chunks": 62,
  "num_chunks_with_transcript": 61,
  "num_chunks_with_ocr": 18,
  "num_chunks_with_vlm_caption": 62,
  "num_vectors": 62,
  "errors": [],
  "warnings": [
    "chunk_0042 has no transcript text"
  ]
}
```

This is boring but incredibly useful.

---

# How to organize the repo for indexing only

```text
raggers/
  cpp/
    frame_extract/
      CMakeLists.txt
      main.cpp
      frame_extractor.cpp
      frame_extractor.h

  raggers/
    schemas.py

    index/
      register_video.py
      probe_media.py
      extract_audio.py
      transcribe.py
      run_ocr.py
      run_vlm_caption.py
      build_chunks.py
      build_search_text.py
      embed_chunks.py
      store_vectors.py
      validate_index.py

    utils/
      time.py
      jsonl.py
      ffmpeg.py
      paths.py

  data/
    videos/
    audio/
    frames/
    thumbnails/
    transcripts/
    ocr/
    captions/
    chunks/
    indexes/
    manifests/
```

---

# Suggested build order

Do not build everything at once. Build in this order:

```text
1. Define schemas
2. Register video
3. Probe media
4. Extract audio
5. Transcribe audio
6. C++ frame sampler
7. Build fixed 30s chunks with transcript only
8. Embed transcript-only chunks
9. Store in Chroma
10. Validate index
11. Add OCR
12. Add VLM captions
13. Compare indexing variants
```

Your first working milestone should be:

```text
local video → transcript → 30s chunks → embeddings → Chroma index
```

Your second milestone:

```text
local video → frames → VLM captions → transcript + VLM chunks → embeddings → Chroma index
```

Your third milestone:

```text
same video indexed four ways:
A. transcript only
B. transcript + OCR
C. transcript + VLM captions
D. transcript + OCR + VLM captions
```

That sets you up perfectly for retrieval testing later.

[1]: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html?utm_source=chatgpt.com "cv::VideoCapture Class Reference"
