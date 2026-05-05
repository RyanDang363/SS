# raggers_frame_extract

Stage 5's compute kernel: open a video, seek to a list of timestamps, write
JPEG frames. Built in C++17 against OpenCV. Driven by Python via the
`video_rag.index.sample_frames` orchestrator (no pybind11 / FFI / linking).

## Build

Prereqs:

- A C++17 toolchain (clang or gcc).
- CMake >= 3.18.
- OpenCV development headers and libs (must include `imgcodecs`, `imgproc`,
  `videoio`, `core`).

System install of OpenCV:

- macOS: `brew install opencv`
- Ubuntu/Debian: `sudo apt install libopencv-dev`
- Fedora: `sudo dnf install opencv-devel`

Configure and build:

```bash
cmake -S cpp/frame_extract -B cpp/frame_extract/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/frame_extract/build -j
```

The resulting binary is at `cpp/frame_extract/build/raggers_frame_extract`.
The Python orchestrator looks for it there by default; override with the
`--binary-path` CLI flag or the `RAGGERS_FRAME_EXTRACT_BIN` environment
variable.

## Wire protocol

```
raggers_frame_extract \
  --video <path> \
  --out-dir <path> \
  --timestamps t1,t2,...   (seconds) \
  [--quality 85]
```

- `stdout`: one JSON object per produced frame, newline-delimited.
  Example: `{"timestamp":15.0,"frame_path":"/abs/data/frames/.../frame_000015.jpg","width":1920,"height":1080}`
- `stderr`: human-readable error messages prefixed with `FAIL`.
- exit code: `0` on success, non-zero on any failure.

The binary is intentionally agnostic to the wider artifact contract: it does
not know about `video_id`, manifests, or the JSONL final layout. Python adds
`video_id` and `sampling_method` and writes `frame_manifest.jsonl`.

## Layout

```
cpp/frame_extract/
  CMakeLists.txt
  include/frame_extractor.h     # public API: Config, FrameRecord, ExtractFrames
  src/frame_extractor.cpp       # OpenCV decode + JPEG write
  src/main.cpp                  # CLI: arg parsing, JSONL emission, error policy
  README.md
  .gitignore
```

## V0 limitations

- Integer-second timestamps only (filename pattern is `frame_NNNNNN.jpg`).
- Sampling is approximate: `cv::CAP_PROP_POS_MSEC` may snap to the nearest
  preceding keyframe. Acceptable for indexing-time sampling at multi-second
  intervals.
- No hardware decode, no batching, no thumbnails. Optimization is deferred.
