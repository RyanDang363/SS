# raggers_frame_extract

> **Optional.** Stage 5 (`video_rag.index.sample_frames`) now uses
> `opencv-python-headless` by default — no build step required. This C++
> extractor is kept as an optional high-throughput alternative that shares the
> same output contract. Use it if you need maximum decode throughput on large
> batches.

Stage 5's compute kernel: open a video, seek to a list of timestamps, write
JPEG frames. Built in C++17 against OpenCV.

For full project setup (Python venv, tests, Stage 5 CLI), see the [repo README](../../README.md).

## Prerequisites

- C++17 toolchain (clang or gcc)
- CMake ≥ 3.18
- OpenCV with `core`, `imgcodecs`, `imgproc`, `videoio`

### macOS (Homebrew)

```bash
# Required once before compiling on a new Mac:
sudo xcodebuild -license accept

brew install cmake opencv
```

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install cmake build-essential libopencv-dev
```

### Fedora

```bash
sudo dnf install cmake gcc-c++ opencv-devel
```

### Windows (PowerShell + Visual Studio Build Tools)

Install dependencies (example via winget):

```powershell
winget install Kitware.CMake
winget install OpenCV.OpenCV
```

Build tools requirement: Visual Studio Build Tools (C++ workload) or Visual Studio Community with Desktop C++.

## Build

From the **repository root**:

```bash
# macOS: help CMake find OpenCV (often required with Homebrew)
export OpenCV_DIR="$(brew --prefix opencv)/lib/cmake/opencv4"

cmake -S cpp/frame_extract -B cpp/frame_extract/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR="${OpenCV_DIR:-}"

cmake --build cpp/frame_extract/build -j
```

On Linux, omit `-DOpenCV_DIR` if `find_package(OpenCV)` succeeds without it.

Windows configure/build (multi-config generator):

```powershell
cmake -S cpp/frame_extract -B cpp/frame_extract/build -A x64
cmake --build cpp/frame_extract/build --config Release -j
```

Windows binary output is typically:

```text
cpp/frame_extract/build/Release/raggers_frame_extract.exe
```

If configure fails with "Could not find OpenCV", locate the config file:

```bash
# macOS example
find "$(brew --prefix opencv)" -name OpenCVConfig.cmake
```

Set `-DOpenCV_DIR` to the **directory** that contains `OpenCVConfig.cmake`.

Binary output:

```text
cpp/frame_extract/build/raggers_frame_extract
```

Smoke test:

```bash
cpp/frame_extract/build/raggers_frame_extract --help
```

## How to use the binary

Since Stage 5 uses Python OpenCV by default, you must call the binary directly
(or wire it yourself). The binary accepts the same flags it always has:

```bash
cpp/frame_extract/build/raggers_frame_extract \
  --video path/to/video.mp4 \
  --out-dir path/to/frames/ \
  --timestamps 0.000,5.000,10.000 \
  --quality 85
```

Python parses its JSONL stdout and writes `frame_manifest.jsonl` (see Wire
protocol below). You would need to reintroduce the subprocess call in
`video_rag/index/sample_frames.py` if you want to route through this binary
instead of the default Python path.

## Wire protocol

```text
raggers_frame_extract \
  --video <path> \
  --out-dir <path> \
  --timestamps t1,t2,...   (seconds) \
  [--quality 85]
```

| Stream | Content |
|--------|---------|
| stdout | One JSON object per frame, newline-delimited |
| stderr | Errors prefixed with `FAIL` |
| exit code | `0` success, non-zero on failure |

Example stdout line:

```json
{"timestamp":15.0,"frame_path":"/abs/path/frame_000015.jpg","width":1920,"height":1080}
```

Python adds `video_id`, `sampling_method`, and writes `frame_manifest.jsonl`.

## Source layout

```text
cpp/frame_extract/
  CMakeLists.txt
  include/frame_extractor.h   # Config, FrameRecord, ExtractFrames
  src/frame_extractor.cpp     # OpenCV decode + JPEG write
  src/main.cpp                # CLI + JSONL emission
  README.md
  .gitignore                  # ignores build/
```

## Tests

Unit tests monkeypatch `_extract_frames` and do not require this binary or
any native dependencies beyond `opencv-python-headless`.

Integration test (exercises real OpenCV decode + JPEG writes, needs `ffmpeg`
to synthesize a tiny fixture video):

```bash
brew install ffmpeg   # macOS
pytest tests/test_sample_frames_integration.py -v
```

## V0 limitations

- Integer-second timestamps only (`frame_NNNNNN.jpg` filenames).
- Seeking via `CAP_PROP_POS_MSEC` may snap to the nearest keyframe.
- No thumbnails, hardware decode, or batching in this binary.
