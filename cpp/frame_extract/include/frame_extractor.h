// Frame extractor: open a video, seek to requested timestamps, write JPEGs.
//
// This header is the only public surface of the C++ extractor. The Python
// orchestrator (video_rag/index/sample_frames.py) does NOT include or link
// against this; it only invokes the resulting CLI via subprocess.

#pragma once

#include <string>
#include <vector>

namespace raggers::frame_extract {

struct Config {
  std::string video_path;
  std::string out_dir;
  std::vector<double> timestamps_seconds;
  int jpeg_quality = 85;
};

struct FrameRecord {
  double timestamp;
  std::string frame_path;
  int width;
  int height;
};

// Decode the requested timestamps from `config.video_path` and write JPEGs
// into `config.out_dir`. Returns one FrameRecord per successfully written
// frame. Throws std::runtime_error on any failure (cannot open video, decode
// failure, write failure, etc.).
std::vector<FrameRecord> ExtractFrames(const Config& config);

}  // namespace raggers::frame_extract
