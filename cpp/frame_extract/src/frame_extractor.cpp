#include "frame_extractor.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace raggers::frame_extract {

namespace {

// Filename pattern matches the issue example: `frame_NNNNNN.jpg` where
// NNNNNN is the timestamp rounded to whole seconds, zero-padded to 6 digits.
// V0 only supports integer-second intervals so collisions are impossible.
std::string FrameFilename(double timestamp_seconds) {
  const long long whole_seconds =
      static_cast<long long>(std::llround(timestamp_seconds));
  char buf[64];
  std::snprintf(buf, sizeof(buf), "frame_%06lld.jpg",
                static_cast<long long>(whole_seconds));
  return std::string(buf);
}

void EnsureOutDir(const std::string& out_dir) {
  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);
  if (ec) {
    throw std::runtime_error("failed to create out-dir '" + out_dir +
                             "': " + ec.message());
  }
}

}  // namespace

std::vector<FrameRecord> ExtractFrames(const Config& config) {
  if (config.video_path.empty()) {
    throw std::runtime_error("video_path is empty");
  }
  if (config.out_dir.empty()) {
    throw std::runtime_error("out_dir is empty");
  }
  if (config.jpeg_quality < 1 || config.jpeg_quality > 100) {
    throw std::runtime_error("jpeg_quality must be in [1, 100]");
  }

  EnsureOutDir(config.out_dir);

  cv::VideoCapture cap(config.video_path);
  if (!cap.isOpened()) {
    throw std::runtime_error("failed to open video '" + config.video_path + "'");
  }

  const std::vector<int> imwrite_params = {cv::IMWRITE_JPEG_QUALITY,
                                           config.jpeg_quality};

  std::vector<FrameRecord> records;
  records.reserve(config.timestamps_seconds.size());

  cv::Mat frame;
  for (const double ts : config.timestamps_seconds) {
    if (ts < 0.0) {
      throw std::runtime_error("negative timestamp is not allowed");
    }
    const double ms = ts * 1000.0;
    if (!cap.set(cv::CAP_PROP_POS_MSEC, ms)) {
      std::ostringstream oss;
      oss << "seek to " << ts << "s failed";
      throw std::runtime_error(oss.str());
    }
    if (!cap.read(frame) || frame.empty()) {
      std::ostringstream oss;
      oss << "decode at " << ts << "s returned an empty frame";
      throw std::runtime_error(oss.str());
    }

    const std::filesystem::path out_path =
        std::filesystem::path(config.out_dir) / FrameFilename(ts);
    if (!cv::imwrite(out_path.string(), frame, imwrite_params)) {
      throw std::runtime_error("failed to write JPEG to '" + out_path.string() +
                               "'");
    }

    FrameRecord rec;
    rec.timestamp = ts;
    rec.frame_path = out_path.string();
    rec.width = frame.cols;
    rec.height = frame.rows;
    records.push_back(std::move(rec));
  }

  return records;
}

}  // namespace raggers::frame_extract
