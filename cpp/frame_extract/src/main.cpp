// Thin CLI wrapper around raggers::frame_extract::ExtractFrames.
//
// Wire protocol with Python (video_rag/index/sample_frames.py):
//   stdin:   unused
//   stdout:  one JSON object per produced frame, newline-delimited
//   stderr:  human-readable error messages prefixed with "FAIL"
//   exit:    0 on success, non-zero on any failure
//
// Flags:
//   --video <path>        path to the input video file
//   --out-dir <path>      directory to write JPEG frames into
//   --timestamps <list>   comma-separated seconds (e.g. "0,5,10")
//   --quality <int>       JPEG quality 1..100 (default 85)

#include "frame_extractor.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void Die(const std::string& message) {
  std::cerr << "FAIL  " << message << "\n";
  std::exit(1);
}

std::vector<double> ParseTimestamps(const std::string& spec) {
  std::vector<double> out;
  if (spec.empty()) {
    return out;
  }
  std::stringstream ss(spec);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    try {
      out.push_back(std::stod(token));
    } catch (const std::exception&) {
      throw std::runtime_error("invalid timestamp value: '" + token + "'");
    }
  }
  return out;
}

std::string EscapeJsonString(const std::string& input) {
  std::string out;
  out.reserve(input.size() + 2);
  for (const char c : input) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

void EmitJsonl(const raggers::frame_extract::FrameRecord& record) {
  std::ostringstream oss;
  oss << "{"
      << "\"timestamp\":" << record.timestamp << ","
      << "\"frame_path\":\"" << EscapeJsonString(record.frame_path) << "\","
      << "\"width\":" << record.width << ","
      << "\"height\":" << record.height
      << "}";
  std::cout << oss.str() << "\n";
}

const std::string& RequireArg(int argc, char** argv, int& i,
                              const std::string& flag) {
  if (i + 1 >= argc) {
    throw std::runtime_error("missing value for " + flag);
  }
  static thread_local std::string value;
  value = argv[++i];
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    raggers::frame_extract::Config config;
    bool have_timestamps = false;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--video") {
        config.video_path = RequireArg(argc, argv, i, arg);
      } else if (arg == "--out-dir") {
        config.out_dir = RequireArg(argc, argv, i, arg);
      } else if (arg == "--timestamps") {
        const std::string spec = RequireArg(argc, argv, i, arg);
        config.timestamps_seconds = ParseTimestamps(spec);
        have_timestamps = true;
      } else if (arg == "--quality") {
        const std::string val = RequireArg(argc, argv, i, arg);
        config.jpeg_quality = std::stoi(val);
      } else if (arg == "--help" || arg == "-h") {
        std::cout
            << "usage: raggers_frame_extract --video <path> --out-dir <path>"
               " --timestamps <s1,s2,...> [--quality <1-100>]\n";
        return 0;
      } else {
        throw std::runtime_error("unknown argument: " + arg);
      }
    }

    if (config.video_path.empty()) {
      throw std::runtime_error("--video is required");
    }
    if (config.out_dir.empty()) {
      throw std::runtime_error("--out-dir is required");
    }
    if (!have_timestamps) {
      throw std::runtime_error("--timestamps is required");
    }

    const auto records = raggers::frame_extract::ExtractFrames(config);
    for (const auto& rec : records) {
      EmitJsonl(rec);
    }
    return 0;
  } catch (const std::exception& e) {
    Die(e.what());
  } catch (...) {
    Die("unknown error");
  }
}
