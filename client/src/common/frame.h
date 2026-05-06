#pragma once

#include <chrono>
#include <cstdint>
#include <opencv2/core.hpp>

namespace vlm {

// Capture mode controls the rate-limiting policy of the pipeline.
enum class CaptureMode {
  kIdle,    // SCD + AFF gated (event-driven)
  kTyping,  // 1 FPS forced capture while user types
  kQuery    // Locked 1 FPS until query response arrives
};

inline const char* ToString(CaptureMode m) {
  switch (m) {
    case CaptureMode::kIdle:   return "idle";
    case CaptureMode::kTyping: return "typing";
    case CaptureMode::kQuery:  return "query";
  }
  return "unknown";
}

struct Frame {
  cv::Mat image;       // BGR
  uint64_t seq = 0;
  std::chrono::steady_clock::time_point ts{};
};

}  // namespace vlm
