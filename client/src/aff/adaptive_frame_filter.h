#pragma once

#include <chrono>
#include <deque>
#include <opencv2/core.hpp>

namespace vlm {

// Adaptive Frame Filtering: even after a scene change is signalled by
// SCD, we may want to throttle / pick the best frame from a small
// window. This filter
//   * drops near-duplicate frames using a frame-difference heuristic,
//   * caps the maximum FPS forwarded to the encoder,
//   * picks the sharpest frame in a small temporal window when several
//     scene changes happen in rapid succession (de-flicker).
class AdaptiveFrameFilter {
 public:
  struct Config {
    double max_fps = 5.0;            // ceiling in idle / SCD-driven mode
    double min_diff_ratio = 0.015;   // minimum motion ratio to accept
    int    diff_threshold = 25;      // |Δpixel| threshold (0..255)
    int    work_size = 320;          // downscaled comparison size
    double sharpness_floor = 25.0;   // variance-of-Laplacian below which we drop
    Config() {}
  };

  enum class Decision { kAccept, kDropSimilar, kDropRate, kDropBlurry };

  explicit AdaptiveFrameFilter(Config cfg = {});

  // Returns the pipeline decision for the given frame.
  Decision Evaluate(const cv::Mat& bgr);

  // Bypass throttling; used when the pipeline forces 1 FPS during
  // typing / query mode.
  Decision EvaluateForced(const cv::Mat& bgr);

  void Reset();

 private:
  Config cfg_;
  cv::Mat prev_gray_small_;
  std::chrono::steady_clock::time_point last_emit_{};

  static double Sharpness(const cv::Mat& bgr);
  double DiffRatio(const cv::Mat& bgr);
};

}  // namespace vlm
