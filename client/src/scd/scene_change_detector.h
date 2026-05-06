#pragma once

#include <opencv2/core.hpp>

namespace vlm {

// Histogram + perceptual-hash based scene change detector. Cheap to run
// per frame and robust against minor cursor movement.
class SceneChangeDetector {
 public:
  struct Config {
    // Bhattacharyya distance threshold above which the histogram of the
    // current frame is considered a "scene change". 0..1.
    double hist_threshold = 0.35;
    // Hamming distance (in bits) over a 64-bit pHash. Triggers a change
    // when distance >= phash_threshold.
    int phash_threshold = 12;
    // Downscale used for histogram + pHash to make the detector cheap.
    int work_size = 256;
    Config() {}
  };

  explicit SceneChangeDetector(Config cfg = {});

  // Returns true the first time it is called, and afterwards whenever
  // the new frame differs from the previously accepted frame by more
  // than the configured thresholds.
  bool IsSceneChange(const cv::Mat& bgr);

  void Reset();

 private:
  Config cfg_;
  bool has_prev_ = false;
  cv::Mat prev_hist_;
  uint64_t prev_phash_ = 0;

  static cv::Mat ComputeHistogram(const cv::Mat& bgr_small);
  static uint64_t ComputePHash(const cv::Mat& bgr_small);
};

}  // namespace vlm
