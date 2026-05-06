#include "aff/adaptive_frame_filter.h"

#include <opencv2/imgproc.hpp>

namespace vlm {

AdaptiveFrameFilter::AdaptiveFrameFilter(Config cfg) : cfg_(cfg) {}

void AdaptiveFrameFilter::Reset() {
  prev_gray_small_.release();
  last_emit_ = {};
}

double AdaptiveFrameFilter::Sharpness(const cv::Mat& bgr) {
  cv::Mat gray, lap;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  cv::Laplacian(gray, lap, CV_64F);
  cv::Scalar mean, stddev;
  cv::meanStdDev(lap, mean, stddev);
  return stddev[0] * stddev[0];
}

double AdaptiveFrameFilter::DiffRatio(const cv::Mat& bgr) {
  cv::Mat gray, small;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  cv::resize(gray, small, cv::Size(cfg_.work_size, cfg_.work_size));

  if (prev_gray_small_.empty()) {
    prev_gray_small_ = small.clone();
    return 1.0;  // force "different" the first time
  }

  cv::Mat diff;
  cv::absdiff(prev_gray_small_, small, diff);
  cv::threshold(diff, diff, cfg_.diff_threshold, 255, cv::THRESH_BINARY);
  const double ratio = static_cast<double>(cv::countNonZero(diff)) /
                       static_cast<double>(diff.total());
  prev_gray_small_ = small;
  return ratio;
}

AdaptiveFrameFilter::Decision AdaptiveFrameFilter::Evaluate(const cv::Mat& bgr) {
  using clock = std::chrono::steady_clock;
  const auto now = clock::now();

  if (cfg_.max_fps > 0.0 && last_emit_.time_since_epoch().count() != 0) {
    const auto min_interval =
        std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(1.0 / cfg_.max_fps));
    if (now - last_emit_ < min_interval) {
      return Decision::kDropRate;
    }
  }

  if (DiffRatio(bgr) < cfg_.min_diff_ratio) {
    return Decision::kDropSimilar;
  }
  if (Sharpness(bgr) < cfg_.sharpness_floor) {
    return Decision::kDropBlurry;
  }

  last_emit_ = now;
  return Decision::kAccept;
}

AdaptiveFrameFilter::Decision AdaptiveFrameFilter::EvaluateForced(const cv::Mat& bgr) {
  // In forced 1 FPS mode we still update the diff baseline so a later
  // switch back to idle mode does not see a stale reference frame.
  (void)DiffRatio(bgr);
  last_emit_ = std::chrono::steady_clock::now();
  return Decision::kAccept;
}

}  // namespace vlm
