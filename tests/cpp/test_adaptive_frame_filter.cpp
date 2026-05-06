// Tests for vlm::AdaptiveFrameFilter.
//
// AFF combines three gates: rate limiting, motion delta, sharpness. We
// drive each gate independently with synthetic frames so failures point
// at one specific decision branch.

#include "aff/adaptive_frame_filter.h"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <thread>

namespace vlm {
namespace {

constexpr int kSize = 320;

cv::Mat MakeSolid(int value) {
  return cv::Mat(kSize, kSize, CV_8UC3,
                 cv::Scalar(value, value, value));
}

// A "rich" image with sharp edges so cv::Laplacian variance is high
// (well above the default sharpness floor).
cv::Mat MakeCheckerboard(int seed = 0) {
  cv::Mat m(kSize, kSize, CV_8UC3, cv::Scalar(0, 0, 0));
  const int square = 16;
  for (int y = 0; y < m.rows; ++y) {
    for (int x = 0; x < m.cols; ++x) {
      const bool on = ((x / square) + (y / square) + seed) & 1;
      if (on) {
        m.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
      }
    }
  }
  return m;
}

// Heavily blurred checkerboard: motion is high (vs. uniform baseline)
// but Laplacian variance falls below the default floor.
cv::Mat MakeBlurred(int seed = 0) {
  cv::Mat src = MakeCheckerboard(seed);
  cv::Mat blurred;
  cv::GaussianBlur(src, blurred, cv::Size(31, 31), 12.0);
  return blurred;
}

class AdaptiveFrameFilterTest : public ::testing::Test {};

TEST_F(AdaptiveFrameFilterTest, FirstFrameIsAccepted) {
  AdaptiveFrameFilter::Config cfg;
  cfg.max_fps = 0.0;  // disable rate cap for the first-frame check
  AdaptiveFrameFilter aff(cfg);

  EXPECT_EQ(aff.Evaluate(MakeCheckerboard()),
            AdaptiveFrameFilter::Decision::kAccept);
}

TEST_F(AdaptiveFrameFilterTest, IdenticalFollowupDropsAsSimilar) {
  AdaptiveFrameFilter::Config cfg;
  cfg.max_fps = 0.0;
  AdaptiveFrameFilter aff(cfg);

  const auto frame = MakeCheckerboard();
  ASSERT_EQ(aff.Evaluate(frame), AdaptiveFrameFilter::Decision::kAccept);
  // Same pixel data -> motion ratio is 0.
  EXPECT_EQ(aff.Evaluate(frame),
            AdaptiveFrameFilter::Decision::kDropSimilar);
}

TEST_F(AdaptiveFrameFilterTest, RateCapDropsFastFollowup) {
  AdaptiveFrameFilter::Config cfg;
  cfg.max_fps = 1.0;  // 1 FPS ceiling
  AdaptiveFrameFilter aff(cfg);

  ASSERT_EQ(aff.Evaluate(MakeCheckerboard(0)),
            AdaptiveFrameFilter::Decision::kAccept);
  // Same instant: even a brand-new scene should be rate-limited.
  EXPECT_EQ(aff.Evaluate(MakeCheckerboard(1)),
            AdaptiveFrameFilter::Decision::kDropRate);
}

TEST_F(AdaptiveFrameFilterTest, BlurryFrameIsRejected) {
  AdaptiveFrameFilter::Config cfg;
  cfg.max_fps = 0.0;
  cfg.sharpness_floor = 25.0;
  AdaptiveFrameFilter aff(cfg);

  // Seed the diff baseline (solid frames are themselves blurry, so we
  // ignore the decision here — we only need prev_gray_small_ populated).
  aff.Evaluate(MakeSolid(0));
  // A uniform 200-grey frame: huge diff vs the black baseline (so the
  // similarity gate passes) but Laplacian variance is ~0 → blurry.
  EXPECT_EQ(aff.Evaluate(MakeSolid(200)),
            AdaptiveFrameFilter::Decision::kDropBlurry);
}

TEST_F(AdaptiveFrameFilterTest, AcceptsAfterRateWindowElapses) {
  AdaptiveFrameFilter::Config cfg;
  cfg.max_fps = 30.0;  // ~33 ms minimum interval
  AdaptiveFrameFilter aff(cfg);

  ASSERT_EQ(aff.Evaluate(MakeCheckerboard(0)),
            AdaptiveFrameFilter::Decision::kAccept);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(aff.Evaluate(MakeCheckerboard(1)),
            AdaptiveFrameFilter::Decision::kAccept);
}

TEST_F(AdaptiveFrameFilterTest, EvaluateForcedAlwaysAccepts) {
  AdaptiveFrameFilter::Config cfg;
  cfg.max_fps = 1.0;
  AdaptiveFrameFilter aff(cfg);

  const auto frame = MakeCheckerboard(0);
  EXPECT_EQ(aff.EvaluateForced(frame),
            AdaptiveFrameFilter::Decision::kAccept);
  // Even an immediate identical follow-up is accepted in forced mode.
  EXPECT_EQ(aff.EvaluateForced(frame),
            AdaptiveFrameFilter::Decision::kAccept);
}

TEST_F(AdaptiveFrameFilterTest, ForcedUpdatesBaselineForLaterIdleEval) {
  // A forced evaluation must update the diff baseline, otherwise
  // returning to idle mode after typing would see a stale comparison
  // frame and emit a spurious "scene change".
  AdaptiveFrameFilter::Config cfg;
  cfg.max_fps = 0.0;
  AdaptiveFrameFilter aff(cfg);

  const auto a = MakeCheckerboard(0);
  const auto b = MakeCheckerboard(1);
  ASSERT_EQ(aff.Evaluate(a), AdaptiveFrameFilter::Decision::kAccept);
  // Forced replaces the baseline with `b`.
  aff.EvaluateForced(b);
  // Now feeding `b` again in idle mode must look like a near-duplicate.
  EXPECT_EQ(aff.Evaluate(b),
            AdaptiveFrameFilter::Decision::kDropSimilar);
}

TEST_F(AdaptiveFrameFilterTest, ResetClearsHistory) {
  AdaptiveFrameFilter::Config cfg;
  cfg.max_fps = 0.0;
  AdaptiveFrameFilter aff(cfg);

  const auto frame = MakeCheckerboard();
  ASSERT_EQ(aff.Evaluate(frame), AdaptiveFrameFilter::Decision::kAccept);
  ASSERT_EQ(aff.Evaluate(frame),
            AdaptiveFrameFilter::Decision::kDropSimilar);

  aff.Reset();
  // After reset, the same frame is treated as the "first" again.
  EXPECT_EQ(aff.Evaluate(frame),
            AdaptiveFrameFilter::Decision::kAccept);
}

}  // namespace
}  // namespace vlm
