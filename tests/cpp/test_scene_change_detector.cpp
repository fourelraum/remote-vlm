// Tests for vlm::SceneChangeDetector.
//
// SCD combines an HSV histogram comparison (Bhattacharyya distance)
// with an 8x8 perceptual hash hamming distance. We construct images
// that are clearly different / clearly identical and verify the
// detector reacts as expected, plus exercise Reset() semantics.

#include "scd/scene_change_detector.h"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

namespace vlm {
namespace {

cv::Mat SolidColor(uint8_t b, uint8_t g, uint8_t r, int size = 256) {
  return cv::Mat(size, size, CV_8UC3, cv::Scalar(b, g, r));
}

// A structured image with deterministic stripes so two seeds produce
// visually distinct content (different histogram and pHash).
cv::Mat Stripes(int seed, int size = 256) {
  cv::Mat m(size, size, CV_8UC3, cv::Scalar(0, 0, 0));
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      const uint8_t v = static_cast<uint8_t>(((x + y * 3 + seed * 47) % 256));
      m.at<cv::Vec3b>(y, x) = cv::Vec3b(v, 255 - v, (v + 64) & 0xff);
    }
  }
  return m;
}

class SceneChangeDetectorTest : public ::testing::Test {};

TEST_F(SceneChangeDetectorTest, FirstFrameIsAlwaysASceneChange) {
  SceneChangeDetector scd;
  EXPECT_TRUE(scd.IsSceneChange(SolidColor(10, 20, 30)));
}

TEST_F(SceneChangeDetectorTest, IdenticalFollowupIsNotSceneChange) {
  SceneChangeDetector scd;
  const auto frame = SolidColor(10, 20, 30);
  ASSERT_TRUE(scd.IsSceneChange(frame));
  EXPECT_FALSE(scd.IsSceneChange(frame));
}

TEST_F(SceneChangeDetectorTest, BigColorJumpIsSceneChange) {
  SceneChangeDetector scd;
  // Pure red vs pure blue → histograms collapse to opposite ends of the
  // hue axis (H,S only, so we have to vary hue, not just luminance).
  ASSERT_TRUE(scd.IsSceneChange(SolidColor(/*B=*/0, /*G=*/0, /*R=*/255)));
  EXPECT_TRUE(scd.IsSceneChange(SolidColor(/*B=*/255, /*G=*/0, /*R=*/0)));
}

TEST_F(SceneChangeDetectorTest, StructuredFrameSwapIsSceneChange) {
  SceneChangeDetector scd;
  ASSERT_TRUE(scd.IsSceneChange(Stripes(0)));
  // Same generator, different seed -> visibly different content.
  EXPECT_TRUE(scd.IsSceneChange(Stripes(7)));
}

TEST_F(SceneChangeDetectorTest, ResetForcesNextFrameToBeSceneChange) {
  SceneChangeDetector scd;
  const auto frame = SolidColor(50, 100, 150);
  ASSERT_TRUE(scd.IsSceneChange(frame));
  ASSERT_FALSE(scd.IsSceneChange(frame));
  scd.Reset();
  EXPECT_TRUE(scd.IsSceneChange(frame));
}

TEST_F(SceneChangeDetectorTest, BaselineUpdatesOnlyOnSceneChange) {
  SceneChangeDetector scd;
  const auto red  = SolidColor(/*B=*/0, /*G=*/0, /*R=*/255);
  const auto blue = SolidColor(/*B=*/255, /*G=*/0, /*R=*/0);

  ASSERT_TRUE(scd.IsSceneChange(red));   // first frame -> baseline=red
  // Same color repeated must not be flagged.
  EXPECT_FALSE(scd.IsSceneChange(red));
  EXPECT_FALSE(scd.IsSceneChange(red));
  // Hue-axis jump still trips the detector.
  EXPECT_TRUE(scd.IsSceneChange(blue));
}

}  // namespace
}  // namespace vlm
