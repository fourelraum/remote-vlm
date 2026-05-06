#include "scd/scene_change_detector.h"

#include <bitset>
#include <opencv2/imgproc.hpp>

namespace vlm {

SceneChangeDetector::SceneChangeDetector(Config cfg) : cfg_(cfg) {}

void SceneChangeDetector::Reset() {
  has_prev_ = false;
  prev_hist_.release();
  prev_phash_ = 0;
}

cv::Mat SceneChangeDetector::ComputeHistogram(const cv::Mat& bgr_small) {
  cv::Mat hsv;
  cv::cvtColor(bgr_small, hsv, cv::COLOR_BGR2HSV);
  const int h_bins = 30, s_bins = 32;
  const int hist_size[]   = {h_bins, s_bins};
  const float h_ranges[] = {0.f, 180.f};
  const float s_ranges[] = {0.f, 256.f};
  const float* ranges[] = {h_ranges, s_ranges};
  const int channels[]  = {0, 1};

  cv::Mat hist;
  cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, hist_size, ranges, true, false);
  cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
  return hist;
}

uint64_t SceneChangeDetector::ComputePHash(const cv::Mat& bgr_small) {
  cv::Mat gray, resized, dct_in, dct_out;
  cv::cvtColor(bgr_small, gray, cv::COLOR_BGR2GRAY);
  cv::resize(gray, resized, cv::Size(32, 32));
  resized.convertTo(dct_in, CV_32F);
  cv::dct(dct_in, dct_out);

  // Top-left 8x8 (excluding DC) is the perceptual signature.
  cv::Mat top = dct_out(cv::Rect(0, 0, 8, 8)).clone();
  // Set DC to median to avoid biasing the threshold.
  std::vector<float> vals;
  vals.reserve(64);
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      if (r == 0 && c == 0) continue;
      vals.push_back(top.at<float>(r, c));
    }
  }
  std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
  const float median = vals[vals.size() / 2];

  uint64_t hash = 0;
  int bit = 0;
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c, ++bit) {
      if (top.at<float>(r, c) > median) hash |= (1ULL << bit);
    }
  }
  return hash;
}

bool SceneChangeDetector::IsSceneChange(const cv::Mat& bgr) {
  cv::Mat small;
  cv::resize(bgr, small, cv::Size(cfg_.work_size, cfg_.work_size));

  cv::Mat hist = ComputeHistogram(small);
  uint64_t phash = ComputePHash(small);

  if (!has_prev_) {
    has_prev_ = true;
    prev_hist_ = hist;
    prev_phash_ = phash;
    return true;  // first frame is always emitted
  }

  const double bhatt = cv::compareHist(prev_hist_, hist, cv::HISTCMP_BHATTACHARYYA);
  const int hamming = std::bitset<64>(prev_phash_ ^ phash).count();

  const bool changed = bhatt > cfg_.hist_threshold || hamming >= cfg_.phash_threshold;
  if (changed) {
    prev_hist_  = hist;
    prev_phash_ = phash;
  }
  return changed;
}

}  // namespace vlm
