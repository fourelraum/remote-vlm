#pragma once

#include <atomic>
#include <memory>
#include <opencv2/core.hpp>
#include <string>

namespace vlm {

// Thin X11 / XShm screen grabber. One instance per display.
class ScreenCapture {
 public:
  struct Config {
    std::string display;   // e.g. ":0". Empty -> $DISPLAY
    int x = 0;
    int y = 0;
    int width = 0;         // 0 -> full root-window width
    int height = 0;        // 0 -> full root-window height
  };

  explicit ScreenCapture(Config cfg);
  ~ScreenCapture();

  ScreenCapture(const ScreenCapture&) = delete;
  ScreenCapture& operator=(const ScreenCapture&) = delete;

  bool Open();
  void Close();

  // Grabs the current frame into `out` as BGR. Returns false on failure.
  bool Grab(cv::Mat& out);

  int width()  const { return cfg_.width;  }
  int height() const { return cfg_.height; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Config cfg_;
  std::atomic<bool> open_{false};
};

}  // namespace vlm
