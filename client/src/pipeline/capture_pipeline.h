#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "aff/adaptive_frame_filter.h"
#include "capture/screen_capture.h"
#include "common/frame.h"
#include "scd/scene_change_detector.h"
#include "webrtc/webrtc_client.h"

namespace vlm {

// Glues together the capture loop, SCD, AFF and the WebRTC sender. The
// pipeline runs an internal thread that grabs frames at a high target
// rate (default 30 FPS); whether a given frame is forwarded depends on
// the active CaptureMode.
class CapturePipeline {
 public:
  struct Config {
    ScreenCapture::Config capture;
    SceneChangeDetector::Config scd;
    AdaptiveFrameFilter::Config aff;
    WebRtcClient::Config webrtc;

    // Polling rate of the grab loop. Actual emit rate is gated by AFF
    // (idle mode) or fixed to 1 FPS in typing/query modes.
    double grab_fps = 30.0;
  };

  // Subscriber notified about an incoming VLM response. Used by the
  // gRPC service to forward chunks back to the prompt app.
  using ResponseSubscriber =
      std::function<void(const std::string& session_id,
                         const std::string& text,
                         uint32_t chunk,
                         bool done)>;

  explicit CapturePipeline(Config cfg);
  ~CapturePipeline();

  bool Start();
  void Stop();

  // Capture-mode controls — invoked by the gRPC service on behalf of
  // the prompt app.
  void SetTyping(bool typing);
  // Locks 1 FPS until the matching response stream completes.
  void BeginQuery(const std::string& session_id);
  void EndQuery(const std::string& session_id);

  // Fire-and-forget: enqueue a query into the WebRTC control channel.
  bool SubmitQuery(const std::string& session_id,
                   const std::string& text,
                   uint32_t num_frames_hint);

  void SetResponseSubscriber(ResponseSubscriber cb);

  CaptureMode mode() const { return mode_.load(); }
  bool webrtc_connected() const {
    return webrtc_ ? webrtc_->IsConnected() : false;
  }

 private:
  void GrabLoop();
  void OnVlmResponse(const std::string& session_id,
                     const std::string& text,
                     uint32_t chunk,
                     bool done);

  Config cfg_;
  std::unique_ptr<ScreenCapture>       capture_;
  std::unique_ptr<SceneChangeDetector> scd_;
  std::unique_ptr<AdaptiveFrameFilter> aff_;
  std::unique_ptr<WebRtcClient>        webrtc_;

  std::thread thread_;
  std::atomic<bool> running_{false};
  std::atomic<CaptureMode> mode_{CaptureMode::kIdle};

  std::mutex sub_mu_;
  ResponseSubscriber subscriber_;

  std::atomic<uint64_t> frame_seq_{0};
};

}  // namespace vlm
