#include "pipeline/capture_pipeline.h"

#include <chrono>

#include "common/logger.h"

namespace vlm {

namespace {
constexpr double kForcedFps = 1.0;
}

CapturePipeline::CapturePipeline(Config cfg) : cfg_(std::move(cfg)) {
  capture_ = std::make_unique<ScreenCapture>(cfg_.capture);
  scd_     = std::make_unique<SceneChangeDetector>(cfg_.scd);
  aff_     = std::make_unique<AdaptiveFrameFilter>(cfg_.aff);
  webrtc_  = std::make_unique<WebRtcClient>(cfg_.webrtc);
}

CapturePipeline::~CapturePipeline() { Stop(); }

bool CapturePipeline::Start() {
  if (running_.exchange(true)) return true;

  if (!capture_->Open()) {
    running_.store(false);
    return false;
  }
  if (!webrtc_->Connect()) {
    VLM_LOG_WARN("pipeline", "webrtc connect failed; will keep retrying in grab loop");
  }
  webrtc_->SetResponseCallback(
      [this](const std::string& sid, const std::string& text, uint32_t chunk, bool done) {
        OnVlmResponse(sid, text, chunk, done);
      });

  thread_ = std::thread([this] { GrabLoop(); });
  return true;
}

void CapturePipeline::Stop() {
  if (!running_.exchange(false)) return;
  if (thread_.joinable()) thread_.join();
  webrtc_->Close();
  capture_->Close();
}

void CapturePipeline::SetResponseSubscriber(ResponseSubscriber cb) {
  std::lock_guard<std::mutex> lk(sub_mu_);
  subscriber_ = std::move(cb);
}

void CapturePipeline::SetTyping(bool typing) {
  if (typing) {
    if (mode_.load() != CaptureMode::kQuery) {
      mode_.store(CaptureMode::kTyping);
      VLM_LOG_INFO("pipeline", "mode -> typing (1 FPS)");
    }
  } else {
    if (mode_.load() == CaptureMode::kTyping) {
      mode_.store(CaptureMode::kIdle);
      scd_->Reset();
      aff_->Reset();
      VLM_LOG_INFO("pipeline", "mode -> idle (SCD/AFF gated)");
    }
  }
}

void CapturePipeline::BeginQuery(const std::string& session_id) {
  mode_.store(CaptureMode::kQuery);
  VLM_LOG_INFO("pipeline", "mode -> query session=", session_id);
}

void CapturePipeline::EndQuery(const std::string& /*session_id*/) {
  mode_.store(CaptureMode::kIdle);
  scd_->Reset();
  aff_->Reset();
  VLM_LOG_INFO("pipeline", "mode -> idle (query complete)");
}

bool CapturePipeline::SubmitQuery(const std::string& session_id,
                                  const std::string& text,
                                  uint32_t num_frames_hint,
                                  int64_t trigger_ts_us,
                                  uint32_t pre_window_ms,
                                  uint32_t post_window_ms) {
  BeginQuery(session_id);
  return webrtc_->SendQuery(session_id, text, num_frames_hint,
                            trigger_ts_us, pre_window_ms, post_window_ms);
}

void CapturePipeline::OnVlmResponse(const std::string& session_id,
                                    const std::string& text,
                                    uint32_t chunk,
                                    bool done) {
  ResponseSubscriber cb;
  {
    std::lock_guard<std::mutex> lk(sub_mu_);
    cb = subscriber_;
  }
  if (cb) cb(session_id, text, chunk, done);
  if (done) EndQuery(session_id);
}

void CapturePipeline::GrabLoop() {
  using clock = std::chrono::steady_clock;
  const auto grab_interval =
      std::chrono::duration_cast<clock::duration>(
          std::chrono::duration<double>(1.0 / cfg_.grab_fps));
  auto next_tick = clock::now();
  auto last_forced_emit = clock::time_point{};

  cv::Mat frame;
  while (running_.load()) {
    next_tick += grab_interval;
    if (!capture_->Grab(frame) || frame.empty()) {
      std::this_thread::sleep_until(next_tick);
      continue;
    }

    bool emit = false;
    const CaptureMode mode = mode_.load();
    if (mode == CaptureMode::kTyping || mode == CaptureMode::kQuery) {
      // Forced 1 FPS.
      const auto now = clock::now();
      const auto forced_interval =
          std::chrono::duration_cast<clock::duration>(
              std::chrono::duration<double>(1.0 / kForcedFps));
      if (last_forced_emit.time_since_epoch().count() == 0 ||
          (now - last_forced_emit) >= forced_interval) {
        last_forced_emit = now;
        aff_->EvaluateForced(frame);
        emit = true;
      }
    } else {
      // Idle mode: SCD + AFF gating.
      if (scd_->IsSceneChange(frame)) {
        const auto decision = aff_->Evaluate(frame);
        emit = (decision == AdaptiveFrameFilter::Decision::kAccept);
      }
    }

    if (emit) {
      // Lazy reconnect.
      if (!webrtc_->IsConnected()) {
        if (!webrtc_->Connect()) {
          std::this_thread::sleep_until(next_tick);
          continue;
        }
      }
      const uint64_t seq = ++frame_seq_;
      if (!webrtc_->SendFrame(frame, seq)) {
        VLM_LOG_WARN("pipeline", "webrtc SendFrame failed seq=", seq);
      }
    }

    std::this_thread::sleep_until(next_tick);
  }
}

}  // namespace vlm
