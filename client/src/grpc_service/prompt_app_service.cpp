#include "grpc_service/prompt_app_service.h"

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "common/logger.h"
#include "pipeline/capture_pipeline.h"
#include "prompt_app.grpc.pb.h"

namespace vlm {

namespace {

struct ChunkBuffer {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<::vlm::QueryResponse> chunks;
  bool finished = false;
  bool cancelled = false;
};

class ServiceImpl final : public ::vlm::PromptApp::Service {
 public:
  explicit ServiceImpl(CapturePipeline* pipeline) : pipeline_(pipeline) {
    pipeline_->SetResponseSubscriber(
        [this](const std::string& sid, const std::string& text,
               uint32_t chunk, bool done) {
          DispatchChunk(sid, text, chunk, done);
        });
  }

  grpc::Status NotifyTyping(grpc::ServerContext*,
                            const ::vlm::TypingState* req,
                            ::vlm::Ack* resp) override {
    pipeline_->SetTyping(req->is_typing());
    resp->set_ok(true);
    resp->set_message(req->is_typing() ? "typing on" : "typing off");
    return grpc::Status::OK;
  }

  grpc::Status SubmitQuery(grpc::ServerContext* ctx,
                           const ::vlm::Query* req,
                           grpc::ServerWriter<::vlm::QueryResponse>* writer) override {
    const std::string sid = req->session_id().empty()
                                ? ("sess-" + std::to_string(NextSessionId()))
                                : req->session_id();

    auto buf = std::make_shared<ChunkBuffer>();
    {
      std::lock_guard<std::mutex> lk(sessions_mu_);
      sessions_[sid] = buf;
    }

    if (!pipeline_->SubmitQuery(sid, req->text(), req->num_frames_hint(),
                                req->trigger_ts_us(), req->pre_window_ms(),
                                req->post_window_ms())) {
      ::vlm::QueryResponse err;
      err.set_session_id(sid);
      err.set_text("[error] WebRTC channel not ready");
      err.set_done(true);
      writer->Write(err);
      EraseSession(sid);
      return grpc::Status::OK;
    }

    while (true) {
      ::vlm::QueryResponse out;
      bool finished = false;
      {
        std::unique_lock<std::mutex> lk(buf->mu);
        buf->cv.wait_for(lk, std::chrono::seconds(1), [&] {
          return !buf->chunks.empty() || buf->finished || ctx->IsCancelled();
        });
        if (ctx->IsCancelled()) {
          buf->cancelled = true;
          break;
        }
        if (!buf->chunks.empty()) {
          out = std::move(buf->chunks.front());
          buf->chunks.pop_front();
        } else if (buf->finished) {
          finished = true;
        } else {
          continue;  // spurious wakeup / timeout
        }
      }
      if (finished) break;
      if (!writer->Write(out)) break;
      if (out.done()) break;
    }

    EraseSession(sid);
    return grpc::Status::OK;
  }

  grpc::Status HealthCheck(grpc::ServerContext*,
                           const ::vlm::Empty*,
                           ::vlm::HealthStatus* resp) override {
    resp->set_client_ok(true);
    resp->set_webrtc_connected(pipeline_->webrtc_connected());
    resp->set_capture_mode(ToString(pipeline_->mode()));
    return grpc::Status::OK;
  }

 private:
  void DispatchChunk(const std::string& sid, const std::string& text,
                     uint32_t chunk, bool done) {
    std::shared_ptr<ChunkBuffer> buf;
    {
      std::lock_guard<std::mutex> lk(sessions_mu_);
      auto it = sessions_.find(sid);
      if (it == sessions_.end()) return;
      buf = it->second;
    }
    {
      std::lock_guard<std::mutex> lk(buf->mu);
      ::vlm::QueryResponse r;
      r.set_session_id(sid);
      r.set_text(text);
      r.set_chunk_index(chunk);
      r.set_done(done);
      buf->chunks.emplace_back(std::move(r));
      if (done) buf->finished = true;
    }
    buf->cv.notify_all();
  }

  void EraseSession(const std::string& sid) {
    std::lock_guard<std::mutex> lk(sessions_mu_);
    sessions_.erase(sid);
  }

  static uint64_t NextSessionId() {
    static std::atomic<uint64_t> ctr{0};
    return ++ctr;
  }

  CapturePipeline* pipeline_;
  std::mutex sessions_mu_;
  std::unordered_map<std::string, std::shared_ptr<ChunkBuffer>> sessions_;
};

}  // namespace

struct PromptAppService::Impl {
  Config cfg;
  CapturePipeline* pipeline = nullptr;
  std::unique_ptr<ServiceImpl> service;
  std::unique_ptr<grpc::Server> server;
};

PromptAppService::PromptAppService(Config cfg, CapturePipeline* pipeline)
    : impl_(std::make_unique<Impl>()) {
  impl_->cfg = std::move(cfg);
  impl_->pipeline = pipeline;
}

PromptAppService::~PromptAppService() { Stop(); }

bool PromptAppService::Start() {
  impl_->service = std::make_unique<ServiceImpl>(impl_->pipeline);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(impl_->cfg.listen_addr, grpc::InsecureServerCredentials());
  builder.RegisterService(impl_->service.get());
  impl_->server = builder.BuildAndStart();
  if (!impl_->server) {
    VLM_LOG_ERROR("grpc", "failed to bind ", impl_->cfg.listen_addr);
    return false;
  }
  VLM_LOG_INFO("grpc", "listening on ", impl_->cfg.listen_addr);
  return true;
}

void PromptAppService::Stop() {
  if (impl_ && impl_->server) {
    impl_->server->Shutdown(std::chrono::system_clock::now() +
                            std::chrono::milliseconds(200));
    impl_->server.reset();
  }
}

}  // namespace vlm
