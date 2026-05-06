#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "common/frame.h"

namespace grpc { class Server; }

namespace vlm {

class CapturePipeline;

// gRPC service exposed to the prompt app. Drives capture-mode changes
// on the pipeline and forwards user queries to the server, streaming
// the response back to the prompt app.
class PromptAppService {
 public:
  struct Config {
    std::string listen_addr = "0.0.0.0:50051";
  };

  PromptAppService(Config cfg, CapturePipeline* pipeline);
  ~PromptAppService();

  bool Start();
  void Stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vlm
