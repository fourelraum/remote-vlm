#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include "common/logger.h"
#include "grpc_service/prompt_app_service.h"
#include "pipeline/capture_pipeline.h"

namespace {

std::atomic<bool> g_stop{false};

void OnSignal(int /*sig*/) { g_stop.store(true); }

std::string EnvOr(const char* name, const std::string& def) {
  const char* v = std::getenv(name);
  return (v && *v) ? std::string(v) : def;
}

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "  --signaling-url URL    HTTP signaling endpoint of the server\n"
      << "                         (default: http://localhost:8080/offer)\n"
      << "  --grpc-listen ADDR     gRPC bind address for prompt-app\n"
      << "                         (default: 0.0.0.0:50051)\n"
      << "  --display NAME         X11 display (default: $DISPLAY)\n"
      << "  --grab-fps N           polling rate of grab loop (default: 30)\n"
      << "  --max-fps N            idle-mode emit ceiling (default: 5)\n"
      << "  --stun URL             STUN server (repeatable)\n"
      << "  -h, --help             this help\n";
}

}  // namespace

int main(int argc, char** argv) {
  vlm::CapturePipeline::Config cfg;
  cfg.webrtc.signaling_url = EnvOr("REMOTE_VLM_SIGNALING_URL",
                                   "http://localhost:8080/offer");
  vlm::PromptAppService::Config grpc_cfg;
  grpc_cfg.listen_addr = EnvOr("REMOTE_VLM_GRPC_LISTEN", "0.0.0.0:50051");

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << a << "\n";
        std::exit(2);
      }
      return std::string(argv[++i]);
    };
    if (a == "-h" || a == "--help") { PrintUsage(argv[0]); return 0; }
    else if (a == "--signaling-url") cfg.webrtc.signaling_url = next();
    else if (a == "--grpc-listen")   grpc_cfg.listen_addr = next();
    else if (a == "--display")       cfg.capture.display = next();
    else if (a == "--grab-fps")      cfg.grab_fps = std::stod(next());
    else if (a == "--max-fps")       cfg.aff.max_fps = std::stod(next());
    else if (a == "--stun")          cfg.webrtc.ice_servers.push_back(next());
    else { std::cerr << "unknown arg: " << a << "\n"; PrintUsage(argv[0]); return 2; }
  }

  std::signal(SIGINT,  OnSignal);
  std::signal(SIGTERM, OnSignal);

  vlm::CapturePipeline pipeline(cfg);
  if (!pipeline.Start()) {
    VLM_LOG_ERROR("main", "failed to start capture pipeline");
    return 1;
  }

  vlm::PromptAppService grpc_service(grpc_cfg, &pipeline);
  if (!grpc_service.Start()) {
    pipeline.Stop();
    return 1;
  }

  VLM_LOG_INFO("main", "remote-vlm client running; ctrl-c to exit");
  while (!g_stop.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  grpc_service.Stop();
  pipeline.Stop();
  return 0;
}
