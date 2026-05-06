#pragma once

#include <chrono>
#include <ctime>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string_view>

namespace vlm {

// Lightweight thread-safe logger. Avoids pulling in spdlog/glog.
class Logger {
 public:
  static Logger& Instance() {
    static Logger inst;
    return inst;
  }

  template <typename... Args>
  void Info(std::string_view tag, Args&&... args) {
    Log("INFO ", tag, std::forward<Args>(args)...);
  }
  template <typename... Args>
  void Warn(std::string_view tag, Args&&... args) {
    Log("WARN ", tag, std::forward<Args>(args)...);
  }
  template <typename... Args>
  void Error(std::string_view tag, Args&&... args) {
    Log("ERROR", tag, std::forward<Args>(args)...);
  }

 private:
  std::mutex mu_;

  template <typename... Args>
  void Log(std::string_view level, std::string_view tag, Args&&... args) {
    std::ostringstream oss;
    using clock = std::chrono::system_clock;
    auto now = clock::to_time_t(clock::now());
    std::tm tm{};
    localtime_r(&now, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", &tm);
    oss << '[' << buf << "][" << level << "][" << tag << "] ";
    (oss << ... << std::forward<Args>(args));
    std::lock_guard<std::mutex> lk(mu_);
    std::cerr << oss.str() << std::endl;
  }
};

#define VLM_LOG_INFO(tag, ...)  ::vlm::Logger::Instance().Info(tag, __VA_ARGS__)
#define VLM_LOG_WARN(tag, ...)  ::vlm::Logger::Instance().Warn(tag, __VA_ARGS__)
#define VLM_LOG_ERROR(tag, ...) ::vlm::Logger::Instance().Error(tag, __VA_ARGS__)

}  // namespace vlm
