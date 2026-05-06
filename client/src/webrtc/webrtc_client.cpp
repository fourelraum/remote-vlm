#include "webrtc/webrtc_client.h"

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rtc/rtc.hpp>

#include <chrono>
#include <cstring>
#include <future>
#include <sstream>
#include <thread>

#include "common/logger.h"

// Tiny synchronous HTTP POST helper using a TCP socket. We deliberately
// avoid pulling libcurl since signaling is a single short request.
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace vlm {

using nlohmann::json;

namespace {

bool HttpPost(const std::string& url,
              const std::string& body,
              std::string& out_response) {
  // url: http://host[:port]/path
  if (url.rfind("http://", 0) != 0) {
    VLM_LOG_ERROR("webrtc", "only http:// signaling URLs are supported, got ", url);
    return false;
  }
  std::string rest = url.substr(7);
  auto slash = rest.find('/');
  std::string hostport = (slash == std::string::npos) ? rest : rest.substr(0, slash);
  std::string path     = (slash == std::string::npos) ? "/"  : rest.substr(slash);

  std::string host = hostport;
  int port = 80;
  auto colon = hostport.find(':');
  if (colon != std::string::npos) {
    host = hostport.substr(0, colon);
    port = std::stoi(hostport.substr(colon + 1));
  }

  addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0) {
    VLM_LOG_ERROR("webrtc", "getaddrinfo failed for ", host);
    return false;
  }

  int sock = -1;
  for (addrinfo* p = res; p; p = p->ai_next) {
    sock = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (sock < 0) continue;
    if (::connect(sock, p->ai_addr, p->ai_addrlen) == 0) break;
    ::close(sock);
    sock = -1;
  }
  freeaddrinfo(res);
  if (sock < 0) {
    VLM_LOG_ERROR("webrtc", "tcp connect failed to ", host, ":", port);
    return false;
  }

  std::ostringstream req;
  req << "POST " << path << " HTTP/1.1\r\n"
      << "Host: " << host << ':' << port << "\r\n"
      << "Content-Type: application/json\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "Connection: close\r\n\r\n"
      << body;
  const std::string req_str = req.str();
  if (::send(sock, req_str.data(), req_str.size(), 0) < 0) {
    ::close(sock);
    return false;
  }

  std::string raw;
  char buf[4096];
  ssize_t n;
  while ((n = ::recv(sock, buf, sizeof(buf), 0)) > 0) {
    raw.append(buf, buf + n);
  }
  ::close(sock);

  auto header_end = raw.find("\r\n\r\n");
  if (header_end == std::string::npos) {
    VLM_LOG_ERROR("webrtc", "malformed signaling response");
    return false;
  }
  out_response = raw.substr(header_end + 4);
  return true;
}

}  // namespace

WebRtcClient::WebRtcClient(Config cfg) : cfg_(std::move(cfg)) {}
WebRtcClient::~WebRtcClient() { Close(); }

void WebRtcClient::SetResponseCallback(ResponseCallback cb) {
  std::lock_guard<std::mutex> lk(cb_mu_);
  response_cb_ = std::move(cb);
}

bool WebRtcClient::ExchangeSdp(const std::string& offer_sdp, std::string& answer_sdp) {
  json req = {
      {"sdp", offer_sdp},
      {"type", "offer"},
  };
  std::string resp_body;
  if (!HttpPost(cfg_.signaling_url, req.dump(), resp_body)) return false;
  try {
    auto resp = json::parse(resp_body);
    answer_sdp = resp.at("sdp").get<std::string>();
    return true;
  } catch (const std::exception& e) {
    VLM_LOG_ERROR("webrtc", "bad SDP answer: ", e.what(), " body=", resp_body);
    return false;
  }
}

bool WebRtcClient::Connect() {
  rtc::Configuration rtc_cfg;
  for (const auto& s : cfg_.ice_servers) {
    rtc_cfg.iceServers.emplace_back(s);
  }

  pc_ = std::make_shared<rtc::PeerConnection>(rtc_cfg);

  // libdatachannel uses trickle ICE: onLocalDescription fires
  // immediately with no candidates, then ICE candidates arrive
  // separately. For a single-shot HTTP signaling exchange we need to
  // wait until gathering completes and then read pc_->localDescription().
  auto gather_done = std::make_shared<std::promise<void>>();
  auto gather_future = gather_done->get_future();
  std::atomic<bool> gather_signaled{false};
  pc_->onGatheringStateChange(
      [gather_done, &gather_signaled](rtc::PeerConnection::GatheringState s) {
        if (s == rtc::PeerConnection::GatheringState::Complete &&
            !gather_signaled.exchange(true)) {
          gather_done->set_value();
        }
      });

  pc_->onStateChange([this](rtc::PeerConnection::State s) {
    VLM_LOG_INFO("webrtc", "pc state=", static_cast<int>(s));
    if (s == rtc::PeerConnection::State::Connected) {
      connected_.store(true);
    } else if (s == rtc::PeerConnection::State::Disconnected ||
               s == rtc::PeerConnection::State::Failed ||
               s == rtc::PeerConnection::State::Closed) {
      connected_.store(false);
    }
  });

  control_dc_ = pc_->createDataChannel("control");
  control_dc_->onOpen([] { VLM_LOG_INFO("webrtc", "control channel open"); });
  control_dc_->onMessage([this](rtc::message_variant msg) {
    if (std::holds_alternative<std::string>(msg)) {
      OnControlMessage(std::get<std::string>(msg));
    }
  });

  rtc::DataChannelInit frame_init;
  frame_init.reliability.unordered = true;
  frame_init.reliability.maxRetransmits = 0;  // best-effort: drop late frames
  frame_dc_ = pc_->createDataChannel("frames", frame_init);
  frame_dc_->onOpen([] { VLM_LOG_INFO("webrtc", "frames channel open"); });

  pc_->setLocalDescription();

  if (gather_future.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
    VLM_LOG_ERROR("webrtc", "timeout waiting for ICE gathering to complete");
    return false;
  }
  auto local_desc = pc_->localDescription();
  if (!local_desc.has_value()) {
    VLM_LOG_ERROR("webrtc", "no local description after gathering");
    return false;
  }
  const std::string offer_sdp = std::string(*local_desc);

  std::string answer_sdp;
  if (!ExchangeSdp(offer_sdp, answer_sdp)) return false;

  pc_->setRemoteDescription(rtc::Description(answer_sdp, "answer"));

  // Wait briefly for ICE to come up.
  for (int i = 0; i < 100 && !connected_.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return connected_.load();
}

void WebRtcClient::Close() {
  if (frame_dc_)   { frame_dc_->close();   frame_dc_.reset();   }
  if (control_dc_) { control_dc_->close(); control_dc_.reset(); }
  if (pc_)         { pc_->close();         pc_.reset();         }
  connected_.store(false);
}

bool WebRtcClient::SendFrame(const cv::Mat& bgr_frame, uint64_t seq) {
  if (!frame_dc_ || !frame_dc_->isOpen()) return false;

  std::vector<uchar> jpeg;
  const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 75};
  if (!cv::imencode(".jpg", bgr_frame, jpeg, params)) return false;

  // Frame protocol:
  //   [seq:u32 LE][capture_ts_us:u64 LE][size:u32 LE][JPEG bytes]
  const uint64_t ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();
  constexpr size_t kHeaderSize = 16;
  std::vector<std::byte> packet(kHeaderSize + jpeg.size());
  uint32_t seq32 = static_cast<uint32_t>(seq);
  uint64_t ts64  = ts_us;
  uint32_t size  = static_cast<uint32_t>(jpeg.size());
  std::memcpy(packet.data(),      &seq32, 4);
  std::memcpy(packet.data() + 4,  &ts64,  8);
  std::memcpy(packet.data() + 12, &size,  4);
  std::memcpy(packet.data() + kHeaderSize, jpeg.data(), jpeg.size());

  return frame_dc_->send(rtc::binary(packet.begin(), packet.end()));
}

bool WebRtcClient::SendQuery(const std::string& session_id,
                             const std::string& text,
                             uint32_t num_frames_hint,
                             int64_t trigger_ts_us,
                             uint32_t pre_window_ms,
                             uint32_t post_window_ms) {
  if (!control_dc_ || !control_dc_->isOpen()) return false;
  json msg = {
      {"type", "query"},
      {"session_id", session_id},
      {"text", text},
      {"num_frames_hint", num_frames_hint},
      {"trigger_ts_us", trigger_ts_us},
      {"pre_window_ms", pre_window_ms},
      {"post_window_ms", post_window_ms},
  };
  return control_dc_->send(msg.dump());
}

void WebRtcClient::OnControlMessage(const std::string& payload) {
  try {
    auto msg = json::parse(payload);
    const std::string type = msg.value("type", "");
    if (type != "response") return;
    const std::string sid  = msg.value("session_id", "");
    const std::string text = msg.value("text", "");
    const uint32_t chunk   = msg.value("chunk_index", 0u);
    const bool done        = msg.value("done", false);

    ResponseCallback cb;
    {
      std::lock_guard<std::mutex> lk(cb_mu_);
      cb = response_cb_;
    }
    if (cb) cb(sid, text, chunk, done);
  } catch (const std::exception& e) {
    VLM_LOG_WARN("webrtc", "invalid control message: ", e.what());
  }
}

}  // namespace vlm
