#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace rtc {
class PeerConnection;
class DataChannel;
class Track;
}  // namespace rtc

namespace vlm {

// Wraps a libdatachannel PeerConnection. The client uses an out-of-band
// HTTP signaling exchange (POST offer, receive answer) hosted by the
// Python server — keeping the C++ side dependency-free of HTTP servers.
//
// Two streams are exposed:
//   * a raw video track that ships JPEG-encoded screen frames,
//   * a "control" data channel that carries JSON messages
//       { "type": "query", "session_id": ..., "text": ... }
//       { "type": "response", "session_id": ..., "text": ..., "done": ... }
class WebRtcClient {
 public:
  using ResponseCallback =
      std::function<void(const std::string& session_id,
                         const std::string& text,
                         uint32_t chunk,
                         bool done)>;

  struct Config {
    std::string signaling_url;          // e.g. "http://localhost:8080/offer"
    std::vector<std::string> ice_servers;  // optional STUN servers
  };

  explicit WebRtcClient(Config cfg);
  ~WebRtcClient();

  WebRtcClient(const WebRtcClient&) = delete;
  WebRtcClient& operator=(const WebRtcClient&) = delete;

  // Bring up the peer connection and exchange SDP with the server.
  bool Connect();
  void Close();

  bool IsConnected() const { return connected_.load(); }

  // Send a single screen frame. JPEG-encoded then framed as a binary
  // data-channel payload (since libdatachannel's Track API requires an
  // RTP encoder we don't want to ship; the server demuxes by channel
  // label). The capture timestamp is stamped into the packet header so
  // the server can correlate frames with prompt-trigger timestamps.
  bool SendFrame(const cv::Mat& bgr_frame, uint64_t seq);

  // Send a user query. ``trigger_ts_us`` is the wall-clock microsecond
  // timestamp the prompt is anchored to (e.g. the moment the user
  // started typing); the server samples frames in
  // ``[trigger - pre_window_ms, trigger + post_window_ms]``. Pass 0
  // for trigger_ts_us to disable time-window filtering. The matching
  // response is delivered to the ResponseCallback installed via
  // SetResponseCallback().
  bool SendQuery(const std::string& session_id, const std::string& text,
                 uint32_t num_frames_hint, int64_t trigger_ts_us,
                 uint32_t pre_window_ms, uint32_t post_window_ms);

  void SetResponseCallback(ResponseCallback cb);

 private:
  Config cfg_;
  std::shared_ptr<rtc::PeerConnection> pc_;
  std::shared_ptr<rtc::DataChannel> control_dc_;
  std::shared_ptr<rtc::DataChannel> frame_dc_;
  std::atomic<bool> connected_{false};

  std::mutex cb_mu_;
  ResponseCallback response_cb_;

  void OnControlMessage(const std::string& payload);
  bool ExchangeSdp(const std::string& offer_sdp, std::string& answer_sdp);
};

}  // namespace vlm
