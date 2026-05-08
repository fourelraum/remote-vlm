# remote-vlm

화면을 캡처해서 원격 VLM(Qwen3-VL)에 질의하는 데스크톱 보조 시스템.

```
┌──────────────┐  gRPC   ┌──────────────────────────────────────┐  WebRTC  ┌─────────────────┐
│ Prompt App   │ ◄────► │           Client (C++)                │ ◄──────► │ Server (Python) │
│  (Python)    │        │  Capture → SCD → AFF → WebRTC Tx      │          │ aiortc + Qwen   │
│              │        │  gRPC server (typing/query 알림 수신) │          │  VLM 추론       │
└──────────────┘        └──────────────────────────────────────┘          └─────────────────┘
```

## 디렉터리 구조

| Path | 설명 |
| ---- | ---- |
| `proto/`        | gRPC 인터페이스 (prompt app ↔ client) |
| `client/`       | C++ 클라이언트. X11 캡처 + SCD + AFF + WebRTC Tx + gRPC 서버 |
| `server/`       | Python 서버. aiortc 로 WebRTC 종단, Qwen VLM 추론 |
| `prompt_app/`   | gRPC 로 client 와 통신하는 샘플 prompt CLI |
| `scripts/`      | 빌드 / 코드 생성 스크립트 |

## 데이터 흐름

1. **Idle 모드** — 클라이언트가 30 FPS 로 화면을 폴링하면서
   - **SCD (Scene Change Detection)**: HSV 히스토그램의 Bhattacharyya 거리 +
     8×8 pHash 해밍 거리로 "장면 전환"을 판정
   - **AFF (Adaptive Frame Filtering)**: 모션 비율, 라플라시안 분산(블러),
     최대 FPS 상한으로 가치 있는 프레임만 통과시킴
   - 통과한 프레임만 WebRTC `frames` 데이터 채널로 송신
2. **Typing 모드** — Prompt app 이 키 입력을 감지해
   `NotifyTyping(is_typing=true)` 를 호출하면 클라이언트가 SCD/AFF 를
   우회하고 **1 FPS** 로 강제 캡처
3. **Query 모드** — 사용자가 `<Enter>` 로 질의를 제출하면 prompt app 이
   `SubmitQuery` (server-streaming RPC) 호출 →
   클라이언트는 `control` 데이터 채널에 JSON 으로 query 전달 →
   서버가 최근 N 프레임 + 프롬프트로 Qwen 에 추론 →
   응답 청크를 다시 `control` 채널로 흘려보내고, 클라이언트는 이를
   gRPC 스트림으로 prompt app 에 전달

## 통신 프로토콜

### prompt_app ↔ client (gRPC)

`proto/prompt_app.proto` 참조.

* `NotifyTyping(TypingState) returns (Ack)` — 타이핑 시작/종료 알림
* `SubmitQuery(Query) returns (stream QueryResponse)` — 질의 제출 +
  응답 스트리밍
* `HealthCheck(Empty) returns (HealthStatus)` — 연결/모드 점검

### client ↔ server (WebRTC)

* `frames` data channel (binary, unordered, max-retransmits 0):
  ```
  [seq:u32 LE][capture_ts_us:u64 LE][size:u32 LE][JPEG bytes]
  ```
  `capture_ts_us` 는 클라이언트 wall-clock (Unix epoch µs). 서버는 이 값을
  prompt 의 `trigger_ts_us` 와 비교해 시간-윈도우 안의 프레임만 추론에 사용.
* `control` data channel (text JSON, reliable):
  ```json
  {"type":"query","session_id":"...","text":"...","num_frames_hint":0,
   "trigger_ts_us":1715000123456789,"pre_window_ms":3000,"post_window_ms":3000}
  {"type":"response","session_id":"...","text":"...","chunk_index":0,"done":false}
  ```
  `trigger_ts_us` 는 prompt 가 anchor 되는 wall-clock 순간 (보통 첫 키스트로크
  시각). 서버는 `trigger_ts_us + post_window_ms` 시각 이후 프레임이 도착할
  때까지 (혹은 `post_window_ms + 2 s` 타임아웃까지) 대기한 뒤
  `[trigger - pre, trigger + post]` 범위로 프레임을 필터링해 VLM 에 전달.
  `trigger_ts_us = 0` 이면 시간 필터링을 끄고 deque 전체를 사용.
* Signaling: HTTP `POST /offer` (aiortc 데모와 동일한 패턴)

## 빌드 & 실행

### 시스템 패키지 (Ubuntu 22.04 기준)

```bash
sudo apt install -y build-essential cmake pkg-config \
    libopencv-dev \
    libx11-dev libxext-dev \
    libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
    nlohmann-json3-dev \
    libdatachannel-dev    # 없으면 https://github.com/paullouisageneau/libdatachannel 에서 빌드
```

### 코드 생성 (Python 측)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r prompt_app/requirements.txt
scripts/gen_proto.sh
```

### 서버 실행

```bash
pip install -r server/requirements.txt
# 모델 로딩 없이 파이프라인만 검증
REMOTE_VLM_STUB=1 python -m server.main --host 0.0.0.0 --port 8080
# 실제 Qwen 추론 (default: Qwen/Qwen3-VL-4B-Instruct)
python -m server.main --model Qwen/Qwen3-VL-4B-Instruct
```

### 클라이언트 빌드 / 실행

```bash
scripts/build_client.sh
./client/build/remote_vlm_client \
    --signaling-url http://localhost:8080/offer \
    --grpc-listen   0.0.0.0:50051
```

### Prompt App 실행

```bash
python -m prompt_app.app --client localhost:50051
```

## 모듈 별 핵심 파일

| 모듈 | 위치 |
| ---- | ---- |
| Screen capture (XShm) | `client/src/capture/screen_capture.cpp` |
| Scene Change Detection | `client/src/scd/scene_change_detector.cpp` |
| Adaptive Frame Filter | `client/src/aff/adaptive_frame_filter.cpp` |
| WebRTC client | `client/src/webrtc/webrtc_client.cpp` |
| gRPC service | `client/src/grpc_service/prompt_app_service.cpp` |
| Pipeline orchestrator | `client/src/pipeline/capture_pipeline.cpp` |
| WebRTC server | `server/webrtc_server.py` |
| Qwen VLM engine | `server/vlm_engine.py` |
| Prompt CLI | `prompt_app/app.py` |

## 동작 모드 요약

| 모드 | 트리거 | FPS | 게이팅 |
| ---- | ------ | --- | ------ |
| Idle   | 기본 | ≤ `aff.max_fps` | SCD ∧ AFF |
| Typing | `NotifyTyping(true)` | 1 | 없음 (강제) |
| Query  | `SubmitQuery` | 1 | 없음 (응답 완료까지) |
