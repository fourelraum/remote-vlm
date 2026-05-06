# Tests

## Python

```bash
pip install -r server/requirements.txt -r prompt_app/requirements.txt pytest pytest-asyncio
pytest -q
```

`tests/python/conftest.py` regenerates the gRPC stubs from
`proto/prompt_app.proto` if they are missing, so the suite is hermetic.

Coverage:

| File | Unit | What it pins |
| ---- | ---- | ------------ |
| `test_vlm_engine.py` | `VlmEngine._chunk_text`, `_sample_frames`, stub `generate` | chunking, frame sub-sampling, stub streaming bypasses model load |
| `test_webrtc_session.py` | `Session.on_frame_packet`, `on_control_message` | binary frame parsing edge cases, query routing, error paths |
| `test_typing_debouncer.py` | `prompt_app.app.TypingDebouncer` | start-of-typing once, idle-stop, shutdown semantics |

## C++

```bash
cmake -S tests/cpp -B tests/cpp/build
cmake --build tests/cpp/build -j
ctest --test-dir tests/cpp/build --output-on-failure
```

Requires `libopencv-dev`. GoogleTest is found via `find_package`; if
absent, it is fetched at configure time.

Coverage:

| File | Unit | What it pins |
| ---- | ---- | ------------ |
| `test_capture_mode.cpp` | `CaptureMode` enum + `ToString` | enum string mapping, `Frame` defaults |
| `test_adaptive_frame_filter.cpp` | `AdaptiveFrameFilter` | first-frame accept, similar/rate/blurry drops, `EvaluateForced` keeps baseline fresh, `Reset` |
| `test_scene_change_detector.cpp` | `SceneChangeDetector` | first-frame fires, identical follow-up suppressed, hue-axis jumps fire, baseline only updates on change, `Reset` |
