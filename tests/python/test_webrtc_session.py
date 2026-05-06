"""Unit tests for ``server.webrtc_server.Session``.

The Session class glues data-channel parsing, frame buffering and VLM
dispatch. We exercise it with a stub VlmEngine and a fake control
channel — no real RTCPeerConnection is created.
"""

from __future__ import annotations

import asyncio
import json
import struct
from typing import AsyncIterator, List, Optional, Tuple
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from server.vlm_engine import VlmConfig, VlmEngine
from server.webrtc_server import (
    DEFAULT_POST_WINDOW_MS,
    DEFAULT_PRE_WINDOW_MS,
    FRAME_HEADER_FMT,
    FRAME_HEADER_SIZE,
    MAX_BUFFERED_FRAMES,
    POST_WAIT_OVERHEAD_MS,
    Session,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _jpeg(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    return buf.tobytes()


def _packet(
    seq: int,
    jpeg_bytes: bytes,
    *,
    ts_us: int = 0,
    declared_size: Optional[int] = None,
) -> bytes:
    size = len(jpeg_bytes) if declared_size is None else declared_size
    return struct.pack(FRAME_HEADER_FMT, seq, ts_us, size) + jpeg_bytes


def _make_session(engine: VlmEngine = None) -> Session:
    pc = MagicMock(name="RTCPeerConnection")
    eng = engine or VlmEngine(VlmConfig(stub=True))
    return Session(pc, eng)


def _checkerboard(h: int = 32, w: int = 32) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, ::2] = 255
    return img


# ---------------------------------------------------------------------------
# on_frame_packet
# ---------------------------------------------------------------------------
class TestOnFramePacket:
    def test_truncated_packet_is_dropped(self):
        s = _make_session()
        # Anything shorter than the 16-byte header is dropped.
        s.on_frame_packet(b"\x00" * (FRAME_HEADER_SIZE - 1))
        assert len(s.frames) == 0
        assert s.last_frame_seq is None

    def test_size_mismatch_is_dropped(self):
        s = _make_session()
        body = _jpeg(_checkerboard())
        pkt = _packet(seq=1, jpeg_bytes=body, declared_size=len(body) + 100)
        s.on_frame_packet(pkt)
        assert len(s.frames) == 0
        assert s.last_frame_seq is None

    def test_bad_jpeg_is_dropped(self):
        s = _make_session()
        bogus = b"\xff\xd8\xff\xd9NOT_A_JPEG"
        s.on_frame_packet(_packet(seq=7, jpeg_bytes=bogus))
        assert len(s.frames) == 0
        assert s.last_frame_seq is None

    def test_valid_packet_appends_frame_with_timestamp(self):
        s = _make_session()
        img = _checkerboard()
        s.on_frame_packet(_packet(seq=42, jpeg_bytes=_jpeg(img), ts_us=1_700_000_000_000_000))

        assert len(s.frames) == 1
        ts_us, decoded = s.frames[-1]
        assert ts_us == 1_700_000_000_000_000
        assert decoded.shape == img.shape
        assert decoded.dtype == np.uint8
        assert s.last_frame_seq == 42
        assert s.last_frame_at > 0

    def test_buffer_capped_at_max(self):
        s = _make_session()
        body = _jpeg(_checkerboard(8, 8))
        for i in range(MAX_BUFFERED_FRAMES + 5):
            s.on_frame_packet(_packet(seq=i, jpeg_bytes=body, ts_us=1000 * i))

        assert len(s.frames) == MAX_BUFFERED_FRAMES
        assert s.last_frame_seq == MAX_BUFFERED_FRAMES + 4
        # Oldest entries should have been dropped.
        first_ts = s.frames[0][0]
        assert first_ts == 1000 * 5

    def test_frame_event_set_after_append(self):
        s = _make_session()
        s.on_frame_packet(
            _packet(seq=1, jpeg_bytes=_jpeg(_checkerboard()), ts_us=42)
        )
        assert s._frame_event.is_set()


# ---------------------------------------------------------------------------
# on_control_message
# ---------------------------------------------------------------------------
class _FakeChannel:
    def __init__(self):
        self.sent: List[str] = []

    def send(self, payload: str) -> None:
        self.sent.append(payload)


class _ScriptedEngine:
    """Minimal VlmEngine stand-in that yields a predefined chunk list and
    records the frames it was handed."""

    def __init__(self, chunks: List[str], *, raise_after: int = -1):
        self._chunks = chunks
        self._raise_after = raise_after
        self.last_prompt: str = ""
        self.last_frames: List[np.ndarray] = []
        self.cfg = VlmConfig(stub=True)

    async def generate(self, prompt: str, frames) -> AsyncIterator[str]:
        self.last_prompt = prompt
        self.last_frames = list(frames)
        for i, chunk in enumerate(self._chunks):
            if self._raise_after >= 0 and i == self._raise_after:
                raise RuntimeError("kaboom")
            yield chunk


def _seed_frames(session: Session, ts_list: List[int]) -> None:
    """Push pre-decoded frames with explicit timestamps into the buffer
    without going through the data-channel parser."""
    for ts in ts_list:
        session.frames.append((ts, _checkerboard()))


class TestOnControlMessage:
    @pytest.mark.asyncio
    async def test_invalid_json_is_silently_dropped(self):
        s = _make_session()
        s.control_channel = _FakeChannel()
        await s.on_control_message("{not json")
        assert s.control_channel.sent == []

    @pytest.mark.asyncio
    async def test_non_query_type_ignored(self):
        s = _make_session()
        s.control_channel = _FakeChannel()
        await s.on_control_message(json.dumps({"type": "ping"}))
        assert s.control_channel.sent == []

    @pytest.mark.asyncio
    async def test_query_without_trigger_ts_uses_full_buffer(self):
        engine = _ScriptedEngine(["hello", " ", "world"])
        s = _make_session(engine=engine)
        s.control_channel = _FakeChannel()
        _seed_frames(s, [1_000_000, 2_000_000, 3_000_000])

        await s.on_control_message(
            json.dumps({"type": "query", "session_id": "abc", "text": "what?"})
        )

        assert engine.last_prompt == "what?"
        assert len(engine.last_frames) == 3

        decoded = [json.loads(p) for p in s.control_channel.sent]
        assert len(decoded) == 4  # 3 chunks + done
        for i, msg in enumerate(decoded[:3]):
            assert msg == {
                "type": "response",
                "session_id": "abc",
                "text": ["hello", " ", "world"][i],
                "chunk_index": i,
                "done": False,
            }
        assert decoded[-1]["done"] is True
        assert decoded[-1]["text"] == ""
        assert decoded[-1]["chunk_index"] == 3

    @pytest.mark.asyncio
    async def test_trigger_ts_filters_frames_to_window(self):
        engine = _ScriptedEngine(["x"])
        s = _make_session(engine=engine)
        s.control_channel = _FakeChannel()

        trigger = 10_000_000  # 10 s in microseconds
        # Frames at -5s, -2s, -0.5s, +1s, +2.5s, +6s relative to trigger.
        offsets_s = [-5, -2, -0.5, 1, 2.5, 6]
        timestamps = [trigger + int(off * 1_000_000) for off in offsets_s]
        _seed_frames(s, timestamps)

        await s.on_control_message(
            json.dumps(
                {
                    "type": "query",
                    "session_id": "sid",
                    "text": "q",
                    "trigger_ts_us": trigger,
                    "pre_window_ms": 3000,
                    "post_window_ms": 3000,
                }
            )
        )
        # Window = [trigger - 3s, trigger + 3s] -> keeps -2s, -0.5s, +1s, +2.5s.
        assert len(engine.last_frames) == 4

    @pytest.mark.asyncio
    async def test_post_wait_returns_when_late_frame_arrives(self):
        engine = _ScriptedEngine(["ok"])
        s = _make_session(engine=engine)
        s.control_channel = _FakeChannel()

        trigger = 5_000_000_000  # arbitrary
        # Pre-frame only. The window-filter will need a post frame too.
        _seed_frames(s, [trigger - 1_000_000])

        async def _push_late_frame() -> None:
            await asyncio.sleep(0.05)
            s.frames.append((trigger + 1_500_000, _checkerboard()))
            s._frame_event.set()

        late_task = asyncio.create_task(_push_late_frame())
        await s.on_control_message(
            json.dumps(
                {
                    "type": "query",
                    "session_id": "sid",
                    "text": "q",
                    "trigger_ts_us": trigger,
                    "pre_window_ms": 3000,
                    "post_window_ms": 1,  # tiny window, late frame easily clears it
                }
            )
        )
        await late_task

        # Both frames are inside [trigger - 3s, trigger + 1ms]?
        # The pre-frame is, the post-frame at +1.5s is outside the 1ms
        # post window. So we expect just the pre-frame.
        assert len(engine.last_frames) == 1

    @pytest.mark.asyncio
    async def test_post_wait_timeout_when_no_post_frame_arrives(self):
        # max_wait = post_window_ms + POST_WAIT_OVERHEAD_MS. With
        # post_window=0 the wait collapses to ~POST_WAIT_OVERHEAD_MS,
        # but using a tiny post_window keeps the test fast.
        engine = _ScriptedEngine(["x"])
        s = _make_session(engine=engine)
        s.control_channel = _FakeChannel()

        trigger = 1_000_000_000
        _seed_frames(s, [trigger - 500_000])

        # Patch POST_WAIT_OVERHEAD_MS via monkey-patch by passing tiny
        # windows: max wait = post_window_ms + overhead, so we shrink
        # both for a fast test.
        import server.webrtc_server as ws_mod

        orig_overhead = ws_mod.POST_WAIT_OVERHEAD_MS
        ws_mod.POST_WAIT_OVERHEAD_MS = 50  # 50 ms overhead for the test
        try:
            await s.on_control_message(
                json.dumps(
                    {
                        "type": "query",
                        "session_id": "sid",
                        "text": "q",
                        "trigger_ts_us": trigger,
                        "pre_window_ms": 3000,
                        "post_window_ms": 50,
                    }
                )
            )
        finally:
            ws_mod.POST_WAIT_OVERHEAD_MS = orig_overhead

        # No new frame arrived -> we time out and proceed with what we have.
        assert len(engine.last_frames) == 1

    @pytest.mark.asyncio
    async def test_zero_windows_fall_back_to_defaults(self):
        engine = _ScriptedEngine(["x"])
        s = _make_session(engine=engine)
        s.control_channel = _FakeChannel()

        trigger = 100_000_000
        # Frame just inside the default 3-second pre window, frame just outside.
        inside = trigger - (DEFAULT_PRE_WINDOW_MS - 100) * 1000
        outside = trigger - (DEFAULT_PRE_WINDOW_MS + 100) * 1000
        post_inside = trigger + (DEFAULT_POST_WINDOW_MS - 100) * 1000
        _seed_frames(s, [outside, inside, post_inside])

        await s.on_control_message(
            json.dumps(
                {
                    "type": "query",
                    "session_id": "sid",
                    "text": "q",
                    "trigger_ts_us": trigger,
                    "pre_window_ms": 0,
                    "post_window_ms": 0,
                }
            )
        )
        assert len(engine.last_frames) == 2

    @pytest.mark.asyncio
    async def test_num_frames_hint_trims_snapshot(self):
        engine = _ScriptedEngine(["x"])
        s = _make_session(engine=engine)
        s.control_channel = _FakeChannel()
        _seed_frames(s, list(range(1_000_000, 11_000_000, 1_000_000)))

        await s.on_control_message(
            json.dumps(
                {
                    "type": "query",
                    "session_id": "sid",
                    "text": "q",
                    "num_frames_hint": 3,
                }
            )
        )
        assert len(engine.last_frames) == 3

    @pytest.mark.asyncio
    async def test_engine_exception_emits_terminal_error(self):
        engine = _ScriptedEngine(["ok-chunk", "boom"], raise_after=1)
        s = _make_session(engine=engine)
        s.control_channel = _FakeChannel()
        _seed_frames(s, [1_000_000])

        await s.on_control_message(
            json.dumps({"type": "query", "session_id": "sid", "text": "q"})
        )

        decoded = [json.loads(p) for p in s.control_channel.sent]
        assert decoded[0]["text"] == "ok-chunk"
        assert decoded[0]["done"] is False
        assert decoded[-1]["done"] is True
        assert decoded[-1]["text"].startswith("[error]")
        assert "kaboom" in decoded[-1]["text"]

    @pytest.mark.asyncio
    async def test_send_response_without_channel_is_noop(self):
        engine = _ScriptedEngine(["hi"])
        s = _make_session(engine=engine)
        assert s.control_channel is None
        # Should complete cleanly (logs warnings) without raising.
        await s.on_control_message(
            json.dumps({"type": "query", "session_id": "x", "text": "q"})
        )
