"""Unit tests for ``server.webrtc_server.Session``.

The Session class glues data-channel parsing, frame buffering and VLM
dispatch. We exercise it with a stub VlmEngine and a fake control
channel — no real RTCPeerConnection is created.
"""

from __future__ import annotations

import asyncio
import collections
import json
import struct
from typing import AsyncIterator, List
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from server.vlm_engine import VlmConfig, VlmEngine
from server.webrtc_server import MAX_BUFFERED_FRAMES, Session


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _jpeg(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    return buf.tobytes()


def _packet(seq: int, jpeg_bytes: bytes, *, declared_size: int = None) -> bytes:
    size = len(jpeg_bytes) if declared_size is None else declared_size
    return struct.pack("<II", seq, size) + jpeg_bytes


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
        s.on_frame_packet(b"\x00\x01\x02")  # < 8 bytes
        assert len(s.frames) == 0
        assert s.last_frame_seq is None

    def test_size_mismatch_is_dropped(self):
        s = _make_session()
        body = _jpeg(_checkerboard())
        # Declared size larger than actual body.
        pkt = _packet(seq=1, jpeg_bytes=body, declared_size=len(body) + 100)
        s.on_frame_packet(pkt)
        assert len(s.frames) == 0
        assert s.last_frame_seq is None

    def test_bad_jpeg_is_dropped(self):
        s = _make_session()
        bogus = b"\xff\xd8\xff\xd9NOT_A_JPEG"  # has SOI/EOI but no real data
        s.on_frame_packet(_packet(seq=7, jpeg_bytes=bogus))
        # cv2.imdecode returns None -> session drops the frame.
        assert len(s.frames) == 0
        assert s.last_frame_seq is None

    def test_valid_packet_appends_frame(self):
        s = _make_session()
        img = _checkerboard()
        s.on_frame_packet(_packet(seq=42, jpeg_bytes=_jpeg(img)))

        assert len(s.frames) == 1
        decoded = s.frames[-1]
        assert decoded.shape == img.shape
        assert decoded.dtype == np.uint8
        assert s.last_frame_seq == 42
        assert s.last_frame_at > 0

    def test_buffer_capped_at_max(self):
        s = _make_session()
        body = _jpeg(_checkerboard(8, 8))
        for i in range(MAX_BUFFERED_FRAMES + 5):
            s.on_frame_packet(_packet(seq=i, jpeg_bytes=body))

        assert len(s.frames) == MAX_BUFFERED_FRAMES
        assert s.last_frame_seq == MAX_BUFFERED_FRAMES + 4


# ---------------------------------------------------------------------------
# on_control_message
# ---------------------------------------------------------------------------
class _FakeChannel:
    def __init__(self):
        self.sent: List[str] = []

    def send(self, payload: str) -> None:
        self.sent.append(payload)


class _ScriptedEngine:
    """Minimal VlmEngine stand-in that yields a predefined chunk list."""

    def __init__(self, chunks: List[str], *, raise_after: int = -1):
        self._chunks = chunks
        self._raise_after = raise_after
        self.last_prompt: str = ""
        self.last_frame_count: int = -1
        self.cfg = VlmConfig(stub=True)

    async def generate(self, prompt: str, frames) -> AsyncIterator[str]:
        self.last_prompt = prompt
        self.last_frame_count = len(frames)
        for i, chunk in enumerate(self._chunks):
            if self._raise_after >= 0 and i == self._raise_after:
                raise RuntimeError("kaboom")
            yield chunk


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
    async def test_query_streams_chunks_then_done(self):
        engine = _ScriptedEngine(["hello", " ", "world"])
        s = _make_session(engine=engine)
        s.control_channel = _FakeChannel()
        s.frames.extend([_checkerboard()] * 3)

        await s.on_control_message(
            json.dumps({"type": "query", "session_id": "abc", "text": "what?"})
        )

        assert engine.last_prompt == "what?"
        assert engine.last_frame_count == 3

        decoded = [json.loads(p) for p in s.control_channel.sent]
        # 3 streamed chunks + 1 terminal done message.
        assert len(decoded) == 4
        for i, msg in enumerate(decoded[:3]):
            assert msg == {
                "type": "response",
                "session_id": "abc",
                "text": ["hello", " ", "world"][i],
                "chunk_index": i,
                "done": False,
            }
        # Terminal message: empty text, done=True, chunk_index continues.
        assert decoded[-1]["done"] is True
        assert decoded[-1]["text"] == ""
        assert decoded[-1]["chunk_index"] == 3

    @pytest.mark.asyncio
    async def test_num_frames_hint_trims_snapshot(self):
        engine = _ScriptedEngine(["x"])
        s = _make_session(engine=engine)
        s.control_channel = _FakeChannel()
        for _ in range(10):
            s.frames.append(_checkerboard())

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
        assert engine.last_frame_count == 3

    @pytest.mark.asyncio
    async def test_num_frames_hint_zero_uses_full_buffer(self):
        engine = _ScriptedEngine(["x"])
        s = _make_session(engine=engine)
        s.control_channel = _FakeChannel()
        for _ in range(7):
            s.frames.append(_checkerboard())

        await s.on_control_message(
            json.dumps(
                {
                    "type": "query",
                    "session_id": "sid",
                    "text": "q",
                    "num_frames_hint": 0,
                }
            )
        )
        assert engine.last_frame_count == 7

    @pytest.mark.asyncio
    async def test_engine_exception_emits_terminal_error(self):
        engine = _ScriptedEngine(["ok-chunk", "boom"], raise_after=1)
        s = _make_session(engine=engine)
        s.control_channel = _FakeChannel()
        s.frames.append(_checkerboard())

        await s.on_control_message(
            json.dumps({"type": "query", "session_id": "sid", "text": "q"})
        )

        decoded = [json.loads(p) for p in s.control_channel.sent]
        # First chunk delivered, then a terminal error message.
        assert decoded[0]["text"] == "ok-chunk"
        assert decoded[0]["done"] is False
        assert decoded[-1]["done"] is True
        assert decoded[-1]["text"].startswith("[error]")
        assert "kaboom" in decoded[-1]["text"]

    @pytest.mark.asyncio
    async def test_send_response_without_channel_is_noop(self):
        # Engine yields chunks but the data channel never opened.
        engine = _ScriptedEngine(["hi"])
        s = _make_session(engine=engine)
        assert s.control_channel is None

        # Should complete cleanly (logs warnings) without raising.
        await s.on_control_message(
            json.dumps({"type": "query", "session_id": "x", "text": "q"})
        )
