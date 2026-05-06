"""WebRTC server: terminates the C++ client's PeerConnection.

* Signaling is the standard aiortc demo pattern: the client POSTs an
  SDP offer to ``/offer`` and we reply with an answer.

* The client's ``frames`` data channel carries
  ``[seq:u32 LE][capture_ts_us:u64 LE][size:u32 LE][JPEG bytes]``
  packets. We decode them and maintain a per-session ring buffer of the
  most recent (capture_ts_us, frame) pairs.

* The ``control`` data channel carries JSON messages:

      {"type":"query", "session_id": "...", "text": "...",
       "num_frames_hint": N,
       "trigger_ts_us": <wall-clock us>,
       "pre_window_ms": N, "post_window_ms": N}
      {"type":"response", "session_id": "...", "text": "...",
       "chunk_index": k, "done": bool}

  When ``trigger_ts_us`` is non-zero the server waits until at least one
  frame timestamped at-or-after ``trigger_ts_us + post_window_ms`` has
  been received (or a timeout elapses) before sampling frames in
  ``[trigger_ts_us - pre_window_ms, trigger_ts_us + post_window_ms]`` and
  invoking the VLM. This gives "지금/방금" style queries access to both
  pre- and post-event frames, dashcam-style.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import struct
import time
import uuid
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

from .vlm_engine import VlmConfig, VlmEngine

logger = logging.getLogger("vlm.webrtc")

FRAME_BUFFER_SECONDS = 30
FRAME_BUFFER_FPS = 1
MAX_BUFFERED_FRAMES = FRAME_BUFFER_SECONDS * FRAME_BUFFER_FPS

# Default trigger window if the prompt app sends 0.
DEFAULT_PRE_WINDOW_MS = 3000
DEFAULT_POST_WINDOW_MS = 3000
# How long past the post-window we'll wait for a frame to arrive before
# giving up and going to inference with what we have.
POST_WAIT_OVERHEAD_MS = 2000

# Frame protocol:
#   [seq:u32 LE][capture_ts_us:u64 LE][size:u32 LE][JPEG bytes]
FRAME_HEADER_FMT = "<IQI"
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER_FMT)


class Session:
    """Per-PeerConnection state."""

    def __init__(self, pc: RTCPeerConnection, engine: VlmEngine):
        self.id = uuid.uuid4().hex[:8]
        self.pc = pc
        self.engine = engine
        # Each entry is (capture_ts_us, BGR ndarray).
        self.frames: Deque[Tuple[int, np.ndarray]] = collections.deque(
            maxlen=MAX_BUFFERED_FRAMES
        )
        self.control_channel = None  # set when the channel opens
        self.last_frame_seq: Optional[int] = None
        self.last_frame_at: float = 0.0
        # Set whenever a new frame is appended; queries waiting for a
        # post-window frame block on this.
        self._frame_event = asyncio.Event()

    # ------------------------------------------------------------------
    # frame channel
    # ------------------------------------------------------------------
    def on_frame_packet(self, payload: bytes) -> None:
        if len(payload) < FRAME_HEADER_SIZE:
            logger.warning("[%s] truncated frame packet", self.id)
            return
        seq, ts_us, size = struct.unpack(
            FRAME_HEADER_FMT, payload[:FRAME_HEADER_SIZE]
        )
        body = payload[FRAME_HEADER_SIZE : FRAME_HEADER_SIZE + size]
        if len(body) != size:
            logger.warning(
                "[%s] frame packet size mismatch seq=%d declared=%d got=%d",
                self.id,
                seq,
                size,
                len(body),
            )
            return
        arr = np.frombuffer(body, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning("[%s] failed to decode JPEG seq=%d", self.id, seq)
            return
        self.frames.append((ts_us, bgr))
        self.last_frame_seq = seq
        self.last_frame_at = time.time()
        self._frame_event.set()
        if seq % 10 == 0:
            logger.info(
                "[%s] frame seq=%d size=%d res=%dx%d buffer=%d",
                self.id,
                seq,
                size,
                bgr.shape[1],
                bgr.shape[0],
                len(self.frames),
            )

    # ------------------------------------------------------------------
    # control channel
    # ------------------------------------------------------------------
    async def on_control_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] bad control payload: %s", self.id, exc)
            return
        if msg.get("type") != "query":
            return

        session_id = msg.get("session_id", "")
        prompt = msg.get("text", "")
        hint = int(msg.get("num_frames_hint", 0))
        trigger_ts_us = int(msg.get("trigger_ts_us", 0) or 0)
        pre_window_ms = int(msg.get("pre_window_ms", 0) or 0) or DEFAULT_PRE_WINDOW_MS
        post_window_ms = (
            int(msg.get("post_window_ms", 0) or 0) or DEFAULT_POST_WINDOW_MS
        )
        logger.info(
            "[%s] query sid=%s text=%r hint=%d trigger=%d pre=%d post=%d buffered=%d",
            self.id,
            session_id,
            prompt,
            hint,
            trigger_ts_us,
            pre_window_ms,
            post_window_ms,
            len(self.frames),
        )

        if trigger_ts_us > 0:
            await self._await_post_window(
                trigger_ts_us=trigger_ts_us,
                post_window_ms=post_window_ms,
                max_wait_ms=post_window_ms + POST_WAIT_OVERHEAD_MS,
            )
            lo = trigger_ts_us - pre_window_ms * 1000
            hi = trigger_ts_us + post_window_ms * 1000
            snapshot = [f for ts, f in self.frames if lo <= ts <= hi]
        else:
            snapshot = [f for _, f in self.frames]

        if hint > 0 and hint < len(snapshot):
            snapshot = snapshot[-hint:]

        chunk_index = 0
        try:
            async for chunk in self.engine.generate(prompt, snapshot):
                self._send_response(session_id, chunk, chunk_index, done=False)
                chunk_index += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] VLM generation failed", self.id)
            self._send_response(
                session_id,
                f"[error] {exc}",
                chunk_index,
                done=True,
            )
            return
        self._send_response(session_id, "", chunk_index, done=True)

    async def _await_post_window(
        self,
        trigger_ts_us: int,
        post_window_ms: int,
        max_wait_ms: int,
    ) -> None:
        """Block until a frame timestamped >= trigger + post is in the
        buffer, or ``max_wait_ms`` elapses."""
        target_ts_us = trigger_ts_us + post_window_ms * 1000
        deadline = time.monotonic() + max_wait_ms / 1000.0
        while True:
            if self.frames and self.frames[-1][0] >= target_ts_us:
                return
            self._frame_event.clear()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                await asyncio.wait_for(self._frame_event.wait(), remaining)
            except asyncio.TimeoutError:
                return

    def _send_response(
        self, session_id: str, text: str, chunk_index: int, done: bool
    ) -> None:
        if self.control_channel is None:
            logger.warning("[%s] control channel not ready, dropping response", self.id)
            return
        self.control_channel.send(
            json.dumps(
                {
                    "type": "response",
                    "session_id": session_id,
                    "text": text,
                    "chunk_index": chunk_index,
                    "done": done,
                }
            )
        )


class WebRtcServer:
    def __init__(self, vlm_cfg: Optional[VlmConfig] = None):
        self.engine = VlmEngine(vlm_cfg)
        self._sessions: Dict[str, Session] = {}

    async def offer(self, request: web.Request) -> web.Response:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        session = Session(pc, self.engine)
        self._sessions[session.id] = session
        logger.info("[%s] new peer connection from %s", session.id, request.remote)

        @pc.on("connectionstatechange")
        async def on_state_change() -> None:
            logger.info("[%s] pc state -> %s", session.id, pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                self._sessions.pop(session.id, None)
                await pc.close()

        @pc.on("datachannel")
        def on_datachannel(channel) -> None:  # noqa: WPS430
            logger.info("[%s] datachannel '%s' open", session.id, channel.label)
            if channel.label == "frames":
                @channel.on("message")
                def on_message(msg) -> None:  # noqa: WPS430
                    if isinstance(msg, (bytes, bytearray)):
                        session.on_frame_packet(bytes(msg))
            elif channel.label == "control":
                session.control_channel = channel

                @channel.on("message")
                def on_message(msg) -> None:  # noqa: WPS430
                    if isinstance(msg, str):
                        asyncio.ensure_future(session.on_control_message(msg))

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response(
            {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            }
        )

    async def health(self, _: web.Request) -> web.Response:
        return web.json_response(
            {
                "ok": True,
                "sessions": len(self._sessions),
                "model": self.engine.cfg.model_id,
                "stub": self.engine.cfg.stub,
            }
        )

    async def shutdown(self, _: web.Application) -> None:
        await asyncio.gather(*(s.pc.close() for s in self._sessions.values()))
        self._sessions.clear()

    def app(self) -> web.Application:
        application = web.Application()
        application.router.add_post("/offer", self.offer)
        application.router.add_get("/health", self.health)
        application.on_shutdown.append(self.shutdown)
        return application
