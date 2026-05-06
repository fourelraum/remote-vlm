"""WebRTC server: terminates the C++ client's PeerConnection.

* Signaling is the standard aiortc demo pattern: the client POSTs an
  SDP offer to ``/offer`` and we reply with an answer.

* The client's ``frames`` data channel carries
  ``[seq:u32][size:u32][jpeg bytes]`` packets. We decode them and
  maintain a per-session ring buffer of the most recent frames.

* The ``control`` data channel carries JSON messages:

      {"type":"query", "session_id": "...", "text": "...",
       "num_frames_hint": N}
      {"type":"response", "session_id": "...", "text": "...",
       "chunk_index": k, "done": bool}
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import struct
import time
import uuid
from typing import Deque, Dict, Optional

import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

from .vlm_engine import VlmConfig, VlmEngine

logger = logging.getLogger("vlm.webrtc")

FRAME_BUFFER_SECONDS = 30
FRAME_BUFFER_FPS = 1
MAX_BUFFERED_FRAMES = FRAME_BUFFER_SECONDS * FRAME_BUFFER_FPS  # 30 most-recent 1 FPS frames


class Session:
    """Per-PeerConnection state."""

    def __init__(self, pc: RTCPeerConnection, engine: VlmEngine):
        self.id = uuid.uuid4().hex[:8]
        self.pc = pc
        self.engine = engine
        self.frames: Deque[np.ndarray] = collections.deque(maxlen=MAX_BUFFERED_FRAMES)
        self.control_channel = None  # set when the channel opens
        self.last_frame_seq: Optional[int] = None
        self.last_frame_at: float = 0.0

    # ------------------------------------------------------------------
    # frame channel
    # ------------------------------------------------------------------
    def on_frame_packet(self, payload: bytes) -> None:
        if len(payload) < 8:
            logger.warning("[%s] truncated frame packet", self.id)
            return
        seq, size = struct.unpack("<II", payload[:8])
        body = payload[8 : 8 + size]
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
        self.frames.append(bgr)
        self.last_frame_seq = seq
        self.last_frame_at = time.time()
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
        logger.info(
            "[%s] query sid=%s text=%r hint=%d buffered=%d",
            self.id,
            session_id,
            prompt,
            hint,
            len(self.frames),
        )

        # Snapshot frames at submission time (the client may keep
        # streaming while we generate).
        snapshot = list(self.frames)
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
