"""Entry point for the remote-vlm Python server."""

from __future__ import annotations

import argparse
import logging
import os

from aiohttp import web

from .vlm_engine import VlmConfig
from .webrtc_server import WebRtcServer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="remote-vlm server")
    p.add_argument("--host", default=os.environ.get("REMOTE_VLM_HOST", "0.0.0.0"))
    p.add_argument(
        "--port", type=int, default=int(os.environ.get("REMOTE_VLM_PORT", "8080"))
    )
    p.add_argument(
        "--model",
        default=os.environ.get("REMOTE_VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct"),
        help="HuggingFace model id (Qwen3-VL family)",
    )
    p.add_argument("--device", default=os.environ.get("REMOTE_VLM_DEVICE", "auto"))
    p.add_argument(
        "--stub",
        action="store_true",
        default=bool(int(os.environ.get("REMOTE_VLM_STUB", "0"))),
        help="Skip model load and emit deterministic responses (for E2E plumbing tests).",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = VlmConfig(model_id=args.model, device=args.device, stub=args.stub)
    server = WebRtcServer(cfg)
    web.run_app(server.app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
