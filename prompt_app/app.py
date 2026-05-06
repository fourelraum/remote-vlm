"""Reference prompt application.

Talks to the C++ client over gRPC:
  * Sends ``NotifyTyping`` events as the user types so the client can
    switch to 1 FPS capture.
  * On <Enter>, calls ``SubmitQuery`` and renders the streamed response
    in place. The query carries a ``trigger_ts_us`` (typically the time
    the current typing burst started) so the server can sample frames
    captured around that exact moment, dashcam-style.

Run after generating the gRPC stubs (``scripts/gen_proto.sh``) which
populates ``prompt_app/generated/``.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

GEN_DIR = Path(__file__).resolve().parent / "generated"
if GEN_DIR.exists():
    sys.path.insert(0, str(GEN_DIR))

import grpc  # noqa: E402

try:
    import prompt_app_pb2 as pb  # type: ignore  # noqa: E402
    import prompt_app_pb2_grpc as pb_grpc  # type: ignore  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "gRPC stubs not found. Run scripts/gen_proto.sh first."
    ) from exc

from prompt_toolkit import PromptSession


# Default windows, in milliseconds, around the trigger ts. A 3-second
# pre/post window comfortably covers "지금 / 방금 / 아까" style queries
# without inflating first-token latency too much.
DEFAULT_PRE_WINDOW_MS = 3000
DEFAULT_POST_WINDOW_MS = 3000


def _now_us() -> int:
    return int(time.time() * 1_000_000)


class TypingDebouncer:
    """Notifies the client when typing starts/stops, with hysteresis.

    Also remembers when the most recent typing burst began so the
    submit code can anchor the VLM query to "the moment the user
    started typing".
    """

    def __init__(self, stub: pb_grpc.PromptAppStub, idle_seconds: float = 0.8):
        self._stub = stub
        self._idle_seconds = idle_seconds
        self._lock = threading.Lock()
        self._last_keystroke = 0.0
        self._typing = False
        self._typing_start_us: Optional[int] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def keystroke(self) -> None:
        now = time.monotonic()
        notify_start = False
        with self._lock:
            self._last_keystroke = now
            if not self._typing:
                self._typing = True
                self._typing_start_us = _now_us()
                notify_start = True
        if notify_start:
            self._send(True)

    def last_typing_start_us(self) -> Optional[int]:
        """Wall-clock microseconds of the most recent typing-burst start.

        Returns ``None`` if the user has never typed in this session.
        The value is *not* cleared when typing stops, so a call made
        immediately after <Enter> still reflects the current burst.
        """
        with self._lock:
            return self._typing_start_us

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        if self._typing:
            self._send(False)

    def _run(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.1)
            with self._lock:
                if (
                    self._typing
                    and (time.monotonic() - self._last_keystroke) >= self._idle_seconds
                ):
                    self._typing = False
                    notify = True
                else:
                    notify = False
            if notify:
                self._send(False)

    def _send(self, is_typing: bool) -> None:
        try:
            self._stub.NotifyTyping(
                pb.TypingState(
                    is_typing=is_typing,
                    timestamp_ms=int(time.time() * 1000),
                ),
                timeout=2.0,
            )
        except grpc.RpcError as exc:
            print(f"[typing notify failed] {exc.code().name}", file=sys.stderr)


def submit_query(
    stub: pb_grpc.PromptAppStub,
    text: str,
    trigger_ts_us: int,
    pre_window_ms: int = DEFAULT_PRE_WINDOW_MS,
    post_window_ms: int = DEFAULT_POST_WINDOW_MS,
) -> None:
    sid = uuid.uuid4().hex[:8]
    req = pb.Query(
        session_id=sid,
        text=text,
        num_frames_hint=0,
        trigger_ts_us=trigger_ts_us,
        pre_window_ms=pre_window_ms,
        post_window_ms=post_window_ms,
    )
    print(f"\n[assistant] ", end="", flush=True)
    try:
        for resp in stub.SubmitQuery(req, timeout=120.0):
            if resp.text:
                print(resp.text, end="", flush=True)
            if resp.done:
                break
    except grpc.RpcError as exc:
        print(f"\n[error] {exc.code().name}: {exc.details()}", file=sys.stderr)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="remote-vlm prompt app")
    parser.add_argument(
        "--client",
        default=os.environ.get("REMOTE_VLM_CLIENT_ADDR", "localhost:50051"),
        help="C++ client gRPC address",
    )
    args = parser.parse_args()

    channel = grpc.insecure_channel(args.client)
    stub = pb_grpc.PromptAppStub(channel)

    try:
        health = stub.HealthCheck(pb.Empty(), timeout=2.0)
        print(
            f"connected to client@{args.client}; "
            f"webrtc={health.webrtc_connected} mode={health.capture_mode}"
        )
    except grpc.RpcError as exc:
        print(f"could not reach client@{args.client}: {exc.code().name}", file=sys.stderr)
        sys.exit(1)

    debouncer = TypingDebouncer(stub)
    session: PromptSession = PromptSession()

    # Hook the input buffer's on_text_changed event so every keystroke
    # — including paste / backspace — bumps the debouncer.
    def _on_text_changed(_buffer) -> None:  # noqa: WPS430
        debouncer.keystroke()

    print("Type a question and press <Enter>. Ctrl-D / Ctrl-C to quit.")
    try:
        while True:
            session.default_buffer.on_text_changed += _on_text_changed
            try:
                line = session.prompt("you> ")
            except (EOFError, KeyboardInterrupt):
                break
            finally:
                session.default_buffer.on_text_changed -= _on_text_changed
            line = line.strip()
            if not line:
                continue
            trigger_ts_us = debouncer.last_typing_start_us() or _now_us()
            submit_query(stub, line, trigger_ts_us=trigger_ts_us)
    finally:
        debouncer.shutdown()
        channel.close()


if __name__ == "__main__":
    main()
