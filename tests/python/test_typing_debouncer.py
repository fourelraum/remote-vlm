"""Unit tests for ``prompt_app.app.TypingDebouncer``.

The debouncer fires NotifyTyping(True) on the first keystroke after an
idle gap and NotifyTyping(False) once the user stops typing for
``idle_seconds``. We replace the gRPC stub with a thread-safe fake that
records every NotifyTyping invocation.
"""

from __future__ import annotations

import threading
import time
from typing import List, Tuple

import pytest

from prompt_app.app import TypingDebouncer

import prompt_app_pb2 as pb  # type: ignore


class _RecordingStub:
    """Captures NotifyTyping calls. Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls: List[Tuple[bool, int]] = []

    def NotifyTyping(self, state: pb.TypingState, timeout: float = 0):
        with self._lock:
            self.calls.append((state.is_typing, state.timestamp_ms))

        class _Ack:
            ok = True
            message = ""

        return _Ack()

    def snapshot(self) -> List[Tuple[bool, int]]:
        with self._lock:
            return list(self.calls)


def _wait_for(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class TestTypingDebouncer:
    def test_first_keystroke_notifies_true_exactly_once(self):
        stub = _RecordingStub()
        deb = TypingDebouncer(stub, idle_seconds=10.0)
        try:
            deb.keystroke()
            deb.keystroke()
            deb.keystroke()
            # Three keystrokes within the idle window -> single True.
            assert _wait_for(lambda: len(stub.snapshot()) >= 1)
            calls = stub.snapshot()
            assert calls == [(True, calls[0][1])]
        finally:
            deb.shutdown()

    def test_idle_timeout_emits_false(self):
        stub = _RecordingStub()
        deb = TypingDebouncer(stub, idle_seconds=0.15)
        try:
            deb.keystroke()
            assert _wait_for(
                lambda: len(stub.snapshot()) >= 2, timeout=2.0
            ), f"expected start+stop, got {stub.snapshot()}"
            calls = stub.snapshot()
            assert calls[0][0] is True
            assert calls[1][0] is False
            # Stop must arrive at least idle_seconds after start.
            assert calls[1][1] - calls[0][1] >= 100  # ms
        finally:
            deb.shutdown()

    def test_continued_typing_keeps_false_at_bay(self):
        stub = _RecordingStub()
        deb = TypingDebouncer(stub, idle_seconds=0.2)
        try:
            for _ in range(5):
                deb.keystroke()
                time.sleep(0.05)
            # We've kept the inactivity timer reset for ~250 ms total —
            # the "stopped typing" event should not have fired yet.
            assert all(c[0] is True for c in stub.snapshot())
            # Now stop typing and let the idle timer expire.
            assert _wait_for(
                lambda: any(c[0] is False for c in stub.snapshot()),
                timeout=2.0,
            )
        finally:
            deb.shutdown()

    def test_shutdown_emits_pending_false_when_typing(self):
        stub = _RecordingStub()
        deb = TypingDebouncer(stub, idle_seconds=10.0)
        deb.keystroke()
        # Wait for the True notification to be observed.
        assert _wait_for(lambda: len(stub.snapshot()) >= 1)
        deb.shutdown()
        calls = stub.snapshot()
        assert calls[0] == (True, calls[0][1])
        assert calls[-1][0] is False

    def test_shutdown_when_idle_emits_nothing_extra(self):
        stub = _RecordingStub()
        deb = TypingDebouncer(stub, idle_seconds=0.05)
        # Never type at all.
        deb.shutdown()
        assert stub.snapshot() == []

    def test_typing_start_us_unset_before_first_keystroke(self):
        stub = _RecordingStub()
        deb = TypingDebouncer(stub, idle_seconds=0.05)
        try:
            assert deb.last_typing_start_us() is None
        finally:
            deb.shutdown()

    def test_typing_start_us_records_first_keystroke_in_burst(self):
        stub = _RecordingStub()
        deb = TypingDebouncer(stub, idle_seconds=10.0)
        try:
            before = int(time.time() * 1_000_000)
            deb.keystroke()
            after = int(time.time() * 1_000_000)
            ts = deb.last_typing_start_us()
            assert ts is not None
            assert before <= ts <= after
            # Subsequent keystrokes within the same burst do not reset.
            time.sleep(0.01)
            deb.keystroke()
            assert deb.last_typing_start_us() == ts
        finally:
            deb.shutdown()

    def test_typing_start_us_persists_after_idle_stop(self):
        # The submit path reads last_typing_start_us *after* the user
        # has stopped typing (Enter pressed, typing-burst ended). The
        # value must still reflect the most recent burst.
        stub = _RecordingStub()
        deb = TypingDebouncer(stub, idle_seconds=0.1)
        try:
            deb.keystroke()
            ts = deb.last_typing_start_us()
            assert ts is not None
            # Wait until the idle stop fires.
            assert _wait_for(
                lambda: any(c[0] is False for c in stub.snapshot()),
                timeout=2.0,
            )
            assert deb.last_typing_start_us() == ts
        finally:
            deb.shutdown()

    def test_typing_start_us_advances_on_new_burst(self):
        stub = _RecordingStub()
        deb = TypingDebouncer(stub, idle_seconds=0.1)
        try:
            deb.keystroke()
            first = deb.last_typing_start_us()
            assert _wait_for(
                lambda: any(c[0] is False for c in stub.snapshot()),
                timeout=2.0,
            )
            time.sleep(0.05)
            deb.keystroke()
            second = deb.last_typing_start_us()
            assert second is not None and first is not None
            assert second > first
        finally:
            deb.shutdown()
