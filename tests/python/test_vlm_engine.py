"""Unit tests for ``server.vlm_engine``.

These exercise the pure helpers (`_chunk_text`, `_sample_frames`) and the
stub-mode streaming path that does not require torch or a GPU.
"""

from __future__ import annotations

import asyncio
from typing import List

import numpy as np
import pytest

from server.vlm_engine import VlmConfig, VlmEngine


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------
class TestChunkText:
    def test_chunks_at_default_size(self):
        text = "a" * 130
        out = VlmEngine._chunk_text(text)
        assert out == ["a" * 64, "a" * 64, "a" * 2]

    def test_custom_chunk_size(self):
        out = VlmEngine._chunk_text("abcdef", chunk_size=2)
        assert out == ["ab", "cd", "ef"]

    def test_empty_string_yields_single_empty_chunk(self):
        # The pipeline always emits at least one chunk so the receiver
        # observes a stream rather than nothing.
        assert VlmEngine._chunk_text("") == [""]

    def test_text_shorter_than_chunk(self):
        assert VlmEngine._chunk_text("hi") == ["hi"]


# ---------------------------------------------------------------------------
# _sample_frames
# ---------------------------------------------------------------------------
def _frame(value: int) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


class TestSampleFrames:
    def test_empty_returns_empty(self):
        assert VlmEngine._sample_frames([], max_frames=4) == []

    def test_returns_input_when_under_cap(self):
        frames = [_frame(i) for i in range(3)]
        out = VlmEngine._sample_frames(frames, max_frames=4)
        assert out is frames  # function should not copy when no sampling needed

    def test_uniform_sampling_includes_endpoints(self):
        frames = [_frame(i) for i in range(10)]
        out = VlmEngine._sample_frames(frames, max_frames=4)
        assert len(out) == 4
        # Endpoints are kept (first and last frame).
        assert int(out[0][0, 0, 0]) == 0
        assert int(out[-1][0, 0, 0]) == 9

    def test_sampling_is_monotonic(self):
        frames = [_frame(i) for i in range(20)]
        out = VlmEngine._sample_frames(frames, max_frames=5)
        values = [int(f[0, 0, 0]) for f in out]
        assert values == sorted(values)
        assert len(set(values)) == len(values)


# ---------------------------------------------------------------------------
# VlmConfig env defaults
# ---------------------------------------------------------------------------
class TestVlmConfig:
    def test_defaults(self):
        cfg = VlmConfig()
        assert cfg.max_new_tokens == 256
        assert isinstance(cfg.stub, bool)

    def test_explicit_overrides(self):
        cfg = VlmConfig(model_id="x/y", device="cpu", stub=True, max_new_tokens=8)
        assert cfg.model_id == "x/y"
        assert cfg.device == "cpu"
        assert cfg.stub is True
        assert cfg.max_new_tokens == 8


# ---------------------------------------------------------------------------
# Stub generation path
# ---------------------------------------------------------------------------
async def _collect(agen):
    return [chunk async for chunk in agen]


class TestStubGenerate:
    @pytest.mark.asyncio
    async def test_stub_streams_chunks_without_loading_model(self):
        eng = VlmEngine(VlmConfig(stub=True))
        frames = [_frame(0), _frame(1)]
        chunks = await _collect(eng.generate("hello?", frames))

        assert chunks, "stub stream must emit at least one chunk"
        full = "".join(chunks)
        assert "[stub]" in full
        assert "hello?" in full
        # Resolution from the most recent frame is reported.
        assert "4x4" in full
        # Frame count is reported.
        assert "2 frame" in full
        # Model must not have been touched in stub mode.
        assert eng._model is None
        assert eng._processor is None

    @pytest.mark.asyncio
    async def test_stub_handles_empty_frame_list(self):
        eng = VlmEngine(VlmConfig(stub=True))
        chunks = await _collect(eng.generate("anything", []))
        full = "".join(chunks)
        assert "0 frame" in full
        assert "0x0" in full

    @pytest.mark.asyncio
    async def test_stub_chunks_are_bounded(self):
        # Each piece must be small (≤40 chars per the implementation), so
        # the receiver can render incremental progress.
        eng = VlmEngine(VlmConfig(stub=True))
        chunks = await _collect(eng.generate("q", [_frame(0)]))
        assert all(len(c) <= 40 for c in chunks)
