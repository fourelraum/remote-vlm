"""On-device Qwen VLM wrapper.

Loads a Qwen2-VL / Qwen2.5-VL checkpoint via Hugging Face Transformers
and exposes a single ``generate`` coroutine that turns a list of frames
plus a user prompt into a streamed text response.

The heavy model load happens lazily on first call so that a developer
running ``python -m server.main --no-vlm`` can iterate on the WebRTC
plumbing without paying the GPU cost.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("vlm.engine")


@dataclass
class VlmConfig:
    model_id: str = os.environ.get("REMOTE_VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
    device: str = os.environ.get("REMOTE_VLM_DEVICE", "auto")
    max_new_tokens: int = 256
    # If True, the engine echoes a deterministic stub response instead
    # of invoking the model. Useful for end-to-end pipeline tests on
    # machines without a GPU.
    stub: bool = bool(int(os.environ.get("REMOTE_VLM_STUB", "0")))


class VlmEngine:
    def __init__(self, cfg: Optional[VlmConfig] = None):
        self.cfg = cfg or VlmConfig()
        self._model = None
        self._processor = None
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self.cfg.stub or self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            logger.info("loading VLM model: %s", self.cfg.model_id)
            # Imported lazily so the server can boot without torch in
            # stub mode.
            import torch  # noqa: WPS433
            from transformers import (  # noqa: WPS433
                AutoProcessor,
                Qwen2VLForConditionalGeneration,
            )

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            device_map = self.cfg.device if self.cfg.device != "auto" else "auto"

            self._processor = AutoProcessor.from_pretrained(self.cfg.model_id)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.cfg.model_id,
                torch_dtype=dtype,
                device_map=device_map,
            )
            self._model.eval()
            logger.info("VLM model ready (dtype=%s)", dtype)

    async def generate(
        self,
        prompt: str,
        frames: List[np.ndarray],
    ) -> AsyncIterator[str]:
        """Yield response chunks for ``prompt`` conditioned on ``frames``.

        ``frames`` is a list of HxWx3 uint8 BGR arrays (most recent
        last). The engine deduplicates / down-samples internally to a
        small number of representative images.
        """

        if self.cfg.stub:
            async for chunk in self._stub_stream(prompt, frames):
                yield chunk
            return

        await self._ensure_loaded()

        # Cap to a small set of representative frames to fit context.
        sampled = self._sample_frames(frames, max_frames=4)
        pil_images = [
            Image.fromarray(frame[..., ::-1].copy())  # BGR -> RGB
            for frame in sampled
        ]

        text = await asyncio.get_running_loop().run_in_executor(
            None, self._sync_generate, prompt, pil_images
        )

        # Stream pseudo-chunks so the client can render progress. A
        # real streaming implementation would hook a TextStreamer.
        for piece in self._chunk_text(text):
            yield piece

    def _sync_generate(self, prompt: str, images: List[Image.Image]) -> str:
        import torch  # noqa: WPS433

        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": img} for img in images]
                    + [{"type": "text", "text": prompt}]
                ),
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
            )
        trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True
        )[0].strip()

    @staticmethod
    def _sample_frames(frames: List[np.ndarray], max_frames: int) -> List[np.ndarray]:
        if not frames:
            return []
        if len(frames) <= max_frames:
            return frames
        idx = np.linspace(0, len(frames) - 1, num=max_frames).astype(int)
        return [frames[i] for i in idx]

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 64) -> List[str]:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    @staticmethod
    async def _stub_stream(prompt: str, frames: List[np.ndarray]) -> AsyncIterator[str]:
        n = len(frames)
        h, w = (frames[-1].shape[:2] if frames else (0, 0))
        reply = (
            f"[stub] received prompt={prompt!r} with {n} frame(s) "
            f"(last={w}x{h}). Wire up REMOTE_VLM_STUB=0 to invoke Qwen."
        )
        for piece in [reply[i : i + 40] for i in range(0, len(reply), 40)]:
            await asyncio.sleep(0.05)
            yield piece
