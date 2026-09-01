"""GLiNER2-based privacy sanitizer — used by gliner2 proxy variant."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
from typing import Any, List, Tuple

import torch
from privacy_proxy_core.redaction import PrivacySanitizerBase, RedactionContext, RedactionStats

log = logging.getLogger("privacy-proxy.gliner2")


class GLiNER2Sanitizer(PrivacySanitizerBase):
    """Entity extraction via GLiNER2 (fastino/gliner2-...).

    Uses ``extract_entities`` with threshold + include_spans.
    """

    def __init__(self, model_id: str, device: str = "auto",
                 torch_dtype_str: str = "auto",
                 entity_types: List[str] | None = None,
                 min_score: float = 0.50) -> None:
        super().__init__()
        self.model_id = model_id
        self.device = device
        self.torch_dtype_str = torch_dtype_str
        self.entity_types = entity_types or []
        self.min_score = min_score
        self.model: Any = None
        self._model_device: str = "unloaded"
        self._cuda_available: bool | None = None
        self._load_lock = asyncio.Lock()
        self._unload_task: asyncio.Task[None] | None = None

    def _on_idle_unload(self) -> None:
        log.info("Unloading GLiNER2 model after idle timeout")
        self.model = None
        self._model_device = "unloaded"
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def start_idle_watcher(self, check_interval: int = 30) -> None:
        """Background task to periodically check idle timeout."""
        if self._unload_task is not None:
            return

        interval = max(1, min(check_interval,
                              int(os.getenv("MODEL_IDLE_UNLOAD_SECONDS", "300"))))

        async def _watch() -> None:
            try:
                while True:
                    await asyncio.sleep(interval)
                    async with self._load_lock:
                        self.unload_if_idle()  # type: ignore[call-arg]
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("GLiNER2 idle watcher stopped")

        self._unload_task = asyncio.create_task(_watch(), name="gliner2-idle")

    async def stop_idle_watcher(self) -> None:
        task = self._unload_task
        self._unload_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._on_idle_unload()

    def _move_to_device(self, device: str) -> None:
        if self.model is None:
            return
        try:
            if hasattr(self.model, "to"):
                self.model.to(device)
            else:
                inner = getattr(self.model, "model", None)
                if hasattr(inner, "to"):
                    inner.to(device)
            self._model_device = device
        except Exception:
            self._model_device = "unknown"
            log.warning("Cannot move GLiNER2 model to %s", device)

    async def ensure_loaded(self) -> None:
        self.unload_if_idle()  # type: ignore[call-arg]
        if self.model is not None:
            return
        async with self._load_lock:
            if self.model is not None:
                return
            device = self._resolve_device(self.device)
            log.info("Loading GLiNER2 model: %s on %s", self.model_id, device)

            from gliner2 import GLiNER2

            self.model = GLiNER2.from_pretrained(self.model_id)
            self._move_to_device(device)
            self._touch()
            log.info("GLiNER2 loaded on %s", device)

    async def sanitize_text(
        self, text: str, ctx: RedactionContext, stats: RedactionStats,
    ) -> str:
        max_chars = int(os.getenv("MAX_STRING_CHARS", "200000"))

        if not text or len(text) > max_chars:
            return text

        await self.ensure_loaded()
        self._touch()

        try:
            result = self.model.extract_entities(
                text,
                self.entity_types,
                threshold=self.min_score,
                include_confidence=True,
                include_spans=True,
            )
        except TypeError:
            # Older GLiNER2 API
            result = self.model.extract_entities(text, self.entity_types)
        except Exception as exc:
            log.exception("GLiNER2 inference failed")
            raise RuntimeError(f"privacy_filter_failed: {exc}") from exc

        spans = self._parse_result(text, result)
        if not spans:
            return text

        return self._build_output(text, spans, ctx, stats)

    def _parse_result(self, text: str, result: Any) -> List[Tuple[int, int, str]]:
        spans: List[Tuple[int, int, str]] = []
        if not isinstance(result, list):
            return spans

        for ent in result:
            if not isinstance(ent, dict):
                continue
            score = float(ent.get("score", ent.get("confidence", 1.0)) or 0.0)
            if score < self.min_score:
                continue
            start = ent.get("start")
            end = ent.get("end")
            value = ent.get("text") or ent.get("value") or ent.get("word")
            label = ent.get("label") or ent.get("entity_group") or "private"

            if not isinstance(start, int) or not isinstance(end, int):
                if value:
                    idx = text.find(str(value))
                    if idx >= 0:
                        start, end = idx, idx + len(str(value))
                    else:
                        continue
                else:
                    continue

            start, end = int(start), int(end)
            if 0 <= start < end <= len(text):
                spans.append((start, end, str(label)))

        return spans

    def _build_output(
        self, text: str, spans: List[Tuple[int, int, str]],
        ctx: RedactionContext, stats: RedactionStats,
    ) -> str:
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        merged: List[Tuple[int, int, str]] = []
        for span in spans:
            if not merged or span[0] >= merged[-1][1]:
                merged.append(span)
            elif span[1] > merged[-1][1]:
                merged[-1] = (merged[-1][0], span[1], merged[-1][2])

        parts: list[str] = []
        last = 0
        for start, end, label in merged:
            original = text[start:end]
            parts.append(text[last:start])
            parts.append(ctx.placeholder(label, original))
            last = end
            stats.add(label, self.count_tokens(original))
        parts.append(text[last:])
        return "".join(parts)
