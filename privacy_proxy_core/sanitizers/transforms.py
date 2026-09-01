"""Transformers-based (TGC) privacy sanitizer — used by proxy and openmed."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from privacy_proxy_core.redaction import PrivacySanitizerBase, RedactionContext, RedactionStats

log = logging.getLogger("privacy-proxy.transforms")


class TGCPrivacySanitizer(PrivacySanitizerBase):
    """Token-level classification model sanitizer (openai/privacy-filter, OpenMed/...).

    Uses ``transformers`` pipeline with ``aggregation_strategy="simple"``.
    """

    def __init__(self, model_id: str, device: str = "auto",
                 torch_dtype_str: str = "auto",
                 trust_remote_code: bool = False) -> None:
        super().__init__()
        self.model_id = model_id
        self.device = device
        self.torch_dtype_str = torch_dtype_str
        self.trust_remote_code = trust_remote_code
        self.classifier: Any = None
        self.tokenizer: Any = None
        self._load_lock = asyncio.Lock()
        self._model_device: str = "unloaded"

    def _on_idle_unload(self) -> None:
        log.info("Unloading TGC model after idle timeout")
        self.classifier = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def ensure_loaded(self) -> None:
        self.unload_if_idle()  # type: ignore[call-arg]
        if self.classifier is not None:
            return
        async with self._load_lock:
            if self.classifier is not None:
                return
            device = self._resolve_device(self.device)
            log.info("Loading TGC model: %s on %s", self.model_id, device)

            dtype = self._resolve_torch_dtype(self.torch_dtype_str)

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=self.trust_remote_code
            )

            model_kwargs: Dict[str, Any] = {}
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype
            model_kwargs["device_map"] = device

            model = AutoModelForTokenClassification.from_pretrained(
                self.model_id, **model_kwargs
            )
            model.eval()

            self.tokenizer = tokenizer
            self.classifier = pipeline(
                task="token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
            )
            self._model_device = device
            self._touch()
            log.info("TGC model loaded on %s", device)

    async def sanitize_text(
        self, text: str, ctx: RedactionContext, stats: RedactionStats,
    ) -> str:
        max_chars = int(os.getenv("MAX_STRING_CHARS", "200000"))
        min_score = float(os.getenv("MIN_ENTITY_SCORE", "0.50"))

        if not text or len(text) > max_chars:
            return text

        await self.ensure_loaded()
        self._touch()

        try:
            entities = self.classifier(text)
        except Exception as exc:
            log.exception("TGC inference failed")
            raise RuntimeError(f"privacy_filter_failed: {exc}") from exc

        spans: List[Tuple[int, int, str]] = []
        for ent in entities:
            score = float(ent.get("score", 0.0))
            if score < min_score:
                continue
            start = ent.get("start")
            end = ent.get("end")
            if start is None or end is None:
                word = ent.get("word", "")
                if word:
                    idx = text.find(word)
                    if idx >= 0:
                        start, end = idx, idx + len(word)
            if start is None or end is None:
                continue
            start, end = int(start), int(end)
            if 0 <= start < end <= len(text):
                spans.append((start, end, ent.get("entity_group") or "private"))

        if not spans:
            return text

        return self._build_output(text, spans, ctx, stats)

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

    def count_tokens(self, text: str) -> int:
        if not text or self.tokenizer is None:
            return max(1, len(text.split())) if text else 0
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return max(1, len(text.split()))
