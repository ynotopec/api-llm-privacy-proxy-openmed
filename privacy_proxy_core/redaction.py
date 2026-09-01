"""Redaction helpers — shared across all variants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RedactionStats:
    """Accumulates tokens/spans/labels for one request."""
    tokens: int = 0
    spans: int = 0
    labels: Dict[str, int] = field(default_factory=dict)

    def add(self, label: str, token_count: int) -> None:
        self.tokens += token_count
        self.spans += 1
        self.labels[label] = self.labels.get(label, 0) + 1


class RedactionContext:
    """Stable placeholders inside one request.

    Same (label, value) pair always maps to the same placeholder.
    """
    def __init__(self) -> None:
        self.by_value: Dict[Tuple[str, str], str] = {}
        self.next_index: Dict[str, int] = {}

    def placeholder(self, label: str, value: str) -> str:
        label = normalize_label(label)
        key = (label, value)
        if key in self.by_value:
            return self.by_value[key]
        self.next_index[label] = self.next_index.get(label, 0) + 1
        ph = f"[{label.upper()}_{self.next_index[label]}]"
        self.by_value[key] = ph
        return ph


def normalize_label(label: str) -> str:
    """Strip BIOES prefixes and lowercase."""
    return (
        (label or "private")
        .replace("B-", "")
        .replace("I-", "")
        .replace("E-", "")
        .replace("S-", "")
        .lower()
    )


async def sanitize_payload(
    payload: Any,
    ctx: RedactionContext,
    stats: RedactionStats,
    parent_key: Optional[str],
    settings: Any,
) -> Any:
    """Generic recursive JSON sanitizer — same for all variants.

    The caller is responsible for invoking the actual model
    inside this function.  Each variant overrides by subclassing
    ``PrivacySanitizerBase`` and calling ``sanitize_text``.

    Returns the sanitized value (may be the same object).
    """
    from .redaction import sanitize_text  # noqa: F811

    if parent_key in settings.skip_json_keys:
        return payload

    if isinstance(payload, str):
        return await sanitize_text(payload, ctx, stats, settings)

    if isinstance(payload, list):
        return [
            await sanitize_payload(v, ctx, stats, None, settings)
            for v in payload
        ]

    if isinstance(payload, dict):
        return {
            k: await sanitize_payload(v, ctx, stats, k, settings)
            for k, v in payload.items()
        }

    return payload


# --- placeholder base class that every variant must override ---

from typing import Any  # noqa: E402


class PrivacySanitizerBase:
    """Base class: each proxy variant provides its own implementation.

    Subclasses must implement:
      - ``ensure_loaded()`` — load the model if not loaded
      - ``sanitize_text(text, ctx, stats)`` → str — redact one string
      - ``count_tokens(text)`` → int — token counting (optional)
    """

    def __init__(self) -> None:
        self._last_used_at: float = 0.0

    # ── device helpers ────────────────────────────────────────────
    def _resolve_device(self, device_str: str, default: str = "auto") -> str:
        requested = device_str.strip().lower()
        if requested in ("", "auto"):
            try:
                import torch
                cuda_avail = bool(torch.cuda.is_available())
                return "cuda" if cuda_avail else "cpu"
            except Exception:
                return "cpu"
        return requested

    def _resolve_torch_dtype(self, dtype_str: str):
        import torch
        d = dtype_str.strip().lower()
        if d == "bf16":
            return torch.bfloat16
        if d == "fp16":
            return torch.float16
        if d == "fp32":
            return torch.float32
        return None  # let transformers pick default

    def _is_device_available(self, device_name: str) -> bool:
        try:
            import torch
            if device_name == "cuda":
                return torch.cuda.is_available()
            if device_name == "mps":
                return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except Exception:
            return False
        return device_name == "cpu"

    # ── idle unload ───────────────────────────────────────────────
    def _touch(self) -> None:
        import time
        self._last_used_at = time.monotonic()

    def unload_if_idle(self, settings: Any) -> None:
        """Unload model if idle beyond the configured timeout."""
        timeout = settings.model_idle_unload_seconds
        if timeout <= 0 or self._last_used_at == 0:
            return
        import time
        idle_for = time.monotonic() - self._last_used_at
        if idle_for < timeout:
            return
        self._on_idle_unload()
        self._last_used_at = 0.0

    def _on_idle_unload(self) -> None:
        """Called by ``unload_if_idle``.  Subclass frees GPU/CPU memory."""
        pass

    # ── public API (must be implemented by subclass) ─────────────
    async def ensure_loaded(self) -> None:
        """Load model if not already loaded.  Subclass MUST implement."""
        raise NotImplementedError

    async def sanitize_text(
        self, text: str, ctx: RedactionContext, stats: RedactionStats,
    ) -> str:
        """Redact PII from one string.  Subclass MUST implement."""
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        """Estimate token count for one string.  Default fallback."""
        return max(1, len(text.split())) if text else 0

    async def sanitize_payload(
        self, payload: Any, settings: Any,
    ) -> tuple[Any, RedactionStats]:
        """Public entry-point for the proxy route."""
        from .redaction import RedactionContext, RedactionStats  # noqa: F811

        ctx = RedactionContext()
        stats = RedactionStats()
        sanitized = await self._sanitize_any(
            payload, ctx, stats, None, settings,
        )
        return sanitized, stats

    async def _sanitize_any(
        self,
        value: Any,
        ctx: RedactionContext,
        stats: RedactionStats,
        parent_key: Optional[str],
        settings: Any,
    ) -> Any:
        if parent_key in settings.skip_json_keys:
            return value
        if isinstance(value, str):
            return await self.sanitize_text(value, ctx, stats)
        if isinstance(value, list):
            return [
                await self._sanitize_any(v, ctx, stats, None, settings)
                for v in value
            ]
        if isinstance(value, dict):
            return {
                k: await self._sanitize_any(v, ctx, stats, k, settings)
                for k, v in value.items()
            }
        return value
