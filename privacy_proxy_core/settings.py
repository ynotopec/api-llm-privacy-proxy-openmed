"""Application settings — shared across all privacy proxy variants."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _parse_bool(value: str, default: bool) -> bool:
    if value.lower() in ("1", "true", "yes", "on"):
        return True
    if value.lower() in ("0", "false", "no", "off"):
        return False
    return default


@dataclass
class Settings:
    """Shared configuration for the privacy proxy."""

    # ── server ────────────────────────────────────────────────────
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8088"))

    # ── auth ──────────────────────────────────────────────────────
    inbound_api_keys: list[str] = field(
        default_factory=lambda: [
            x.strip()
            for x in os.getenv("INBOUND_API_KEYS", "").split(",")
            if x.strip()
        ]
    )

    # ── upstream ──────────────────────────────────────────────────
    upstream_base_url: str = os.getenv("UPSTREAM_BASE_URL", "").rstrip("/")
    upstream_api_key: str = os.getenv("UPSTREAM_API_KEY", "")

    @property
    def llm_enabled(self) -> bool:
        """Whether the upstream LLM should be called."""
        return bool(self.upstream_base_url)

    # ── privacy model ─────────────────────────────────────────────
    privacy_model_id: str = os.getenv("PRIVACY_MODEL_ID", "")

    device: str = os.getenv("DEVICE", "auto")
    torch_dtype: str = os.getenv("TORCH_DTYPE", "auto")

    # ── redaction behaviour ───────────────────────────────────────
    filter_output: bool = _parse_bool(
        os.getenv("FILTER_OUTPUT", "true"), True
    )
    min_entity_score: float = float(os.getenv("MIN_ENTITY_SCORE", "0.50"))
    max_string_chars: int = int(os.getenv("MAX_STRING_CHARS", "200000"))
    model_idle_unload_seconds: int = int(
        os.getenv("MODEL_IDLE_UNLOAD_SECONDS", "300")
    )
    model_suffix: str = os.getenv("MODEL_SUFFIX", "-anonym")
    skip_json_keys: set[str] = field(
        default_factory=lambda: {
            x.strip()
            for x in os.getenv(
                "SKIP_JSON_KEYS",
                "model,role,type,stream,temperature,max_tokens,top_p,tools,"
                "tool_choice,name,thinking,reasoning,reasoning_effort",
            ).split(",")
            if x.strip()
        }
    )
    metrics_require_auth: bool = _parse_bool(
        os.getenv("METRICS_REQUIRE_AUTH", "true"), True
    )


settings = Settings()


def suffix_model_id(model_id: str) -> str:
    """Append the configured suffix unless already present."""
    if not model_id or not settings.model_suffix or model_id.endswith(settings.model_suffix):
        return model_id
    return f"{model_id}{settings.model_suffix}"


def unsuffix_model_id(model_id: str) -> str:
    """Remove the configured suffix."""
    suffix = settings.model_suffix
    if suffix and model_id.endswith(suffix) and len(model_id) > len(suffix):
        return model_id[: -len(suffix)]
    return model_id


def unsuffix_model_path(full_path: str) -> str:
    """Remove model suffix from path segments that start with 'models/'.

    E.g. ``models/gpt-4o-anonym`` → ``models/gpt-4o``.
    """
    parts = full_path.split("/")
    if len(parts) >= 2 and parts[0] == "models":
        parts[1] = unsuffix_model_id(parts[1])
    return "/".join(parts)
