from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.background import BackgroundTask

from privacy_proxy_core.redaction import PrivacySanitizerBase, RedactionContext, RedactionStats
from privacy_proxy_core.metrics import GlobalMetrics, metrics
from privacy_proxy_core.settings import Settings, settings, suffix_model_id, unsuffix_model_id, unsuffix_model_path

# ── lazy-loaded sanitizer (OpenMed TGC) ─────────────────────────

class OpenMedSanitizer(PrivacySanitizerBase):
    _delegate = None

    async def ensure_loaded(self) -> None:
        from privacy_proxy_core.sanitizers.transforms import TGCPrivacySanitizer
        if self._delegate is None:
            dtype_str = os.getenv("TORCH_DTYPE", "auto")
            trust = os.getenv("TRUST_REMOTE_CODE", "true").lower() in ("1", "true", "yes", "on")
            self._delegate = TGCPrivacySanitizer(
                settings.privacy_model_id,
                device=settings.device,
                torch_dtype_str=dtype_str,
                trust_remote_code=trust,
            )
        self._delegate.unload_if_idle()
        await self._delegate.ensure_loaded()

    async def sanitize_text(
        self, text: str, ctx: RedactionContext, stats: RedactionStats,
    ) -> str:
        if self._delegate is None:
            return text
        return await self._delegate.sanitize_text(text, ctx, stats)

    def count_tokens(self, text: str) -> int:
        if self._delegate is not None:
            return self._delegate.count_tokens(text)
        return max(1, len(text.split())) if text else 0

    async def sanitize_payload(
        self, payload: Any, settings: Settings,
    ) -> tuple[Any, RedactionStats]:
        from privacy_proxy_core.redaction import RedactionContext, RedactionStats
        ctx = RedactionContext()
        stats = RedactionStats()
        sanitized = await self._sanitize_any(payload, ctx, stats, None, settings)
        return sanitized, stats

    async def _sanitize_any(
        self, value: Any, ctx: RedactionContext, stats: RedactionStats,
        parent_key: Optional[str], settings: Settings,
    ) -> Any:
        if parent_key in settings.skip_json_keys:
            return value
        if isinstance(value, str):
            return await self.sanitize_text(value, ctx, stats)
        if isinstance(value, list):
            return [await self._sanitize_any(v, ctx, stats, None, settings) for v in value]
        if isinstance(value, dict):
            return {k: await self._sanitize_any(v, ctx, stats, k, settings) for k, v in value.items()}
        return value


sanitizer = OpenMedSanitizer()

app = FastAPI(title="OpenMed Multilingual Privacy Filter Proxy", version="1.0.0")


def extract_bearer(req: Request) -> str:
    auth = req.headers.get("authorization", "")
    return auth.split(" ", 1)[1].strip() if auth.lower().startswith("bearer ") else ""


def require_auth(req: Request, *, metrics_auth: bool = False) -> None:
    if metrics_auth and not settings.metrics_require_auth:
        return
    if settings.inbound_api_keys and extract_bearer(req) not in settings.inbound_api_keys:
        raise HTTPException(status_code=401, detail="invalid_or_missing_api_token")


def build_upstream_headers(req: Request) -> Dict[str, str]:
    excluded = {"host", "content-length", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"}
    headers = {k: v for k, v in req.headers.items() if k.lower() not in excluded}
    if settings.upstream_api_key:
        headers["authorization"] = f"Bearer {settings.upstream_api_key}"
    headers["content-type"] = "application/json"
    return headers


def headers_for_modified_body(resp: Response) -> Dict[str, str]:
    return {k: v for k, v in resp.headers.items() if k.lower() not in {"content-length", "content-encoding", "transfer-encoding", "connection"}}


def response_models_endpoint(full_path: str) -> bool:
    return full_path == "models" or full_path.startswith("models/")


def add_response_model_suffixes(upstream_resp: Response, full_path: str) -> Response:
    if "application/json" not in upstream_resp.headers.get("content-type", ""):
        return upstream_resp
    try:
        payload = json.loads(upstream_resp.body)
    except Exception:
        return upstream_resp
    payload = rewrite_response_model_ids(payload, models_endpoint=response_models_endpoint(full_path))
    return JSONResponse(content=payload, status_code=upstream_resp.status_code, headers=headers_for_modified_body(upstream_resp))


def rewrite_request_model_ids(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    out = dict(value)
    model_id = out.get("model")
    if isinstance(model_id, str):
        model_id = model_id.strip()
        if not model_id or model_id == settings.model_suffix:
            raise HTTPException(status_code=400, detail="invalid_model_id")
        out["model"] = unsuffix_model_id(model_id)
    return out


def rewrite_response_model_ids(value: Any, *, models_endpoint: bool = False) -> Any:
    if isinstance(value, list):
        return [rewrite_response_model_ids(item, models_endpoint=models_endpoint) for item in value]
    if not isinstance(value, dict):
        return value
    out = dict(value)
    model_id = out.get("model")
    if isinstance(model_id, str):
        out["model"] = suffix_model_id(model_id)
    if models_endpoint:
        object_id = out.get("id")
        if isinstance(object_id, str) and out.get("object") == "model":
            out["id"] = suffix_model_id(object_id)
        data = out.get("data")
        if isinstance(data, list):
            out["data"] = [rewrite_response_model_ids(item, models_endpoint=True) for item in data]
    return out


async def forward_request(req: Request, full_path: str, sanitized_payload: Any, stream: bool = False) -> Response:
    if not settings.llm_enabled:
        raise HTTPException(status_code=503, detail="llm_upstream_disabled")
    url = f"{settings.upstream_base_url}/{unsuffix_model_path(full_path)}"
    timeout = httpx.Timeout(600.0, connect=30.0)
    if stream:
        client = httpx.AsyncClient(timeout=timeout)
        upstream_req = client.build_request(req.method, url, headers=build_upstream_headers(req), params=dict(req.query_params), json=sanitized_payload)
        upstream_stream = await client.send(upstream_req, stream=True)

        async def close_upstream() -> None:
            await upstream_stream.aclose()
            await client.aclose()

        headers = {k: v for k, v in upstream_stream.headers.items() if k.lower() not in {"content-length", "connection"}}
        return StreamingResponse(upstream_stream.aiter_bytes(), status_code=upstream_stream.status_code, headers=headers, media_type=upstream_stream.headers.get("content-type", "application/json"), background=BackgroundTask(close_upstream))
    async with httpx.AsyncClient(timeout=timeout) as client:
        upstream = await client.request(req.method, url, headers=build_upstream_headers(req), params=dict(req.query_params), json=sanitized_payload)
    headers = {k: v for k, v in upstream.headers.items() if k.lower() not in {"content-length", "content-encoding", "transfer-encoding", "connection"}}
    return Response(content=upstream.content, status_code=upstream.status_code, headers=headers, media_type=upstream.headers.get("content-type", "application/json"))


def attach_privacy_headers(response: JSONResponse, stats: RedactionStats, latency_ms: float) -> JSONResponse:
    response.headers["x-privacy-filtered-tokens"] = str(stats.tokens)
    response.headers["x-privacy-filtered-spans"] = str(stats.spans)
    response.headers["x-privacy-filter-latency-ms"] = str(round(latency_ms, 2))
    return response


def privacy_metadata(stats: RedactionStats) -> Dict[str, Any]:
    return {"filtered_tokens": stats.tokens, "filtered_spans": stats.spans, "filtered_by_label": stats.labels}


def extract_sanitized_text(payload: Any) -> str:
    if isinstance(payload, dict):
        messages = payload.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
        for key in ("input", "prompt", "content", "text"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False)


def privacy_only_response(sanitized_payload: Any, stats: RedactionStats, latency_ms: float) -> JSONResponse:
    response = JSONResponse(content={"object": "privacy.redaction", "llm_enabled": False, "data": sanitized_payload, "privacy": privacy_metadata(stats)}, status_code=200)
    return attach_privacy_headers(response, stats, latency_ms)


def openai_privacy_only_response(full_path: str, sanitized_payload: Any, stats: RedactionStats, latency_ms: float) -> JSONResponse:
    created = int(time.time())
    model = "privacy-redaction"
    if isinstance(sanitized_payload, dict) and isinstance(sanitized_payload.get("model"), str):
        model = suffix_model_id(sanitized_payload["model"])
    content = extract_sanitized_text(sanitized_payload)
    metadata = {"llm_enabled": False, "privacy": privacy_metadata(stats)}
    if full_path == "chat/completions":
        payload = {"id": f"chatcmpl-privacy-{created}", "object": "chat.completion", "created": created, "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}], **metadata}
    elif full_path == "completions":
        payload = {"id": f"cmpl-privacy-{created}", "object": "text_completion", "created": created, "model": model, "choices": [{"index": 0, "text": content, "finish_reason": "stop"}], **metadata}
    elif full_path == "responses":
        payload = {"id": f"resp-privacy-{created}", "object": "response", "created_at": created, "model": model, "output_text": content, "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": content}]}], **metadata}
    else:
        return privacy_only_response(sanitized_payload, stats, latency_ms)
    return attach_privacy_headers(JSONResponse(content=payload, status_code=200), stats, latency_ms)


@app.post("/redact")
@app.post("/sanitize")
async def redact_payload(req: Request) -> JSONResponse:
    require_auth(req)
    try:
        payload = await req.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="expected_json_body") from exc
    started_at = time.perf_counter()
    sanitized_payload, stats = await sanitizer.sanitize_payload(payload, settings)
    await metrics.add(stats.tokens, stats.spans, stats.labels)
    return privacy_only_response(sanitized_payload, stats, (time.perf_counter() - started_at) * 1000)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "model": settings.privacy_model_id, "llm_enabled": settings.llm_enabled, "upstream": settings.upstream_base_url or None, "filter_output": settings.filter_output}


@app.get("/metrics")
async def get_metrics(req: Request) -> PlainTextResponse:
    require_auth(req, metrics_auth=True)
    return PlainTextResponse(await metrics.prometheus(), media_type="text/plain")


@app.api_route("/v1/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy_openai(req: Request, full_path: str) -> Response:
    if req.method == "OPTIONS":
        return Response(status_code=204)
    require_auth(req)
    if req.method in ("GET", "DELETE"):
        if not settings.llm_enabled:
            raise HTTPException(status_code=503, detail="llm_upstream_disabled")
        return add_response_model_suffixes(await forward_request(req, full_path, None), full_path)
    try:
        payload = await req.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="expected_json_body") from exc
    started_at = time.perf_counter()
    sanitized_payload, in_stats = await sanitizer.sanitize_payload(payload, settings)
    sanitized_payload = rewrite_request_model_ids(sanitized_payload)
    await metrics.add(in_stats.tokens, in_stats.spans, in_stats.labels)
    if not settings.llm_enabled:
        return openai_privacy_only_response(full_path, sanitized_payload, in_stats, (time.perf_counter() - started_at) * 1000)
    if isinstance(payload, dict) and payload.get("stream") is True:
        return await forward_request(req, full_path, sanitized_payload, stream=True)
    upstream_resp = await forward_request(req, full_path, sanitized_payload)
    upstream_resp.headers["x-privacy-filtered-tokens"] = str(in_stats.tokens)
    upstream_resp.headers["x-privacy-filtered-spans"] = str(in_stats.spans)
    upstream_resp.headers["x-privacy-filter-latency-ms"] = str(round((time.perf_counter() - started_at) * 1000, 2))
    rewritten_resp = add_response_model_suffixes(upstream_resp, full_path)
    if "application/json" not in rewritten_resp.headers.get("content-type", ""):
        return rewritten_resp
    try:
        response_payload = json.loads(rewritten_resp.body)
    except Exception:
        return rewritten_resp
    out_stats = RedactionStats()
    if settings.filter_output:
        response_payload, out_stats = await sanitizer.sanitize_payload(response_payload, settings)
        await metrics.add(out_stats.tokens, out_stats.spans, out_stats.labels, count_request=False)
    final = JSONResponse(content=response_payload, status_code=rewritten_resp.status_code, headers=headers_for_modified_body(rewritten_resp))
    if settings.filter_output:
        final.headers["x-privacy-filtered-output-tokens"] = str(out_stats.tokens)
        final.headers["x-privacy-filtered-output-spans"] = str(out_stats.spans)
    return final


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.host, port=settings.port, log_level=os.getenv("LOG_LEVEL", "info"))
