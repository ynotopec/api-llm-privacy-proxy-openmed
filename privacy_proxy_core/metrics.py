"""Prometheus-compatible metrics — shared across all variants."""

from __future__ import annotations

import asyncio
from typing import Dict


class GlobalMetrics:
    """Thread-safe Prometheus metrics for the privacy proxy."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.requests_total = 0
        self.filtered_requests_total = 0
        self.filtered_tokens_total = 0
        self.filtered_spans_total = 0
        self.filtered_by_label: Dict[str, int] = {}

    async def add(
        self,
        tokens: int,
        spans: int,
        labels: Dict[str, int],
        *,
        count_request: bool = True,
    ) -> None:
        async with self.lock:
            if count_request:
                self.requests_total += 1
            if tokens > 0 or spans > 0:
                self.filtered_requests_total += 1
            self.filtered_tokens_total += tokens
            self.filtered_spans_total += spans
            for key, value in labels.items():
                self.filtered_by_label[key] = (
                    self.filtered_by_label.get(key, 0) + value
                )

    async def prometheus(self) -> str:
        """Render current metrics in Prometheus exposition format."""
        async with self.lock:
            lines = [
                "# HELP privacy_proxy_requests_total Total proxied requests.",
                "# TYPE privacy_proxy_requests_total counter",
                f"privacy_proxy_requests_total {self.requests_total}",
                "# HELP privacy_proxy_filtered_requests_total "
                "Requests where at least one span was filtered.",
                "# TYPE privacy_proxy_filtered_requests_total counter",
                f"privacy_proxy_filtered_requests_total "
                f"{self.filtered_requests_total}",
                "# HELP privacy_proxy_filtered_tokens_total "
                "Estimated number of tokens filtered.",
                "# TYPE privacy_proxy_filtered_tokens_total counter",
                f"privacy_proxy_filtered_tokens_total "
                f"{self.filtered_tokens_total}",
                "# HELP privacy_proxy_filtered_spans_total "
                "Number of PII spans filtered.",
                "# TYPE privacy_proxy_filtered_spans_total counter",
                f"privacy_proxy_filtered_spans_total "
                f"{self.filtered_spans_total}",
            ]
            for label, count in sorted(self.filtered_by_label.items()):
                safe = label.replace('"', '\"')
                lines.append(
                    f"privacy_proxy_filtered_spans_by_label_total"
                    f'{{label="{safe}"}} {count}'
                )
            return "\n".join(lines) + "\n"


metrics = GlobalMetrics()
