"""Prometheus metrics for the inference gateway.

HTTP volume and latency histograms come from prometheus-fastapi-instrumentator
(``http_request_duration_seconds`` — Prometheus computes p95 via
``histogram_quantile``). Inference-specific series:

- ``seki_inference_requests_total`` — completions/embeddings by backend
- ``seki_inference_duration_seconds`` — gateway wall time by backend (p95)
- ``seki_inference_failover_total`` — vLLM unavailable, traffic sent to Ollama
"""

from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

INFERENCE_REQUESTS = Counter(
    "seki_inference_requests_total",
    "Successful inference calls by operation and backend.",
    ("operation", "backend"),
)

INFERENCE_DURATION = Histogram(
    "seki_inference_duration_seconds",
    "Gateway wall time for one inference call (vLLM or Ollama after failover).",
    ("operation", "backend"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

FAILOVER_TRIGGERS = Counter(
    "seki_inference_failover_total",
    "Times vLLM was unreachable and the gateway failed over to Ollama.",
    ("operation",),
)

# One Instrumentator for the process so tests that create multiple apps do not
# re-register http_* time series on the default Prometheus registry.
_instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
    excluded_handlers=["/metrics"],
    inprogress_name="seki_http_requests_inprogress",
)


def _prime_labelsets() -> None:
    for operation in ("chat", "chat_stream", "embeddings"):
        FAILOVER_TRIGGERS.labels(operation=operation)
        for backend in ("vllm", "ollama"):
            INFERENCE_REQUESTS.labels(operation=operation, backend=backend)
            INFERENCE_DURATION.labels(operation=operation, backend=backend)


_prime_labelsets()


def record_inference(
    *,
    operation: str,
    backend: str,
    duration_seconds: float,
    failover: bool,
    success: bool,
) -> None:
    if failover:
        FAILOVER_TRIGGERS.labels(operation=operation).inc()
    if success:
        INFERENCE_REQUESTS.labels(operation=operation, backend=backend).inc()
        INFERENCE_DURATION.labels(operation=operation, backend=backend).observe(
            duration_seconds
        )


def setup_metrics(app: FastAPI) -> None:
    """Instrument HTTP metrics and expose GET /metrics (unauthenticated)."""
    _instrumentator.instrument(app).expose(
        app,
        endpoint="/metrics",
        include_in_schema=False,
        should_gzip=False,
    )
