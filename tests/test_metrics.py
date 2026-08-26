"""GET /metrics — unauthenticated Prometheus scrape."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.backends.failover import FailoverRouter
from app.main import create_app
from app.metrics import FAILOVER_TRIGGERS, INFERENCE_REQUESTS
from tests.fakes import FakeBackend, unavailable


def _counter_value(counter, **labels: str) -> float:
    return counter.labels(**labels)._value.get()


def test_metrics_unauthenticated(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "seki_inference_requests_total" in body
    assert "seki_inference_duration_seconds" in body
    assert "seki_inference_failover_total" in body
    assert "http_request_duration_seconds" in body


def test_metrics_records_chat_volume(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    before = _counter_value(INFERENCE_REQUESTS, operation="chat", backend="vllm")
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers=auth_headers,
    )
    assert response.status_code == 200
    scrape = client.get("/metrics").text
    assert "seki_inference_duration_seconds" in scrape
    assert _counter_value(INFERENCE_REQUESTS, operation="chat", backend="vllm") == before + 1


def test_metrics_records_failover_trigger(
    patched_settings,
    ollama_backend: FakeBackend,
    auth_headers: dict[str, str],
) -> None:
    before = _counter_value(FAILOVER_TRIGGERS, operation="chat")
    primary = FakeBackend("vllm", chat_error=unavailable("vllm"))
    router = FailoverRouter(primary, ollama_backend, patched_settings)
    app = create_app(inference_router=router)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=auth_headers,
        )
        scrape = client.get("/metrics").text
    assert response.status_code == 200
    assert response.headers["x-seki-backend"] == "ollama"
    assert _counter_value(FAILOVER_TRIGGERS, operation="chat") == before + 1
    assert "seki_inference_failover_total" in scrape
