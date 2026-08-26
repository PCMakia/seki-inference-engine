"""HTTP contract: probes, auth, chat, embeddings."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.backends.exceptions import BackendHTTPError, BackendUnavailableError
from app.backends.failover import FailoverRouter
from app.main import create_app
from tests.fakes import FakeBackend, unavailable


def test_health_unauthenticated(client: TestClient) -> None:
    """Probes must not 500 through the Prometheus middleware (FastAPI include_router)."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_when_vllm_up(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["vllm"] is True
    assert body["primary"] == "vllm"


def test_ready_503_when_both_down(patched_settings) -> None:
    router = FailoverRouter(
        FakeBackend("vllm", ping_ok=False),
        FakeBackend("ollama", ping_ok=False),
        patched_settings,
    )
    app = create_app(inference_router=router)
    with TestClient(app) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_chat_requires_bearer(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 401


def test_chat_rejects_wrong_key(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer wrong"},
    )
    assert response.status_code == 401


def test_chat_completions_ok(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.headers["x-seki-backend"] == "vllm"
    assert response.json()["object"] == "chat.completion"


def test_chat_stream(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert response.headers["x-seki-backend"] == "vllm"
    assert b"data: [DONE]" in response.content


def test_embeddings_ok(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        "/v1/embeddings",
        json={"input": "hello world"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.headers["x-seki-backend"] == "vllm"
    assert response.json()["object"] == "list"


def test_chat_failover_header(
    patched_settings,
    ollama_backend: FakeBackend,
    auth_headers: dict[str, str],
) -> None:
    primary = FakeBackend("vllm", chat_error=unavailable("vllm"))
    router = FailoverRouter(primary, ollama_backend, patched_settings)
    app = create_app(inference_router=router)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=auth_headers,
        )
    assert response.status_code == 200
    assert response.headers["x-seki-backend"] == "ollama"


def test_chat_upstream_http_error_passthrough(
    patched_settings,
    ollama_backend: FakeBackend,
    auth_headers: dict[str, str],
) -> None:
    primary = FakeBackend(
        "vllm",
        chat_error=BackendHTTPError(400, {"error": {"message": "context length"}}),
    )
    router = FailoverRouter(primary, ollama_backend, patched_settings)
    app = create_app(inference_router=router)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=auth_headers,
        )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "context length"
    assert ollama_backend.chat_calls == []


def test_both_backends_down_returns_502(
    patched_settings,
    auth_headers: dict[str, str],
) -> None:
    primary = FakeBackend("vllm", chat_error=unavailable("vllm"))
    fallback = FakeBackend("ollama", chat_error=BackendUnavailableError("ollama", "down"))
    router = FailoverRouter(primary, fallback, patched_settings)
    app = create_app(inference_router=router)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=auth_headers,
        )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "backend_unavailable"
