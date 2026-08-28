"""Shared fixtures for the inference gateway tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.backends.failover import FailoverRouter
from app.config import Settings, get_settings
from app.main import create_app
from tests.fakes import FakeBackend


@pytest.fixture
def test_settings() -> Settings:
    get_settings.cache_clear()
    settings = Settings(
        _env_file=None,
        api_key="test-secret",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        embedding_model_name="primary-embed",
        request_timeout=5.0,
        ollama_model="qwen2.5:3b",
        ollama_embedding_model="nomic-embed-text",
        ollama_base_url="http://ollama.test/v1",
    )
    return settings


@pytest.fixture
def patched_settings(test_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setattr("app.config.get_settings", lambda: test_settings)
    monkeypatch.setattr("app.auth.get_settings", lambda: test_settings)
    monkeypatch.setattr("app.main.get_settings", lambda: test_settings)
    return test_settings


@pytest.fixture
def vllm_backend() -> FakeBackend:
    return FakeBackend("vllm")


@pytest.fixture
def ollama_backend() -> FakeBackend:
    return FakeBackend("ollama")


@pytest.fixture
def failover_router(
    vllm_backend: FakeBackend,
    ollama_backend: FakeBackend,
    test_settings: Settings,
) -> FailoverRouter:
    return FailoverRouter(vllm_backend, ollama_backend, test_settings)


@pytest.fixture
def client(
    patched_settings: Settings,
    failover_router: FailoverRouter,
) -> Iterator[TestClient]:
    application = create_app(inference_router=failover_router)
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-secret"}
