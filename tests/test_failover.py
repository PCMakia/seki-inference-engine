"""Failover: vLLM first, Ollama only on timeout or connection error."""

from __future__ import annotations

import pytest

from app.backends.exceptions import BackendHTTPError, BackendUnavailableError
from app.backends.failover import BACKEND_OLLAMA, BACKEND_VLLM, FailoverRouter
from tests.fakes import FakeBackend, unavailable


CHAT_PAYLOAD = {
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "hello"}],
}


@pytest.mark.asyncio
async def test_chat_uses_vllm_when_healthy(
    failover_router: FailoverRouter,
    vllm_backend: FakeBackend,
    ollama_backend: FakeBackend,
) -> None:
    result, backend = await failover_router.chat_completions(CHAT_PAYLOAD)
    assert backend == BACKEND_VLLM
    assert result["id"] == "chatcmpl-vllm"
    assert ollama_backend.chat_calls == []
    assert vllm_backend.chat_calls[0]["model"] == "Qwen/Qwen2.5-3B-Instruct"


@pytest.mark.asyncio
async def test_chat_fills_model_name_when_omitted(
    failover_router: FailoverRouter,
    vllm_backend: FakeBackend,
) -> None:
    await failover_router.chat_completions(
        {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert vllm_backend.chat_calls[0]["model"] == "Qwen/Qwen2.5-3B-Instruct"


@pytest.mark.asyncio
async def test_chat_ollama_primary_maps_client_model_to_ollama_tag(
    test_settings,
    ollama_backend: FakeBackend,
) -> None:
    router = FailoverRouter(ollama_backend, None, test_settings)
    await router.chat_completions(CHAT_PAYLOAD)
    assert ollama_backend.chat_calls[0]["model"] == "qwen2.5:3b"


@pytest.mark.asyncio
async def test_chat_fails_over_on_timeout(test_settings, ollama_backend: FakeBackend) -> None:
    primary = FakeBackend("vllm", chat_error=unavailable("vllm"))
    router = FailoverRouter(primary, ollama_backend, test_settings)
    result, backend = await router.chat_completions(
        {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert backend == BACKEND_OLLAMA
    assert result["id"] == "chatcmpl-ollama"
    assert ollama_backend.chat_calls[0]["model"] == "qwen2.5:3b"


@pytest.mark.asyncio
async def test_chat_does_not_fail_over_on_http_error(
    test_settings,
    ollama_backend: FakeBackend,
) -> None:
    primary = FakeBackend(
        "vllm",
        chat_error=BackendHTTPError(400, {"error": {"message": "bad request"}}),
    )
    router = FailoverRouter(primary, ollama_backend, test_settings)
    with pytest.raises(BackendHTTPError) as exc:
        await router.chat_completions(CHAT_PAYLOAD)
    assert exc.value.status_code == 400
    assert ollama_backend.chat_calls == []


@pytest.mark.asyncio
async def test_embeddings_fail_over_on_connection_error(
    test_settings,
    ollama_backend: FakeBackend,
) -> None:
    primary = FakeBackend("vllm", embed_error=unavailable("vllm"))
    router = FailoverRouter(primary, ollama_backend, test_settings)
    result, backend = await router.embeddings({"input": "hello"})
    assert backend == BACKEND_OLLAMA
    assert ollama_backend.embed_calls[0]["model"] == "nomic-embed-text"
    assert result["object"] == "list"


@pytest.mark.asyncio
async def test_stream_fails_over_on_connect_error(
    test_settings,
    ollama_backend: FakeBackend,
) -> None:
    primary = FakeBackend(
        "vllm",
        chat_error=BackendUnavailableError("vllm", "connect refused"),
    )
    router = FailoverRouter(primary, ollama_backend, test_settings)
    stream, backend = await router.chat_completions_stream(CHAT_PAYLOAD)
    chunks = [chunk async for chunk in stream]
    assert backend == BACKEND_OLLAMA
    assert chunks[-1] == b"data: [DONE]\n\n"
    assert ollama_backend.stream_calls[0]["model"] == "qwen2.5:3b"


@pytest.mark.asyncio
async def test_readiness_503_when_both_down(test_settings) -> None:
    router = FailoverRouter(
        FakeBackend("vllm", ping_ok=False),
        FakeBackend("ollama", ping_ok=False),
        test_settings,
    )
    payload = await router.readiness()
    assert payload["status"] == "not_ready"
    assert payload["primary"] is None
