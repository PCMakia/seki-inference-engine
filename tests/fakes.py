"""Duck-typed backend fakes used by unit and API tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.backends.exceptions import BackendUnavailableError


class FakeBackend:
    """Stand-in for OpenAICompatBackend."""

    def __init__(
        self,
        name: str,
        *,
        chat_result: dict[str, Any] | None = None,
        embed_result: dict[str, Any] | None = None,
        chat_error: BaseException | None = None,
        embed_error: BaseException | None = None,
        stream_chunks: list[bytes] | None = None,
        ping_ok: bool = True,
    ) -> None:
        self.name = name
        self.chat_result = chat_result or {"id": f"chatcmpl-{name}", "object": "chat.completion"}
        self.embed_result = embed_result or {"object": "list", "data": []}
        self.chat_error = chat_error
        self.embed_error = embed_error
        self.stream_chunks = stream_chunks or [b'data: {"choices":[]}\n\n', b"data: [DONE]\n\n"]
        self.ping_ok = ping_ok
        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.chat_calls.append(payload)
        if self.chat_error:
            raise self.chat_error
        return self.chat_result

    async def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.embed_calls.append(payload)
        if self.embed_error:
            raise self.embed_error
        return self.embed_result

    async def chat_completions_stream(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        self.stream_calls.append(payload)
        if self.chat_error:
            raise self.chat_error

        async def _iter() -> AsyncIterator[bytes]:
            for chunk in self.stream_chunks:
                yield chunk

        return _iter()

    async def ping(self) -> bool:
        return self.ping_ok

    async def aclose(self) -> None:
        return None


def unavailable(backend: str = "vllm") -> BackendUnavailableError:
    return BackendUnavailableError(backend, f"{backend} connection timed out")
