"""OpenAI-compatible HTTP client used for both vLLM and Ollama."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.backends.exceptions import BackendHTTPError, BackendUnavailableError


class OpenAICompatBackend:
    """Thin async client for `/chat/completions`, `/embeddings`, and `/models`."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        timeout: float,
        api_key: str = "EMPTY",
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=2.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post_json("/chat/completions", payload)

    async def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post_json("/embeddings", payload)

    async def chat_completions_stream(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        request = self._client.build_request("POST", "/chat/completions", json=payload)
        response = await self._send(request, stream=True)
        if not response.is_success:
            try:
                await response.aread()
                self._raise_for_status(response)
            finally:
                await response.aclose()

        async def _iter() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                raise BackendUnavailableError(self.name, str(exc)) from exc
            finally:
                await response.aclose()

        return _iter()

    async def ping(self) -> bool:
        """Return True if the OpenAI-compatible `/models` endpoint responds."""
        try:
            response = await self._client.get("/models", timeout=httpx.Timeout(5.0, connect=2.0))
            return response.status_code < 500
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError):
            return False

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = self._client.build_request("POST", path, json=payload)
        response = await self._send(request, stream=False)
        self._raise_for_status(response)
        try:
            return response.json()
        except ValueError as exc:
            raise BackendHTTPError(response.status_code, {"error": {"message": response.text}}) from exc

    async def _send(self, request: httpx.Request, *, stream: bool) -> httpx.Response:
        try:
            return await self._client.send(request, stream=stream)
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            raise BackendUnavailableError(self.name, str(exc)) from exc

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        body: Any
        try:
            body = response.json()
        except ValueError:
            body = {
                "error": {
                    "message": response.text or f"upstream HTTP {response.status_code}",
                    "type": "upstream_error",
                    "code": str(response.status_code),
                }
            }
        raise BackendHTTPError(response.status_code, body)
