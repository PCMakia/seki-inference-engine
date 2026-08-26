"""vLLM primary with local Ollama fallback on timeout or connection error."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol, TypeVar

from app.backends.exceptions import BackendUnavailableError
from app.config import Settings
from app.metrics import record_inference

logger = logging.getLogger("seki.inference.failover")

BACKEND_VLLM = "vllm"
BACKEND_OLLAMA = "ollama"

T = TypeVar("T")


class InferenceBackend(Protocol):
    """Minimal interface implemented by vLLM, Ollama, and test fakes."""

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def chat_completions_stream(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[bytes]: ...

    async def ping(self) -> bool: ...


class FailoverRouter:
    """Send traffic to vLLM; route to Ollama only on timeout or connection error."""

    def __init__(
        self,
        primary: InferenceBackend,
        fallback: InferenceBackend,
        settings: Settings,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.settings = settings

    async def _with_failover(
        self,
        operation: str,
        primary_fn: Callable[[], Awaitable[T]],
        fallback_fn: Callable[[], Awaitable[T]],
    ) -> tuple[T, str]:
        started = time.perf_counter()
        backend = BACKEND_VLLM
        failover = False
        success = False
        try:
            try:
                result = await primary_fn()
                backend = BACKEND_VLLM
            except BackendUnavailableError as exc:
                failover = True
                backend = BACKEND_OLLAMA
                logger.warning(
                    "vLLM %s unavailable (%s); falling back to Ollama",
                    operation,
                    exc,
                )
                result = await fallback_fn()
            success = True
            return result, backend
        finally:
            record_inference(
                operation=operation,
                backend=backend,
                duration_seconds=time.perf_counter() - started,
                failover=failover,
                success=success,
            )

    async def chat_completions(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        primary_payload = self._with_model(payload, self.settings.model_name)
        return await self._with_failover(
            "chat",
            lambda: self.primary.chat_completions(primary_payload),
            lambda: self.fallback.chat_completions(
                self._with_model(payload, self.settings.ollama_model, force=True)
            ),
        )

    async def chat_completions_stream(
        self,
        payload: dict[str, Any],
    ) -> tuple[AsyncIterator[bytes], str]:
        primary_payload = self._with_model(payload, self.settings.model_name)
        return await self._with_failover(
            "chat_stream",
            lambda: self.primary.chat_completions_stream(primary_payload),
            lambda: self.fallback.chat_completions_stream(
                self._with_model(payload, self.settings.ollama_model, force=True)
            ),
        )

    async def embeddings(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        primary_payload = self._with_model(payload, self.settings.embedding_model_name)
        return await self._with_failover(
            "embeddings",
            lambda: self.primary.embeddings(primary_payload),
            lambda: self.fallback.embeddings(
                self._with_model(
                    payload, self.settings.ollama_embedding_model, force=True
                )
            ),
        )

    async def readiness(self) -> dict[str, Any]:
        vllm_ok, ollama_ok = await asyncio.gather(self.primary.ping(), self.fallback.ping())
        ready = vllm_ok or ollama_ok
        return {
            "status": "ready" if ready else "not_ready",
            "vllm": vllm_ok,
            "ollama": ollama_ok,
            "primary": BACKEND_VLLM if vllm_ok else (BACKEND_OLLAMA if ollama_ok else None),
        }

    @staticmethod
    def _with_model(
        payload: dict[str, Any],
        model: str,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        forwarded = dict(payload)
        if force or not forwarded.get("model"):
            forwarded["model"] = model
        return forwarded
