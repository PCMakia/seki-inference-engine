"""vLLM primary with local Ollama fallback on timeout or connection error."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, Protocol

from app.backends.exceptions import BackendUnavailableError
from app.config import Settings

logger = logging.getLogger("seki.inference.failover")

BACKEND_VLLM = "vllm"
BACKEND_OLLAMA = "ollama"


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

    async def chat_completions(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        primary_payload = self._with_model(payload, self.settings.model_name)
        try:
            result = await self.primary.chat_completions(primary_payload)
            return result, BACKEND_VLLM
        except BackendUnavailableError as exc:
            logger.warning("vLLM chat unavailable (%s); falling back to Ollama", exc)
            fallback_payload = self._with_model(
                payload, self.settings.ollama_model, force=True
            )
            result = await self.fallback.chat_completions(fallback_payload)
            return result, BACKEND_OLLAMA

    async def chat_completions_stream(
        self,
        payload: dict[str, Any],
    ) -> tuple[AsyncIterator[bytes], str]:
        primary_payload = self._with_model(payload, self.settings.model_name)
        try:
            stream = await self.primary.chat_completions_stream(primary_payload)
            return stream, BACKEND_VLLM
        except BackendUnavailableError as exc:
            logger.warning("vLLM chat stream unavailable (%s); falling back to Ollama", exc)
            fallback_payload = self._with_model(
                payload, self.settings.ollama_model, force=True
            )
            stream = await self.fallback.chat_completions_stream(fallback_payload)
            return stream, BACKEND_OLLAMA

    async def embeddings(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        primary_payload = self._with_model(payload, self.settings.embedding_model_name)
        try:
            result = await self.primary.embeddings(primary_payload)
            return result, BACKEND_VLLM
        except BackendUnavailableError as exc:
            logger.warning("vLLM embeddings unavailable (%s); falling back to Ollama", exc)
            fallback_payload = self._with_model(
                payload, self.settings.ollama_embedding_model, force=True
            )
            result = await self.fallback.embeddings(fallback_payload)
            return result, BACKEND_OLLAMA

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
