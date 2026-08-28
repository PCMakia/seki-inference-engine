"""Ollama-first router; optional second backend for tests and leftover dual-engine setups."""

from __future__ import annotations

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
    """Minimal interface implemented by Ollama, test fakes, and optional extras."""

    name: str

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def chat_completions_stream(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[bytes]: ...

    async def ping(self) -> bool: ...


class FailoverRouter:
    """Send traffic to the primary engine; fail over only when a second backend exists."""

    def __init__(
        self,
        primary: InferenceBackend,
        fallback: InferenceBackend | None,
        settings: Settings,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.settings = settings

    def _chat_model_for(self, backend_name: str) -> str:
        if backend_name == BACKEND_OLLAMA:
            return self.settings.ollama_model
        return self.settings.model_name

    def _embed_model_for(self, backend_name: str) -> str:
        if backend_name == BACKEND_OLLAMA:
            return self.settings.ollama_embedding_model
        return self.settings.embedding_model_name

    async def _with_failover(
        self,
        operation: str,
        primary_fn: Callable[[], Awaitable[T]],
        fallback_fn: Callable[[], Awaitable[T]] | None,
    ) -> tuple[T, str]:
        started = time.perf_counter()
        backend = self.primary.name
        failover = False
        success = False
        try:
            try:
                result = await primary_fn()
                backend = self.primary.name
            except BackendUnavailableError as exc:
                if self.fallback is None or fallback_fn is None:
                    raise
                failover = True
                backend = self.fallback.name
                logger.warning(
                    "%s %s unavailable (%s); falling back to %s",
                    self.primary.name,
                    operation,
                    exc,
                    self.fallback.name,
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
        primary_payload = self._with_model(
            payload,
            self._chat_model_for(self.primary.name),
            backend_name=self.primary.name,
        )
        fallback_fn = None
        if self.fallback is not None:
            fb = self.fallback
            fallback_fn = lambda: fb.chat_completions(
                self._with_model(
                    payload,
                    self._chat_model_for(fb.name),
                    force=True,
                    backend_name=fb.name,
                )
            )
        return await self._with_failover(
            "chat",
            lambda: self.primary.chat_completions(primary_payload),
            fallback_fn,
        )

    async def chat_completions_stream(
        self,
        payload: dict[str, Any],
    ) -> tuple[AsyncIterator[bytes], str]:
        primary_payload = self._with_model(
            payload,
            self._chat_model_for(self.primary.name),
            backend_name=self.primary.name,
        )
        fallback_fn = None
        if self.fallback is not None:
            fb = self.fallback
            fallback_fn = lambda: fb.chat_completions_stream(
                self._with_model(
                    payload,
                    self._chat_model_for(fb.name),
                    force=True,
                    backend_name=fb.name,
                )
            )
        return await self._with_failover(
            "chat_stream",
            lambda: self.primary.chat_completions_stream(primary_payload),
            fallback_fn,
        )

    async def embeddings(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        primary_payload = self._with_model(
            payload, self._embed_model_for(self.primary.name)
        )
        fallback_fn = None
        if self.fallback is not None:
            fb = self.fallback
            fallback_fn = lambda: fb.embeddings(
                self._with_model(
                    payload,
                    self._embed_model_for(fb.name),
                    force=True,
                )
            )
        return await self._with_failover(
            "embeddings",
            lambda: self.primary.embeddings(primary_payload),
            fallback_fn,
        )

    async def readiness(self) -> dict[str, Any]:
        primary_ok = await self.primary.ping()
        fallback_ok = False
        if self.fallback is not None:
            fallback_ok = await self.fallback.ping()
        by_name = {self.primary.name: primary_ok}
        if self.fallback is not None:
            by_name[self.fallback.name] = fallback_ok
        ready = primary_ok or fallback_ok
        reported_primary: str | None
        if primary_ok:
            reported_primary = self.primary.name
        elif fallback_ok and self.fallback is not None:
            reported_primary = self.fallback.name
        else:
            reported_primary = None
        return {
            "status": "ready" if ready else "not_ready",
            "vllm": bool(by_name.get(BACKEND_VLLM, False)),
            "ollama": bool(by_name.get(BACKEND_OLLAMA, False)),
            "primary": reported_primary,
            "configured_primary": self.primary.name,
        }

    def _with_model(
        self,
        payload: dict[str, Any],
        model: str,
        *,
        force: bool = False,
        backend_name: str | None = None,
    ) -> dict[str, Any]:
        forwarded = dict(payload)
        if force or not forwarded.get("model"):
            forwarded["model"] = model
        if backend_name == BACKEND_OLLAMA:
            opts = dict(forwarded["options"]) if isinstance(forwarded.get("options"), dict) else {}
            if self.settings.ollama_num_ctx > 0:
                opts.setdefault("num_ctx", self.settings.ollama_num_ctx)
            if self.settings.ollama_draft_num_predict > 0:
                opts.setdefault("draft_num_predict", self.settings.ollama_draft_num_predict)
            if opts:
                forwarded["options"] = opts
        return forwarded
