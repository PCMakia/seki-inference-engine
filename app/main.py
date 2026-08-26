"""Production FastAPI gateway: OpenAI-compatible chat and embeddings."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.backends.failover import FailoverRouter
from app.backends.openai_compat import OpenAICompatBackend
from app.config import get_settings
from app.routers import chat, embeddings, health

logger = logging.getLogger("seki.inference")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def create_app(*, inference_router: FailoverRouter | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        _configure_logging(settings.log_level)
        app.state.settings = settings

        if inference_router is not None:
            app.state.router = inference_router
            yield
            return

        primary = OpenAICompatBackend(
            name="vllm",
            base_url=settings.vllm_base_url,
            timeout=settings.request_timeout,
            api_key=settings.vllm_api_key,
        )
        fallback = OpenAICompatBackend(
            name="ollama",
            base_url=settings.ollama_base_url,
            timeout=settings.request_timeout,
            api_key="ollama",
        )
        app.state.router = FailoverRouter(primary, fallback, settings)
        logger.info(
            "gateway up model=%s vllm=%s ollama=%s timeout=%ss",
            settings.model_name,
            settings.vllm_base_url,
            settings.ollama_base_url,
            settings.request_timeout,
        )
        try:
            yield
        finally:
            await primary.aclose()
            await fallback.aclose()

    application = FastAPI(
        title="seki-inference-engine",
        description=(
            "OpenAI-compatible inference gateway. Primary backend is vLLM; "
            "requests fail over to local Ollama on timeout or connection error."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    application.include_router(health.router)
    application.include_router(chat.router)
    application.include_router(embeddings.router)
    return application


app = create_app()
