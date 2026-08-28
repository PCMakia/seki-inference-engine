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
from app.metrics import setup_metrics
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

        ollama = OpenAICompatBackend(
            name="ollama",
            base_url=settings.ollama_base_url,
            timeout=settings.request_timeout,
            api_key="ollama",
        )
        app.state.router = FailoverRouter(ollama, None, settings)
        logger.info(
            "gateway up ollama_model=%s ollama=%s timeout=%ss num_ctx=%s draft_k=%s",
            settings.ollama_model,
            settings.ollama_base_url,
            settings.request_timeout,
            settings.ollama_num_ctx,
            settings.ollama_draft_num_predict,
        )
        try:
            yield
        finally:
            await ollama.aclose()

    application = FastAPI(
        title="seki-inference-engine",
        description=(
            "OpenAI-compatible inference gateway for 6 GB Turing. "
            "dev-optimize: Qwen2.5-3B target + 0.5B draft (speculative decoding). "
            "vLLM is not started on this branch."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    application.include_router(health.router)
    application.include_router(chat.router)
    application.include_router(embeddings.router)
    setup_metrics(application)
    return application


app = create_app()
