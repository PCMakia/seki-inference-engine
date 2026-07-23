"""Hybrid primary/fallback LLM router (NVIDIA NIM → local Ollama)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

logger = logging.getLogger("seki.llm_router")

PRIMARY_BASE_URL = "https://integrate.api.nvidia.com/v1"
PRIMARY_MODEL = "deepseek-ai/deepseek-v4-flash"
PRIMARY_TIMEOUT_S = 4.0

FALLBACK_BASE_URL = "http://localhost:11434/v1"
FALLBACK_MODEL = "qwen2.5:3b-instruct-q5_K_M"
FALLBACK_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model_used: str
    used_fallback: bool


class HybridLLMRouter:
    """Attempt NVIDIA NIM first; fall back to local Ollama on failure."""

    def __init__(self, nvidia_api_key: str) -> None:
        self._primary = AsyncOpenAI(
            api_key=nvidia_api_key or "unused",
            base_url=PRIMARY_BASE_URL,
            timeout=PRIMARY_TIMEOUT_S,
        )
        self._fallback = AsyncOpenAI(
            api_key="ollama",
            base_url=FALLBACK_BASE_URL,
            timeout=FALLBACK_TIMEOUT_S,
        )

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> GenerationResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            text = await self._complete(
                self._primary,
                PRIMARY_MODEL,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return GenerationResult(
                text=text,
                model_used=PRIMARY_MODEL,
                used_fallback=False,
            )
        except (APITimeoutError, APIConnectionError, APIStatusError, TimeoutError) as exc:
            if isinstance(exc, APIStatusError) and exc.status_code not in {429, 500, 502, 503}:
                # Non-retryable client errors (auth, bad request) still fall through
                # only for quota/rate-limit/network style failures per Phase 5 plan.
                if exc.status_code != 429 and exc.status_code < 500:
                    logger.warning("Primary LLM client error %s; trying fallback", exc.status_code)
            logger.warning("Primary LLM failed (%s); routing to Ollama fallback", exc)

        text = await self._complete(
            self._fallback,
            FALLBACK_MODEL,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return GenerationResult(
            text=text,
            model_used=FALLBACK_MODEL,
            used_fallback=True,
        )

    @staticmethod
    async def _complete(
        client: AsyncOpenAI,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0].message.content if response.choices else None
        if not choice or not choice.strip():
            raise RuntimeError(f"Empty completion from model {model}")
        return choice.strip()
