"""Environment-driven service configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from the process environment and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    model_name: str = Field(
        default="Qwen/Qwen2.5-3B-Instruct",
        description="Primary chat model served by vLLM (Hugging Face id or served name).",
    )
    embedding_model_name: str = Field(
        default="nomic-embed-text",
        description="Primary embedding model name forwarded to vLLM.",
    )
    api_key: str = Field(
        default="",
        description="Bearer token required for /v1 routes. Requests are rejected if empty.",
    )
    request_timeout: float = Field(
        default=120.0,
        ge=1.0,
        description="Upstream HTTP timeout in seconds for vLLM and Ollama.",
    )

    vllm_base_url: str = Field(
        default="http://localhost:8001/v1",
        description="OpenAI-compatible base URL for the primary vLLM server.",
    )
    vllm_api_key: str = Field(
        default="EMPTY",
        description="Token sent to vLLM (vLLM defaults to EMPTY when auth is disabled).",
    )

    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="OpenAI-compatible base URL for the Ollama fallback.",
    )
    ollama_model: str = Field(
        default="qwen2.5:3b",
        description="Chat model name used when failing over to Ollama.",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model name used when failing over to Ollama.",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
