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
        default="qwen2.5:3b-instruct-q5_K_M",
        description="Unused when Ollama is the only engine; kept for dual-backend tests.",
    )
    embedding_model_name: str = Field(
        default="nomic-embed-text",
        description="Embedding model name (Ollama nomic-embed-text on this branch).",
    )
    api_key: str = Field(
        default="",
        description="Bearer token required for /v1 routes. Requests are rejected if empty.",
    )
    request_timeout: float = Field(
        default=120.0,
        ge=1.0,
        description="Upstream HTTP timeout in seconds for Ollama.",
    )

    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="OpenAI-compatible base URL for Ollama.",
    )
    ollama_model: str = Field(
        default="qwen2.5:3b-instruct-q5_K_M",
        description="Chat GGUF tag served by Ollama on a 6 GB card.",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model name forwarded to Ollama.",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
