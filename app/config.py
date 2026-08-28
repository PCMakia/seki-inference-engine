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
    ollama_num_ctx: int = Field(
        default=2048,
        ge=0,
        description="Ollama num_ctx. 2048 keeps 3B+0.5B draft on 6 GB. 0 skips the option.",
    )
    ollama_draft_num_predict: int = Field(
        default=8,
        ge=0,
        description="Speculative draft tokens per step (Leviathan k). 0 disables the option.",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
