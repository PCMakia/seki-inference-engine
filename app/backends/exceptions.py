"""Exceptions raised by inference backends."""

from typing import Any


class BackendUnavailableError(Exception):
    """Timeout or connection failure — the only errors eligible for failover."""

    def __init__(self, backend: str, message: str) -> None:
        self.backend = backend
        super().__init__(f"{backend}: {message}" if message else backend)


class BackendHTTPError(Exception):
    """Non-retryable HTTP error from an upstream OpenAI-compatible server."""

    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"upstream HTTP {status_code}")
