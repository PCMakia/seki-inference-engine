"""Shared HTTP helpers."""

from typing import Any


def openai_error(*, message: str, err_type: str, code: str) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": err_type,
            "code": code,
        }
    }
