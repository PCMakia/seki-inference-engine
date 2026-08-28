"""Read Ollama registry manifests to find the GGUF blob on disk."""

from __future__ import annotations

from typing import Any


MODEL_LAYER = "application/vnd.ollama.image.model"


def blob_filename(digest: str) -> str:
    """Ollama stores ``sha256:abc`` as ``blobs/sha256-abc``."""
    if digest.startswith("sha256:"):
        return "sha256-" + digest[len("sha256:") :]
    if digest.startswith("sha256-"):
        return digest
    return "sha256-" + digest


def model_layer_digest(manifest: dict[str, Any]) -> str:
    for layer in manifest.get("layers") or []:
        if layer.get("mediaType") == MODEL_LAYER:
            digest = layer.get("digest")
            if digest:
                return str(digest)
    raise ValueError("manifest has no application/vnd.ollama.image.model layer")
