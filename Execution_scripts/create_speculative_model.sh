#!/bin/sh
# Build seki-qwen-spec: Qwen2.5-3B target + Qwen2.5-0.5B draft (same tokenizer).
# Ollama DRAFT must be local GGUF files, not library tags.

set -e

OLLAMA_ROOT="${OLLAMA_MODELS:-/root/.ollama}"
TARGET_TAG="${OLLAMA_TARGET_MODEL:-qwen2.5:3b}"
DRAFT_TAG="${OLLAMA_DRAFT_MODEL:-qwen2.5:0.5b}"
SPEC_TAG="${OLLAMA_SPEC_MODEL:-seki-qwen-spec}"
NUM_CTX="${OLLAMA_NUM_CTX:-2048}"
DRAFT_K="${OLLAMA_DRAFT_NUM_PREDICT:-8}"
GGUF_DIR="${OLLAMA_ROOT}/gguf"
MANIFEST_ROOT="${OLLAMA_ROOT}/models/manifests/registry.ollama.ai/library"

alias_target() {
  echo "[spec] WARNING: $1; aliasing ${TARGET_TAG} as ${SPEC_TAG}"
  ollama cp "$TARGET_TAG" "$SPEC_TAG" 2>/dev/null || true
}

tag_to_manifest() {
  name=${1%%:*}
  tag=${1#*:}
  if [ "$name" = "$1" ]; then
    tag=latest
  fi
  printf '%s/%s/%s' "$MANIFEST_ROOT" "$name" "$tag"
}

blob_from_manifest() {
  manifest=$1
  if [ ! -f "$manifest" ]; then
    echo "[spec] missing manifest: $manifest" >&2
    return 1
  fi
  hex=$(awk '
    BEGIN { RS="{" }
    /vnd.ollama.image.model/ {
      if (match($0, /sha256:[a-f0-9]+/)) {
        print substr($0, RSTART + 7, RLENGTH - 7)
        exit
      }
    }
  ' "$manifest")
  if [ -z "$hex" ]; then
    echo "[spec] no model layer in $manifest" >&2
    return 1
  fi
  blob="${OLLAMA_ROOT}/models/blobs/sha256-${hex}"
  if [ ! -f "$blob" ]; then
    echo "[spec] blob missing: $blob" >&2
    return 1
  fi
  printf '%s' "$blob"
}

echo "[spec] target=${TARGET_TAG} draft=${DRAFT_TAG} -> ${SPEC_TAG} num_ctx=${NUM_CTX} k=${DRAFT_K}"

target_blob=$(blob_from_manifest "$(tag_to_manifest "$TARGET_TAG")") || {
  alias_target "cannot resolve target GGUF"
  exit 0
}
draft_blob=$(blob_from_manifest "$(tag_to_manifest "$DRAFT_TAG")") || {
  alias_target "cannot resolve draft GGUF"
  exit 0
}

mkdir -p "$GGUF_DIR"
cp -f "$target_blob" "${GGUF_DIR}/target.gguf"
cp -f "$draft_blob" "${GGUF_DIR}/draft.gguf"

modelfile="${GGUF_DIR}/Modelfile"
cat > "$modelfile" <<EOF
FROM ${GGUF_DIR}/target.gguf
DRAFT ${GGUF_DIR}/draft.gguf
PARAMETER num_ctx ${NUM_CTX}
PARAMETER draft_num_predict ${DRAFT_K}
EOF

echo "[spec] creating ${SPEC_TAG}"
set +e
ollama create "$SPEC_TAG" -f "$modelfile" > /tmp/ollama-create-spec.log 2>&1
create_rc=$?
set -e
cat /tmp/ollama-create-spec.log
if [ "$create_rc" -eq 0 ]; then
  echo "[spec] ${SPEC_TAG} ready (3B + 0.5B draft)"
else
  echo "[spec] ollama create failed (exit $create_rc)" >&2
  alias_target "ollama create with DRAFT failed"
  echo "[spec] FIX: ./scripts/up_llamacpp.sh (llama.cpp draft-simple). GGUF pair: ${GGUF_DIR}/" >&2
fi
