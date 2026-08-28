#!/usr/bin/env bash
# llama.cpp target-only on 1660 Ti (production path for dev-optimize on 6 GB).
# Speculative draft-simple (llama-spec) is bench-only; fair A/B showed ~3x slower on this card.
# Requires ./data/ollama/gguf/target.gguf (draft.gguf only needed for bench profile).
#
#   ./scripts/up_llamacpp.sh
#   ./scripts/up_llamacpp.sh down   # back to Ollama-only

set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE_LLAMA=(docker compose -f docker-compose.yml -f docker-compose.llamacpp.yml)
LLAMA_IMAGE="${LLAMA_CPP_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"

ensure_llamacpp_image() {
  if docker image inspect "$LLAMA_IMAGE" >/dev/null 2>&1; then
    return 0
  fi
  echo "Pulling $LLAMA_IMAGE (first run can take several minutes)..."
  if docker pull "$LLAMA_IMAGE" 2>/tmp/seki-llama-pull.err; then
    return 0
  fi
  if grep -qi 'denied' /tmp/seki-llama-pull.err; then
    echo "GHCR pull denied (stale docker login). Clearing ghcr.io creds and retrying..." >&2
    docker logout ghcr.io >/dev/null 2>&1 || true
    docker pull "$LLAMA_IMAGE"
    return 0
  fi
  cat /tmp/seki-llama-pull.err >&2
  return 1
}

if [ ! -f ./data/ollama/gguf/target.gguf ]; then
  echo "Missing target.gguf. Run once: docker compose up -d ollama" >&2
  exit 1
fi

if [ "${1:-}" = "down" ]; then
  "${COMPOSE_LLAMA[@]}" --profile bench stop llama-spec 2>/dev/null || true
  "${COMPOSE_LLAMA[@]}" stop llama-target 2>/dev/null || true
  "${COMPOSE_LLAMA[@]}" rm -f llama-target llama-spec 2>/dev/null || true
  docker compose up -d ollama inference
  echo "Back on Ollama-only. Gateway :9000, Ollama :9114"
  exit 0
fi

echo "Stopping GPU Ollama (frees VRAM for llama-target)..."
docker compose stop ollama

echo "Starting llama.cpp target-only + gateway (production on 6 GB)..."
ensure_llamacpp_image
"${COMPOSE_LLAMA[@]}" --profile bench stop llama-spec 2>/dev/null || true
"${COMPOSE_LLAMA[@]}" up -d llama-target inference

echo "Wait for http://localhost:9000/ready (first load can take 1–2 min)..."
for _ in $(seq 1 60); do
  if curl -fsS http://localhost:9000/ready >/dev/null 2>&1; then
    echo "ready"
    break
  fi
  sleep 2
done

echo "Gateway -> llama-target (qwen2.5:3b). Bench spec experiment: N=20 ./scripts/bench_speculative.sh"
echo "llama.cpp target: http://localhost:9082"
echo "Embeddings still need Ollama — run ./scripts/up_llamacpp.sh down for full Ollama mesh."
