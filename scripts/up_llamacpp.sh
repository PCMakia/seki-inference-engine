#!/usr/bin/env bash
# llama.cpp draft-simple when Ollama DRAFT fails on CUDA GGUF.
#
#   ./scripts/up_llamacpp.sh
#   ./scripts/up_llamacpp.sh down

set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE_LLAMA=(docker compose -f docker-compose.yml -f docker-compose.llamacpp.yml)

if [ ! -f ./data/ollama/gguf/target.gguf ] || [ ! -f ./data/ollama/gguf/draft.gguf ]; then
  echo "Missing GGUF pair. Run once: docker compose up -d ollama" >&2
  exit 1
fi

if [ "${1:-}" = "down" ]; then
  "${COMPOSE_LLAMA[@]}" stop llama-spec 2>/dev/null || true
  "${COMPOSE_LLAMA[@]}" rm -f llama-spec 2>/dev/null || true
  docker compose up -d ollama inference
  echo "Back on Ollama-only. Gateway :9000, Ollama :9114"
  exit 0
fi

echo "Stopping GPU Ollama..."
docker compose stop ollama

echo "Starting llama.cpp draft-simple + gateway..."
"${COMPOSE_LLAMA[@]}" up -d llama-spec inference

echo "Wait for http://localhost:9000/ready..."
for _ in $(seq 1 60); do
  if curl -fsS http://localhost:9000/ready >/dev/null 2>&1; then
    echo "ready"
    break
  fi
  sleep 2
done

echo "llama.cpp: http://localhost:9081"
