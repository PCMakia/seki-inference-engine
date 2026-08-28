#!/usr/bin/env bash
# llama.cpp target + draft-simple for fair bench (:9082 vs :9081).
#
#   ./scripts/up_llamacpp.sh
#   ./scripts/up_llamacpp.sh down

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

if [ ! -f ./data/ollama/gguf/target.gguf ] || [ ! -f ./data/ollama/gguf/draft.gguf ]; then
  echo "Missing GGUF pair. Run once: docker compose up -d ollama" >&2
  exit 1
fi

if [ "${1:-}" = "down" ]; then
  "${COMPOSE_LLAMA[@]}" stop llama-target llama-spec 2>/dev/null || true
  "${COMPOSE_LLAMA[@]}" rm -f llama-target llama-spec 2>/dev/null || true
  docker compose up -d ollama inference
  echo "Back on Ollama-only. Gateway :9000, Ollama :9114"
  exit 0
fi

echo "Stopping GPU Ollama..."
docker compose stop ollama

echo "Starting llama.cpp target + spec + gateway..."
ensure_llamacpp_image
"${COMPOSE_LLAMA[@]}" up -d llama-target llama-spec inference

echo "Wait for http://localhost:9000/ready..."
for _ in $(seq 1 60); do
  if curl -fsS http://localhost:9000/ready >/dev/null 2>&1; then
    echo "ready"
    break
  fi
  sleep 2
done

echo "Bench: N=20 ./scripts/bench_speculative.sh"
echo "target: http://localhost:9082  spec: http://localhost:9081"
