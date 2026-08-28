#!/usr/bin/env bash
# Start isolated test-seki-* stack for agent-model A/B (no discord-bot).
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

build=0
recreate=0
for arg in "$@"; do
  case "$arg" in
    --build) build=1 ;;
    --recreate) recreate=1 ;;
  esac
done

cmd=(docker compose -f docker-compose.yml -f docker-compose.agent-model.test.yml up -d)
[[ "$build" -eq 1 ]] && cmd+=(--build)
[[ "$recreate" -eq 1 ]] && cmd+=(--force-recreate)
cmd+=(ollama inference agent-core)

echo "Starting test stack (test-seki-* on ports 9100/9180/9124)..."
"${cmd[@]}"

cat <<EOF

OK — test endpoints:
  agent:     http://127.0.0.1:9180/agent/health
  inference: http://127.0.0.1:9100/ready
  ollama:    http://127.0.0.1:9124

Bench:
  python3 seki-inference-engine/scripts/bench_agent_models.py \\
    --agent http://127.0.0.1:9180 --ollama http://127.0.0.1:9124 \\
    --expected-model qwen2.5:3b-instruct-q5_K_M --label base-qwen

Stop:
  docker compose -f docker-compose.yml -f docker-compose.agent-model.test.yml down
EOF
