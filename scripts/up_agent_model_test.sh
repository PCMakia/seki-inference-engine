#!/usr/bin/env bash
# Start standalone test-seki-* stack (ports 10000 / 10080 / 11114).
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

cmd=(docker compose -f docker-compose.agent-model.test.yml up -d)
[[ "$build" -eq 1 ]] && cmd+=(--build)
[[ "$recreate" -eq 1 ]] && cmd+=(--force-recreate)
cmd+=(ollama inference agent-core)

echo "Starting test stack (test-seki-* on ports 10000/10080/11114)..."
"${cmd[@]}"

cat <<EOF

OK — test endpoints:
  agent:     http://127.0.0.1:10080/agent/health
  inference: http://127.0.0.1:10000/ready
  ollama:    http://127.0.0.1:11114

Bench:
  python3 seki-inference-engine/scripts/bench_agent_models.py \\
    --expected-model qwen2.5:3b-instruct-q5_K_M --label base-qwen

Stop:
  docker compose -f docker-compose.agent-model.test.yml down
EOF
