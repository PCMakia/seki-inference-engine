#!/usr/bin/env bash
# Register seki-qwen-3b in the Ollama volume used by docker-compose.
#
# Prerequisite: a Q5_K_M (or compatible) GGUF from training export, e.g.
#   agent-core/seki-qwen-3b/unsloth.Q5_K_M.gguf
# or copy from D:\AI_lab\Seki-Discord\agent-core after re-export.
#
#   chmod +x scripts/install_seki_qwen_ollama.sh
#   GGUF_PATH=/path/to/unsloth.Q5_K_M.gguf ./scripts/install_seki_qwen_ollama.sh
#
# Then set OLLAMA_MODEL=seki-qwen-3b and recreate inference + agent-core.

set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
gguf="${GGUF_PATH:-$root/agent-core/seki-qwen-3b/unsloth.Q5_K_M.gguf}"
modelfile_dir="${MODELFILE_DIR:-$root/agent-core/seki-qwen-3b_gguf}"
ollama_host="${OLLAMA_HOST:-http://127.0.0.1:${OLLAMA_HOST_PORT:-9114}}"
ollama_container=""
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx test-seki-ollama; then
  ollama_container=test-seki-ollama
  ollama_host="${OLLAMA_HOST:-http://127.0.0.1:11114}"
elif docker ps --format '{{.Names}}' 2>/dev/null | grep -qx seki-v2-ollama; then
  ollama_container=seki-v2-ollama
fi

if [ ! -f "$gguf" ]; then
  echo "ERROR: GGUF not found at: $gguf" >&2
  echo "Export from checkpoint (see docs/AGENT_MODEL_BENCHMARK.md § Export) or set GGUF_PATH." >&2
  exit 1
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cp "$gguf" "$work/seki-qwen-3b.Q5_K_M.gguf"
cat >"$work/Modelfile" <<EOF
FROM ./seki-qwen-3b.Q5_K_M.gguf
PARAMETER temperature 0.85
PARAMETER min_p 0.1
SYSTEM """You are Seki, a witty AI companion on Discord. Translate memory and reasoning chains into natural, in-character replies. Stay concise and honest about gaps."""
EOF

echo "Creating seki-qwen-3b from $gguf via $ollama_host ..."
curl -fsS "$ollama_host/api/tags" >/dev/null || {
  echo "Ollama not reachable at $ollama_host — start docker compose first." >&2
  exit 1
}

# ollama create must run where the daemon can read the GGUF; use docker exec when containerized.
if [ -n "$ollama_container" ]; then
  docker cp "$work/seki-qwen-3b.Q5_K_M.gguf" "$ollama_container:/tmp/seki-qwen-3b.Q5_K_M.gguf"
  docker cp "$work/Modelfile" "$ollama_container:/tmp/Modelfile"
  docker exec -w /tmp "$ollama_container" ollama create seki-qwen-3b -f Modelfile
else
  (cd "$work" && ollama create seki-qwen-3b -f Modelfile)
fi

echo "OK — verify: curl -s $ollama_host/api/tags | grep seki-qwen"
