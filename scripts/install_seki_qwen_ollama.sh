#!/usr/bin/env bash
# Register seki-qwen-3b in the Ollama volume used by docker-compose.
#
# WSL:
#   export GGUF_PATH="/mnt/d/AI_lab/Seki-Discord/agent-core/seki-qwen-3b_gguf/qwen2.5-3b-instruct.Q5_K_M.gguf"
#   export OLLAMA_HOST="http://127.0.0.1:11114"
#   bash scripts/install_seki_qwen_ollama.sh
#
# If you see $'\r': command not found (Windows CRLF), run once:
#   sed -i 's/\r$//' scripts/install_seki_qwen_ollama.sh

set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
default_gguf="/mnt/d/AI_lab/Seki-Discord/agent-core/seki-qwen-3b_gguf/qwen2.5-3b-instruct.Q5_K_M.gguf"
gguf="${GGUF_PATH:-$default_gguf}"
if [ ! -f "$gguf" ]; then
  gguf="${GGUF_PATH:-$root/agent-core/seki-qwen-3b/unsloth.Q5_K_M.gguf}"
fi

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
  echo "Set GGUF_PATH to your exported .gguf file." >&2
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

if [ -n "$ollama_container" ]; then
  docker cp "$work/seki-qwen-3b.Q5_K_M.gguf" "$ollama_container:/tmp/seki-qwen-3b.Q5_K_M.gguf"
  docker cp "$work/Modelfile" "$ollama_container:/tmp/Modelfile"
  docker exec -w /tmp "$ollama_container" ollama create seki-qwen-3b -f Modelfile
else
  (cd "$work" && ollama create seki-qwen-3b -f Modelfile)
fi

echo "OK — verify: curl -s $ollama_host/api/tags | grep seki-qwen"
