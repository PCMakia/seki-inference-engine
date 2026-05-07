#!/bin/sh
# Start Ollama server, wait until the API responds, then pull models listed in
# OLLAMA_PULL_MODELS (comma-separated). Used by docker-compose for agent-ollama.
#
# Set in compose: OLLAMA_PULL_MODELS=${OLLAMA_MODEL:-phi3:mini}
# Optional extras: OLLAMA_PULL_MODELS=phi3:mini,llava:7b

set -e

ollama serve &
OLLAMA_PID=$!

echo "[ollama entrypoint] waiting for server..."
i=0
while [ "$i" -lt 120 ]; do
  if ollama list >/dev/null 2>&1; then
    echo "[ollama entrypoint] server ready"
    break
  fi
  sleep 1
  i=$((i + 1))
done

if ! ollama list >/dev/null 2>&1; then
  echo "[ollama entrypoint] ERROR: server did not become ready in time"
  kill "$OLLAMA_PID" 2>/dev/null || true
  exit 1
fi

if [ -n "${OLLAMA_PULL_MODELS:-}" ]; then
  OLD_IFS=$IFS
  IFS=','
  for m in $OLLAMA_PULL_MODELS; do
    m=$(echo "$m" | tr -d '[:space:]')
    [ -z "$m" ] && continue
    echo "[ollama entrypoint] pulling: $m"
    ollama pull "$m" || echo "[ollama entrypoint] WARNING: pull failed for $m (server keeps running)"
  done
  IFS=$OLD_IFS
else
  echo "[ollama entrypoint] OLLAMA_PULL_MODELS empty; skipping pull"
fi

wait "$OLLAMA_PID"
