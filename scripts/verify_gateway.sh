#!/usr/bin/env bash
# Verify seki-inference-engine on localhost:9000 (compose host publish).
set -euo pipefail
base="${INFERENCE_URL:-http://localhost:9000}"
base="${base%/}"
base="${base%/v1}"
key="${INFERENCE_API_KEY:-${API_KEY:-change-me}}"

echo "== GET $base/health =="
curl -fsS "$base/health"
echo

echo "== GET $base/ready =="
ready=""
for _ in $(seq 1 15); do
  if ready=$(curl -fsS -w "\n%{http_code}" "$base/ready" 2>/dev/null); then
    break
  fi
  sleep 2
done
code=$(printf '%s' "$ready" | tail -n1)
body=$(printf '%s' "$ready" | sed '$d')
echo "HTTP $code"
echo "$body"
if [ "$code" != "200" ]; then
  echo "ready is not 200 (is Ollama up?)" >&2
  exit 1
fi

echo
echo "== POST $base/v1/chat/completions =="
hdr=$(mktemp)
trap 'rm -f "$hdr"' EXIT
curl -fsS --max-time 360 "$base/v1/chat/completions" \
  -H "Authorization: Bearer $key" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":16}' \
  -D "$hdr"
echo
backend=$(grep -i '^x-seki-backend:' "$hdr" | awk '{print $2}' | tr -d '\r')
echo "x-seki-backend: $backend"
if [ -z "$backend" ]; then
  echo "missing x-seki-backend header" >&2
  exit 1
fi
echo
echo "OK - completion succeeded via $backend"
