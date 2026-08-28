#!/usr/bin/env bash
# Live wall-clock sample against a running Ollama gateway (Ubuntu 1660 Ti).
# Does not simulate the Discord queue — times real POST /v1/chat/completions.
#
#   docker compose up -d   # seki-inference-engine Low_ends_6GB
#   chmod +x scripts/pressure_live.sh
#   ./scripts/pressure_live.sh
#   N=20 ./scripts/pressure_live.sh
#
# Paste mean_s into PRESSURE_TEST.md: if mean is ~12s instead of ~6.6s, halve the rates.

set -euo pipefail
base="${INFERENCE_URL:-http://localhost:9000}"
base="${base%/}"
base="${base%/v1}"
key="${INFERENCE_API_KEY:-${API_KEY:-change-me}}"
n="${N:-10}"
max_tokens="${MAX_TOKENS:-80}"

now_s() { python3 -c "import time; print(f'{time.perf_counter():.6f}')"; }

echo "== live pressure: $n serial completions against $base (max_tokens=$max_tokens) =="
curl -fsS "$base/ready" >/dev/null || {
  echo "gateway /ready failed — is Ollama up?" >&2
  exit 1
}

times=()
ok=0
fail=0
for i in $(seq 1 "$n"); do
  start=$(now_s)
  code=$(curl -sS -o /tmp/seki-live-body.json -w "%{http_code}" --max-time 360 "$base/v1/chat/completions" \
    -H "Authorization: Bearer $key" \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Reply in one short sentence: ping $i\"}],\"max_tokens\":$max_tokens}" \
    || true)
  end=$(now_s)
  dt=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.3f", e - s }')
  if [ "$code" = "200" ]; then
    ok=$((ok + 1))
    times+=("$dt")
    echo "  $i  ${dt}s  HTTP $code"
  else
    fail=$((fail + 1))
    echo "  $i  ${dt}s  HTTP ${code:-err}"
  fi
done

if [ "${#times[@]}" -eq 0 ]; then
  echo "no successful completions" >&2
  exit 1
fi

sorted=$(printf '%s\n' "${times[@]}" | sort -n)
n_ok=${#times[@]}
mean=$(printf '%s\n' "${times[@]}" | awk '{s+=$1} END {printf "%.2f", s/NR}')
p50=$(printf '%s\n' "$sorted" | awk -v n="$n_ok" 'NR==int((n-1)*0.50)+1 {print; exit}')
p95=$(printf '%s\n' "$sorted" | awk -v n="$n_ok" 'NR==int((n-1)*0.95)+1 {print; exit}')
min_s=$(printf '%s\n' "$sorted" | head -n1)
max_s=$(printf '%s\n' "$sorted" | tail -n1)
echo
echo "ok=$n_ok fail=$fail mean_s=$mean p50_s=$p50 p95_s=$p95 min_s=$min_s max_s=$max_s"
echo "If mean_s is about 6-8, the saved PRESSURE_TEST.md rates stand. If ~12, treat 8/min as 4/min."
echo "Short pings (max_tokens=80) on a warm model are often <2s. That is not a Discord-length turn."
echo "Concurrent batches: python3 scripts/pressure_live_concurrent.py"
