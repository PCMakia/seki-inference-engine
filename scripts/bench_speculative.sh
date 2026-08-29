#!/usr/bin/env bash
# Fair sequential A/B: target-only llama.cpp (:9082) vs draft-simple spec (:9081).
# Stops the other CUDA server before each arm so a 6 GB card is not double-loaded.
#
#   ./scripts/bench_speculative.sh
#   N=20 ./scripts/bench_speculative.sh
#   SHORT=1 ./scripts/bench_speculative.sh
#   BENCH_VIA_GATEWAY=1 ./scripts/bench_speculative.sh
#   BENCH_SEQUENTIAL=0 ./scripts/bench_speculative.sh   # both servers up (noisy on 6 GB)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMPOSE_LLAMA=(docker compose -f docker-compose.yml -f docker-compose.llamacpp.yml)
n="${N:-10}"
max_tokens="${MAX_TOKENS:-512}"
baseline="${OLLAMA_TARGET_MODEL:-qwen2.5:3b}"
spec="${OLLAMA_SPEC_MODEL:-seki-qwen-spec}"
body_file="${BENCH_BODY_FILE:-/tmp/seki-spec-body.json}"
target_url="${TARGET_URL:-http://localhost:9082/v1}"
spec_url="${SPEC_URL:-http://localhost:9081/v1}"
gateway_url="${INFERENCE_URL:-http://localhost:9000}"
key="${INFERENCE_API_KEY:-${API_KEY:-change-me}}"
sequential="${BENCH_SEQUENTIAL:-1}"

if [ "${SHORT:-0}" = "1" ]; then
  max_tokens="${MAX_TOKENS:-80}"
  bench_prompt='Reply in one short sentence: ping'
else
  bench_prompt="${BENCH_PROMPT:-Write three detailed paragraphs about a fantasy guild meeting. Use vivid prose and do not stop early.}"
fi

export bench_prompt max_tokens body_file

normalize_base() {
  local url=$1
  url="${url%/}"
  url="${url%/v1}"
  printf '%s' "$url"
}

now_s() { python3 -c "import time; print(f'{time.perf_counter():.6f}')"; }

wait_health() {
  local name=$1
  local url
  url=$(normalize_base "$2")
  echo "waiting for $name at $url/health..." >&2
  for _ in $(seq 1 90); do
    if curl -fsS "$url/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "$name not healthy at $url" >&2
  return 1
}

ensure_target_only() {
  echo "== sequential: target-only (stop llama-spec, start llama-target) ==" >&2
  "${COMPOSE_LLAMA[@]}" --profile bench stop llama-spec 2>/dev/null || true
  "${COMPOSE_LLAMA[@]}" up -d llama-target
  wait_health "llama-target" "$target_url"
  echo >&2
}

ensure_spec_only() {
  echo "== sequential: spec-only (stop llama-target, start llama-spec) ==" >&2
  "${COMPOSE_LLAMA[@]}" stop llama-target 2>/dev/null || true
  "${COMPOSE_LLAMA[@]}" --profile bench up -d llama-spec
  wait_health "llama-spec" "$spec_url"
  echo >&2
}

restore_production_llama() {
  if [ "${BENCH_NO_RESTORE:-0}" = "1" ]; then
    return 0
  fi
  echo "== restore production: llama-target for gateway ==" >&2
  "${COMPOSE_LLAMA[@]}" --profile bench stop llama-spec 2>/dev/null || true
  "${COMPOSE_LLAMA[@]}" up -d llama-target inference
  wait_health "llama-target" "$target_url" || true
  echo >&2
}

parse_response() {
  python3 <<'PY'
import json
import os

path = os.environ["body_file"]
try:
    data = json.load(open(path, encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    print("n/a n/a n/a n/a")
    raise SystemExit(0)

timings = data.get("timings") or {}
usage = data.get("usage") or {}
draft_n = timings.get("draft_n")
draft_acc = timings.get("draft_n_accepted")
pred = timings.get("predicted_n")
if pred is None:
    pred = usage.get("completion_tokens")

def fmt_num(value):
    if value is None:
        return "n/a"
    return str(int(value))

accept = "n/a"
if draft_n is not None and draft_acc is not None:
    if int(draft_n) > 0:
        accept = f"{int(draft_acc) / int(draft_n):.3f}"
    else:
        accept = "n/a"

print(accept, fmt_num(draft_acc), fmt_num(draft_n), fmt_num(pred))
PY
}

check_health() {
  local name=$1
  local url
  url=$(normalize_base "$2")
  echo "== GET $url/health ($name) =="
  curl -fsS "$url/health" || { echo "$name not ready at $url" >&2; exit 1; }
  echo
}

run_arm() {
  local arm_base=$1
  local tag=$2
  local collect_draft=${3:-0}
  local use_auth=${4:-0}
  local times=()
  local total_draft_n=0
  local total_draft_acc=0
  local draft_runs=0
  local i code start end dt accept draft_acc draft_n pred
  local curl_auth=()

  arm_base=$(normalize_base "$arm_base")
  if [ "$use_auth" = "1" ]; then
    curl_auth=(-H "Authorization: Bearer $key")
  fi

  for i in $(seq 1 "$n"); do
    start=$(now_s)
    code=$(curl -sS -o "$body_file" -w "%{http_code}" --max-time 360 \
      "${curl_auth[@]}" \
      "$arm_base/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "$(BENCH_RUN="$i" BENCH_MODEL="$tag" python3 <<'PY'
import json
import os

run = os.environ["BENCH_RUN"]
model = os.environ["BENCH_MODEL"]
prompt = os.environ["bench_prompt"]
if prompt.endswith("ping"):
    content = f"{prompt} {run}"
else:
    content = f"{prompt} (variation {run})"
print(
    json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": int(os.environ["max_tokens"]),
        }
    )
)
PY
)" \
      || true)
    end=$(now_s)
    dt=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.3f", e - s }')
    read -r accept draft_acc draft_n pred < <(parse_response)
    if [ "$code" = "200" ]; then
      times+=("$dt")
      if [ "$collect_draft" = "1" ] && [ "$draft_n" != "n/a" ] && [ "$draft_acc" != "n/a" ]; then
        total_draft_n=$((total_draft_n + draft_n))
        total_draft_acc=$((total_draft_acc + draft_acc))
        draft_runs=$((draft_runs + 1))
        echo "  $tag  $i  ${dt}s  HTTP $code  tokens=$pred  draft_accept=${draft_acc}/${draft_n}=${accept}" >&2
      else
        echo "  $tag  $i  ${dt}s  HTTP $code  tokens=$pred  draft_accept=$accept" >&2
      fi
    else
      echo "  $tag  $i  ${dt}s  HTTP ${code:-err}" >&2
    fi
  done

  if [ "${#times[@]}" -eq 0 ]; then
    echo "no successes for $tag" >&2
    echo "0"
    return
  fi

  if [ "$collect_draft" = "1" ] && [ "$draft_runs" -gt 0 ] && [ "$total_draft_n" -gt 0 ]; then
    awk -v a="$total_draft_acc" -v d="$total_draft_n" -v r="$draft_runs" 'BEGIN {
      printf "aggregate draft_accept=%d/%d=%.3f across %d runs\n", a, d, a/d, r
    }' >&2
  elif [ "$collect_draft" = "1" ]; then
    echo "aggregate draft_accept=n/a (timings missing — not llama.cpp spec?)" >&2
  fi

  printf '%s\n' "${times[@]}" | awk '{s+=$1} END {printf "%.3f", s/NR}'
}

if [ "${BENCH_VIA_GATEWAY:-0}" != "1" ] && [ "$sequential" = "1" ]; then
  trap restore_production_llama EXIT
fi

if [ "${BENCH_VIA_GATEWAY:-0}" = "1" ]; then
  gateway_base=$(normalize_base "$gateway_url")
  echo "== GET $gateway_base/ready (gateway mode) =="
  curl -fsS "$gateway_base/ready" || { echo "gateway not ready" >&2; exit 1; }
  echo
  baseline_base="$gateway_base"
  spec_base="$gateway_base"
  use_auth=1
  sequential=0
else
  use_auth=0
  if [ "$sequential" = "1" ]; then
    ensure_target_only
    baseline_base="$target_url"
    spec_base="$spec_url"
  else
    check_health "llama-target" "$target_url"
    check_health "llama-spec" "$spec_url"
    baseline_base="$target_url"
    spec_base="$spec_url"
  fi
fi

if [ "${SHORT:-0}" = "1" ]; then
  echo "== mode: short ping (SHORT=1) max_tokens=$max_tokens =="
else
  echo "== mode: long fixed prompt max_tokens=$max_tokens =="
fi
if [ "$sequential" = "1" ] && [ "${BENCH_VIA_GATEWAY:-0}" != "1" ]; then
  echo "== sequential GPU: one llama.cpp server at a time =="
fi

if [ "${BENCH_VIA_GATEWAY:-0}" = "1" ]; then
  echo "== baseline $baseline @ gateway (n=$n) =="
else
  echo "== baseline $baseline @ $(normalize_base "$baseline_base") (n=$n) =="
fi
mean_base=$(run_arm "$baseline_base" "$baseline" 0 "$use_auth")
echo "mean_s=$mean_base"
echo

if [ "$sequential" = "1" ] && [ "${BENCH_VIA_GATEWAY:-0}" != "1" ]; then
  ensure_spec_only
fi

if [ "${BENCH_VIA_GATEWAY:-0}" = "1" ]; then
  echo "== spec $spec @ gateway (n=$n) =="
else
  echo "== spec $spec @ $(normalize_base "$spec_base") (n=$n) =="
fi
mean_spec=$(run_arm "$spec_base" "$spec" 1 "$use_auth")
echo "mean_s=$mean_spec"
echo

awk -v b="$mean_base" -v s="$mean_spec" 'BEGIN {
  if (s <= 0 || b <= 0) { print "speedup=n/a"; exit }
  printf "speedup=%.2fx  (want >= 1.25 for the 25 percent goal)\n", b/s
}'
if [ "${BENCH_VIA_GATEWAY:-0}" != "1" ]; then
  echo "Sequential fair A/B on 6 GB: target-only :9082 vs spec :9081 (one server on GPU at a time)."
fi
