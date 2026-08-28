#!/usr/bin/env python3
"""A/B benchmark: production agent-core path with two Ollama chat models.

Measures end-to-end wall time (prep + LLM) via POST /agent/chat with Discord tags,
matching the Low_ends_6GB mesh. Run once per model arm after switching OLLAMA_MODEL
and restarting inference + agent-core.

Usage (from repo root or Production-grade root with stack up):

  # Arm A — base Qwen (default Low_ends_6GB)
  export OLLAMA_MODEL=qwen2.5:3b-instruct-q5_K_M
  export MODEL_NAME=qwen2.5:3b-instruct-q5_K_M
  export INFERENCE_MODEL=qwen2.5:3b-instruct-q5_K_M
  docker compose up -d --force-recreate inference agent-core   # root mesh

  python3 seki-inference-engine/scripts/bench_agent_models.py \\
    --expected-model qwen2.5:3b-instruct-q5_K_M \\
    --label base-qwen

  # Arm B — fine-tuned Seki (after ollama create seki-qwen-3b)
  export OLLAMA_MODEL=seki-qwen-3b
  export MODEL_NAME=seki-qwen-3b
  export INFERENCE_MODEL=seki-qwen-3b
  docker compose up -d --force-recreate inference agent-core

  python3 seki-inference-engine/scripts/bench_agent_models.py \\
    --expected-model seki-qwen-3b \\
    --label seki-qwen

Results append to bench_agent_models_results.jsonl in the cwd unless --out is set.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_AGENT = os.environ.get(
    "AGENT_BASE_URL",
    os.environ.get("TEST_AGENT_BASE_URL", "http://127.0.0.1:9180"),
).rstrip("/")
DEFAULT_OLLAMA = os.environ.get(
    "OLLAMA_URL",
    os.environ.get(
        "TEST_OLLAMA_URL",
        f"http://127.0.0.1:{os.environ.get('TEST_OLLAMA_HOST_PORT', '9124')}",
    ),
).rstrip("/")

# Representative production-line turns (Discord companion, reasoning chain in prompt).
CASES: list[dict[str, Any]] = [
    {
        "id": "discord_short",
        "message": "Hey Seki, you still awake?",
        "tags": ["discord"],
        "phase": "reply",
    },
    {
        "id": "discord_memory",
        "message": "Remember when we talked about pixel dreams last week?",
        "tags": ["discord"],
        "phase": "reply",
    },
    {
        "id": "discord_hedge",
        "message": "What do you think about rain on the window?",
        "tags": ["discord"],
        "phase": "hedge",
    },
    {
        "id": "discord_followup",
        "message": "Okay but seriously — should I stay up or sleep?",
        "tags": ["discord"],
        "phase": "reply",
    },
]

SECRETARY_MARKERS = (
    "secretary",
    "schedule",
    "prioritization",
    "task execution",
    "boss's personal secretary",
)


@dataclass
class CaseResult:
    case_id: str
    label: str
    expected_model: str
    ok: bool
    latency_s: float
    reply_preview: str
    secretary_tone: bool
    error: str | None = None


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 300.0,
) -> tuple[int, Any]:
    data = None
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body.strip() else {"detail": body[:500]}
        except json.JSONDecodeError:
            parsed = {"detail": body[:500]}
        return exc.code, parsed


def ollama_has_model(base_url: str, model: str) -> bool:
    status, data = _http_json("GET", f"{base_url}/api/tags", timeout=15.0)
    if status != 200 or not isinstance(data, dict):
        return False
    names = {m.get("name", "").split(":")[0] for m in data.get("models", [])}
    want = model.split(":")[0]
    return want in names


def agent_ready(agent_base: str) -> bool:
    status, _ = _http_json("GET", f"{agent_base}/agent/health", timeout=10.0)
    return status == 200


def run_case(agent_base: str, case: dict[str, Any], session_id: str) -> CaseResult:
    started = time.perf_counter()
    status, data = _http_json(
        "POST",
        f"{agent_base}/agent/chat",
        {
            "message": case["message"],
            "session_id": session_id,
            "tags": case.get("tags"),
            "phase": case.get("phase", "reply"),
        },
        timeout=float(os.environ.get("BENCH_TIMEOUT", "300")),
    )
    elapsed = time.perf_counter() - started
    if status != 200:
        err = data.get("detail") if isinstance(data, dict) else str(data)
        return CaseResult(
            case_id=case["id"],
            label="",
            expected_model="",
            ok=False,
            latency_s=round(elapsed, 3),
            reply_preview="",
            secretary_tone=False,
            error=f"HTTP {status}: {err}",
        )
    reply = (data.get("reply") or "").strip()
    lower = reply.lower()
    secretary = any(m in lower for m in SECRETARY_MARKERS)
    return CaseResult(
        case_id=case["id"],
        label="",
        expected_model="",
        ok=bool(reply),
        latency_s=round(elapsed, 3),
        reply_preview=reply[:240].replace("\n", " "),
        secretary_tone=secretary,
    )


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    latencies = [r.latency_s for r in ok]
    return {
        "cases": len(results),
        "ok": len(ok),
        "fail": len(results) - len(ok),
        "mean_s": round(statistics.mean(latencies), 3) if latencies else None,
        "p50_s": round(statistics.median(latencies), 3) if latencies else None,
        "max_s": round(max(latencies), 3) if latencies else None,
        "secretary_hits": sum(1 for r in ok if r.secretary_tone),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark agent-core with a chat model arm.")
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--ollama", default=DEFAULT_OLLAMA)
    parser.add_argument("--expected-model", required=True, help="Ollama tag you configured for this arm.")
    parser.add_argument("--label", default="", help="Short name for results (e.g. base-qwen, seki-qwen).")
    parser.add_argument("--rounds", type=int, default=1, help="Repeat full case list N times.")
    parser.add_argument("--out", default="bench_agent_models_results.jsonl")
    parser.add_argument("--skip-model-check", action="store_true")
    args = parser.parse_args()

    label = args.label or args.expected_model.split(":")[0]

    if not args.skip_model_check and not ollama_has_model(args.ollama, args.expected_model):
        print(
            f"ERROR: Ollama at {args.ollama} does not list {args.expected_model!r}.\n"
            "Run scripts/install_seki_qwen_ollama.sh or set OLLAMA_MODEL and recreate inference.",
            file=sys.stderr,
        )
        return 2

    if not agent_ready(args.agent):
        print(f"ERROR: agent-core not healthy at {args.agent}/agent/health", file=sys.stderr)
        return 2

    print(f"== bench_agent_models label={label} expected={args.expected_model} ==")
    all_results: list[CaseResult] = []
    session_id = f"bench-{label}-{int(time.time())}"

    for r in range(1, args.rounds + 1):
        for case in CASES:
            print(f"  round {r}/{args.rounds}  {case['id']} ...", flush=True)
            res = run_case(args.agent, case, session_id)
            res.label = label
            res.expected_model = args.expected_model
            all_results.append(res)
            status = "ok" if res.ok else "FAIL"
            sec = " SECRETARY-TONE" if res.secretary_tone else ""
            print(f"    {status} {res.latency_s}s{sec}")
            if res.error:
                print(f"    {res.error}")
            elif res.reply_preview:
                print(f"    > {res.reply_preview[:120]}...")

    summary = summarize(all_results)
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": label,
        "expected_model": args.expected_model,
        "summary": summary,
        "results": [asdict(r) for r in all_results],
    }
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print()
    print(
        f"done label={label} ok={summary['ok']}/{summary['cases']} "
        f"mean_s={summary['mean_s']} p50_s={summary['p50_s']} "
        f"secretary_hits={summary['secretary_hits']}"
    )
    print(f"appended {args.out}")
    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
