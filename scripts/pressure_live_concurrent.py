#!/usr/bin/env python3
"""Concurrent live pressure against a running Ollama gateway.

Serial `pressure_live.sh` only measures one-at-a-time pings. This script fires
batches the way a Discord raid would: X in-flight at once, for Y seconds
(or one flash of BATCH requests). Ollama is still `NUM_PARALLEL=1`, so extra
requests wait in HTTP, then 502 if they exceed REQUEST_TIMEOUT.

  python3 scripts/pressure_live_concurrent.py
  MODE=flash BATCH=16 python3 scripts/pressure_live_concurrent.py
  MODE=waves WAVE=8 DURATION=20 INTERVAL=2 python3 scripts/pressure_live_concurrent.py
  MODE=inflight CONCURRENCY=8 DURATION=15 python3 scripts/pressure_live_concurrent.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

BASE = os.environ.get("INFERENCE_URL", "http://localhost:9000").rstrip("/")
if BASE.endswith("/v1"):
    BASE = BASE[:-3].rstrip("/")
KEY = os.environ.get("INFERENCE_API_KEY") or os.environ.get("API_KEY") or "change-me"
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "80"))
TIMEOUT = float(os.environ.get("CHAT_TIMEOUT", "360"))
MODE = os.environ.get("MODE", "flash").strip().lower()


@dataclass
class Sample:
    ok: int = 0
    fail: int = 0
    codes: dict[str, int] = field(default_factory=dict)
    waits: list[float] = field(default_factory=list)

    def add(self, code: int, wait_s: float) -> None:
        key = str(code)
        self.codes[key] = self.codes.get(key, 0) + 1
        if code == 200:
            self.ok += 1
            self.waits.append(wait_s)
        else:
            self.fail += 1


def _chat(i: int) -> tuple[int, float]:
    payload = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Reply in one short sentence: concurrent ping {i}",
                }
            ],
            "max_tokens": MAX_TOKENS,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            resp.read()
            return int(resp.status), time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        return int(exc.code), time.perf_counter() - started
    except Exception:
        return 0, time.perf_counter() - started


def _ready() -> None:
    req = urllib.request.Request(f"{BASE}/ready")
    with urllib.request.urlopen(req, timeout=10) as resp:
        if int(resp.status) != 200:
            raise SystemExit(f"gateway /ready HTTP {resp.status}")


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    lo = int(k)
    hi = min(len(ys) - 1, lo + 1)
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - k) + ys[hi] * (k - lo)


def _summarize(label: str, sample: Sample, wall_s: float) -> None:
    print()
    print(f"== {label} ==")
    print(f"ok={sample.ok} fail={sample.fail} http={sample.codes} wall_s={wall_s:.2f}")
    if sample.waits:
        print(
            "wait_s mean={:.2f} p50={:.2f} p95={:.2f} min={:.3f} max={:.3f}".format(
                statistics.mean(sample.waits),
                _pct(sample.waits, 0.50),
                _pct(sample.waits, 0.95),
                min(sample.waits),
                max(sample.waits),
            )
        )
        print(
            f"completed_per_min={60.0 * sample.ok / max(0.01, wall_s):.1f} "
            "(GPU serial: extra in-flight requests only add wait, not throughput)"
        )


def run_flash(batch: int) -> Sample:
    sample = Sample()
    with ThreadPoolExecutor(max_workers=max(1, batch)) as pool:
        futs = [pool.submit(_chat, i) for i in range(batch)]
        for fut in as_completed(futs):
            code, wait = fut.result()
            sample.add(code, wait)
            print(f"  HTTP {code}  {wait:.3f}s")
    return sample


def run_waves(wave: int, duration: float, interval: float) -> Sample:
    """Fire WAVE requests every INTERVAL seconds for DURATION, overlapping."""
    sample = Sample()
    seq = 0
    deadline = time.perf_counter() + duration
    next_wave = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(32, wave * 8)) as pool:
        futs: dict = {}
        while time.perf_counter() < deadline or futs:
            now = time.perf_counter()
            if now < deadline and now >= next_wave:
                print(f"  wave seq={seq} size={wave}")
                for _ in range(wave):
                    futs[pool.submit(_chat, seq)] = seq
                    seq += 1
                next_wave += interval
            done = [fut for fut in futs if fut.done()]
            for fut in done:
                futs.pop(fut)
                code, wait = fut.result()
                sample.add(code, wait)
                print(f"  HTTP {code}  {wait:.3f}s")
            if time.perf_counter() >= deadline and not futs:
                break
            if not done:
                time.sleep(0.05)
    return sample


def run_inflight(concurrency: int, duration: float) -> Sample:
    sample = Sample()
    seq = 0
    deadline = time.perf_counter() + duration

    def one(i: int) -> tuple[int, float]:
        return _chat(i)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futs = {}
        for _ in range(concurrency):
            futs[pool.submit(one, seq)] = seq
            seq += 1
        while futs:
            done = next(as_completed(futs))
            _i = futs.pop(done)
            code, wait = done.result()
            sample.add(code, wait)
            print(f"  HTTP {code}  {wait:.3f}s")
            if time.perf_counter() < deadline:
                futs[pool.submit(one, seq)] = seq
                seq += 1
    return sample


def main() -> None:
    _ready()
    wall0 = time.perf_counter()
    if MODE == "flash":
        batch = int(os.environ.get("BATCH", os.environ.get("WAVE", "16")))
        print(f"== flash: {batch} requests at t=0 against {BASE} (max_tokens={MAX_TOKENS}) ==")
        sample = run_flash(batch)
        _summarize(f"flash n={batch}", sample, time.perf_counter() - wall0)
    elif MODE == "inflight":
        conc = int(os.environ.get("CONCURRENCY", os.environ.get("WAVE", "8")))
        duration = float(os.environ.get("DURATION", "20"))
        print(
            f"== inflight: {conc} in-flight for {duration}s against {BASE} "
            f"(max_tokens={MAX_TOKENS}) =="
        )
        sample = run_inflight(conc, duration)
        _summarize(f"inflight c={conc} d={duration}s", sample, time.perf_counter() - wall0)
    else:
        wave = int(os.environ.get("WAVE", os.environ.get("CONCURRENCY", "8")))
        duration = float(os.environ.get("DURATION", "20"))
        interval = float(os.environ.get("INTERVAL", "2"))
        print(
            f"== waves: {wave} at a time every {interval}s for {duration}s against {BASE} "
            f"(max_tokens={MAX_TOKENS}) =="
        )
        sample = run_waves(wave, duration, interval)
        _summarize(
            f"waves w={wave} every {interval}s for {duration}s",
            sample,
            time.perf_counter() - wall0,
        )


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as exc:
        print(f"gateway /ready failed — is Ollama up? {exc}", file=sys.stderr)
        sys.exit(1)
