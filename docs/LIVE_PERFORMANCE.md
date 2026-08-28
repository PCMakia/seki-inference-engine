# Live inference performance (Low_ends_6GB)

Measured on **28 Aug 2026** against `seki-v2-inference` + `seki-v2-ollama` (`qwen2.5:3b`, OpenAI `/v1/chat/completions`). Host: Windows + Docker Desktop / WSL, **RTX-class 6 GB** path used for this branch (Ollama only, `OLLAMA_NUM_PARALLEL=1`).

This is **not** the discrete-event Discord sim. That write-up is `seki-discord-bot/docs/PRESSURE_TEST.md` in the Production-grade tree. Live scripts: `scripts/pressure_live.sh`, `scripts/pressure_live_concurrent.py`.

**Prompt used for all live rows:** one short sentence ping, `max_tokens=80`. Gateway `REQUEST_TIMEOUT=300`. Concurrent client `CHAT_TIMEOUT=360`.

---

## 1. What to quote

| Number | Use for | Do not use for |
|---|---|---|
| **~0.9–2s** mean | Warm GGUF, already loaded, tiny reply | Discord + agent-core turns |
| **~200–250 completions/min** | Short-ping GPU ceiling | Mention rate in a guild |
| **~6–8s** (sim) | Real companion replies (decode + agent-core) | These live pings |
| **~8 replies/min** | Comfortable Discord on this card | MEE6-style server counts |

Idle first-load of the GGUF can take **>120s** (gateway used to abort at 120s). After warmup, short pings are ~1s.

`min_s=-1.965` on the serial N=2000 run is a WSL `date +%s.%N` clock glitch. Serial script now uses `time.perf_counter()`.

---

## 2. Serial (one at a time)

`N=2000 ./scripts/pressure_live.sh`

| Metric | Value |
|---|---|
| ok / fail | **2000 / 0** |
| mean | **0.91s** |
| p50 | 0.99s |
| p95 | 1.22s |
| max | 1.59s |
| Implied throughput | **~66/min** if you take 0.91s strictly; later concurrent drains ran **~200–250/min** once the card was fully hot |

These pings are much cheaper than a Discord turn. Do not replace `GEN_MU` in `pressure_sim.py` with 0.91 unless you also shorten `max_tokens` and skip agent-core.

---

## 3. Concurrent batches (raw gateway, no Discord queue)

Ollama still runs **one** job. Extra HTTP calls wait, then 502 if they sit past 300s, or client code `0` if the socket dies / hits 360s.

### 3.1 Known wave (instrumented)

`MODE=waves WAVE=8 DURATION=20 INTERVAL=2` — 10 waves × 8 = **80** requests.

| Metric | Value |
|---|---|
| ok / fail | 80 / 0 |
| wall | 72.1s |
| mean wait | **40.5s** |
| p50 / p95 | 44.4s / 47.7s |
| min / max | 19.7s / 50.2s |
| completions/min | 66.5 |

Same GPU rate as a slow serial pass; clients just wait ~40s in the pile.

### 3.2 Follow-up live waves (operator paste)

Exact `WAVE` / `INTERVAL` were not logged on these three; totals and waits are as printed by the script.

| Run | ok | fail | HTTP mix | wall | mean wait | p50 | p95 | min | max | /min |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Light | 80 | 0 | 200 only | 21.1s | 2.02s | 1.98s | 2.96s | 1.11s | 3.41s | **227** |
| Medium | 126 | 0 | 200 only | 30.7s | 6.86s | 6.41s | 11.4s | 2.27s | 12.6s | **246** |
| Heavy | 2656 | **944** | 200: 2656; **502: 553**; **0: 391** | 814s | **229s** | 254s | 345s | 10.5s | 360s | **196** |

Heavy submitted **3600** calls (2656 + 944). Failures match the timeouts: 502 from the gateway, `0` from the client. Throughput stayed ~200/min; wait went to **~4 minutes**.

### 3.3 Ceiling

| | |
|---|---|
| Short-ping GPU | **~200–250/min** (~3–4/s) |
| Do not run | `WAVE=2000 INTERVAL=1 DURATION=30` (60k sockets, ~16k threads) |
| Expected if you did | ~1k–1.2k OK, rest 502/`0`, likely fd/RAM/Docker failure first |
| Knee for more data | `WAVE=16/32/64`, not 2000 |

---

## 4. Discord (what users actually hit)

The concurrent script **bypasses** the bot. A raid of 3600 @mentions must not become 3600 Ollama jobs.

| Policy | 40 mentions in 20s | GPU jobs | Feel |
|---|---|---|---|
| FIFO max 12 | Keep last 12 | 12 serial | Machine-gun 12 replies |
| Raw gateway (this report) | All calls hit Ollama | All serial, long wait | p50 tens to hundreds of seconds |
| **Streamer (default)** | Keep **1** waiting, lottery the rest | ~1 start per `DISCORD_STREAMER_GAP_SEC` (6s) while flooded | Most pings silently missed |

Knobs: `DISCORD_STREAMER=1`, `DISCORD_STREAMER_FLOOD_N=4` per 10s, `DISCORD_STREAMER_BACKLOG=1`, `DISCORD_STREAMER_GAP_SEC=6`. Set `DISCORD_STREAMER=0` for old FIFO.

Comfortable **in-character** rate remains the sim: **~5–8 mentions/min** global, not 200/min.

---

## 5. Re-run

```bash
cd seki-inference-engine
# warmup / smoke
./scripts/verify_gateway.sh

# serial (use N=20 on Ubuntu; 2000 was a soak)
N=20 ./scripts/pressure_live.sh

# concurrent — stop when p95 wait leaves ~2s or 502s appear
MODE=flash BATCH=16 python3 scripts/pressure_live_concurrent.py
MODE=waves WAVE=8 DURATION=20 INTERVAL=2 python3 scripts/pressure_live_concurrent.py
MODE=waves WAVE=32 DURATION=20 INTERVAL=2 python3 scripts/pressure_live_concurrent.py
```

Keep vLLM and `seki-tts` off this GPU or these numbers are invalid.
