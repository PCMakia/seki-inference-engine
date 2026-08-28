# Speculative decoding experiment report (dev-optimize)

**Date:** 28 Aug 2026  
**Hardware:** RTX 1660 Ti (6 GB VRAM), Ubuntu / WSL + Docker, NVIDIA Container Toolkit  
**Branch:** `dev-optimize` (from `Low_ends_6GB`)  
**Goal:** ≥25% faster chat (`1.25×` wall-clock) via Qwen2.5-3B target + Qwen2.5-0.5B draft (Leviathan-style speculative decoding)

---

## Executive summary: what is actually fastest?

**Best performance on this card is plain 3B target-only decoding** — not speculative decoding.

| Workload | Best setup | Typical performance |
|----------|------------|---------------------|
| **Production Discord / long replies** | Ollama `qwen2.5:3b` **or** llama.cpp `llama-target` (3B only) | **~3.4 s** mean per long completion (~400–500 tokens); **do not use spec** |
| **Short gateway pings** (warm model, `max_tokens=80`) | Ollama `qwen2.5:3b`, serial | **~0.9–1.0 s** mean; **~200–250 completions/min** under concurrent load |
| **Speculative (3B + 0.5B draft)** | llama.cpp `llama-spec` only for experiments | **~9–10 s** for the same long workload — **~2.7× slower** than target-only despite **~84% draft acceptance** |

**Recommendation:** Ship **target-only** on `Low_ends_6GB` and `main`. Keep spec tooling as **documented opt-in / bench-only**. Do not enable `seki-qwen-spec` or `llama-spec` in production on a 6 GB Turing GPU.

---

## 1. What we tested

### 1.1 Hypothesis

Draft a cheap 0.5B model, verify batches with the 3B target (ICLR 2023 speculative decoding). Published work often reports **1.5–3×** on large GPUs; **1.25×** was chosen as a conservative success bar for Discord-length chat.

### 1.2 Stack variants

| # | Path | Description |
|---|------|-------------|
| A | **Ollama + Modelfile `DRAFT`** | `ollama create seki-qwen-spec` from `target.gguf` + `draft.gguf` |
| B | **llama.cpp `draft-simple`** | `llama-spec`: `-md draft.gguf`, `--spec-type draft-simple`, `--spec-draft-p-min 0.75` |
| C | **llama.cpp target-only** | `llama-target`: `target.gguf` only, alias `qwen2.5:3b` |
| D | **Gateway** | FastAPI `:9000` → upstream OpenAI-compatible API |

### 1.3 Benchmarks and load tools

| Script | Purpose |
|--------|---------|
| `scripts/bench_speculative.sh` | A/B wall-clock: baseline vs spec model tag |
| `scripts/pressure_live.sh` | Serial short pings (`max_tokens=80`) |
| `scripts/pressure_live_concurrent.py` | Concurrent waves / flash load |
| `scripts/up_llamacpp.sh` | Start llama.cpp overlay (production: target-only) |
| `scripts/verify_gateway.sh` | Smoke test `/ready` + one completion |

### 1.4 Prompt regimes

- **Short:** “Reply in one short sentence: ping N”, `max_tokens=80` — cheap; high variance.
- **Long (fair spec bench):** Three-paragraph fantasy guild prompt, `max_tokens=512` — exposes decode cost.

---

## 2. Results (chronological)

### 2.1 Ollama `DRAFT` path (failed to speculate)

**Observation:** Ollama entrypoint logged:

```text
[spec] WARNING: ollama create with DRAFT failed; aliasing qwen2.5:3b as seki-qwen-spec
```

**Meaning:** `seki-qwen-spec` was a **rename of the 3B**, not target+draft. CUDA GGUF path on this Ollama build did not apply Modelfile `DRAFT`.

**Bench (gateway, short ping, `N=20`, `max_tokens=80`):**

| Arm | Mean |
|-----|------|
| `qwen2.5:3b` | ~1.03 s |
| `seki-qwen-spec` | ~1.01 s |

**Interpretation:** ~2% difference = noise; **no speculation**.

---

### 2.2 llama.cpp spec via gateway (misleading “wins”)

Gateway pointed at `llama-spec` (`OLLAMA_BASE_URL=http://llama-spec:8080/v1`). Only one model loaded; both request tags hit the **same speculative server**.

| Bench | Baseline tag | Spec tag | Mean baseline | Mean spec | Reported speedup | Valid? |
|-------|--------------|----------|---------------|-----------|------------------|--------|
| Short | `qwen2.5:3b` | `seki-qwen-spec` | 0.682 s | 0.621 s | **1.10×** | **No** — same server |
| Long | `qwen2.5:3b` | `seki-qwen-spec` | 0.929 s | 0.769 s | **1.21×** | **No** — baseline also showed `draft_accept≈0.8` |

**Interpretation:** The **~10–21% “speedup”** was run-to-run variance and a **flawed A/B** (not target-only vs spec). This is the result that looked like success; it does **not** hold under a fair test.

---

### 2.3 llama.cpp fair bench (both servers up — GPU contention)

Target on `:9082`, spec on `:9081`, **both containers running** on one GPU:

| Arm | Mean | Draft acceptance |
|-----|------|------------------|
| Target-only | 3.57 s | n/a |
| Spec | 10.30 s | ~83% aggregate |
| Speedup | **0.35×** | Spec slower |

**Interpretation:** First honest split of target vs spec, but **two CUDA processes** shared 6 GB VRAM.

---

### 2.4 llama.cpp fair **sequential** bench (definitive)

`bench_speculative.sh` with `BENCH_SEQUENTIAL=1` (default): stop the other server before each arm; restore `llama-target` for production after.

Long prompt, `N=20`, `max_tokens=512`:

| Arm | Mean | Tokens (typical) | Draft acceptance |
|-----|------|------------------|------------------|
| **Target-only** (`:9082`) | **3.42 s** | ~400–512 | n/a |
| **Spec** (`:9081`) | **9.32 s** | ~300–512 | **84.2%** (2381/2827) |
| **Speedup** | **0.37×** | — | Spec **~2.7× slower** |

**Interpretation:** High draft acceptance **does not** overcome draft+verify overhead on Turing 6 GB. **This is the number to trust for production decisions.**

---

### 2.5 Baseline throughput (no spec) — `Low_ends_6GB` live tests

From `docs/LIVE_PERFORMANCE.md` (Ollama `qwen2.5:3b`, short pings):

| Test | Result |
|------|--------|
| Serial `N=2000` | mean **~0.91 s**, 0 failures |
| Concurrent light | **~227 completions/min** |
| Concurrent medium | **~246 completions/min** |
| Heavy flood | **~196 completions/min**, 502s and long waits |

**Interpretation:** GPU is **serial** (`OLLAMA_NUM_PARALLEL=1`). Extra concurrency increases **wait**, not throughput. Real Discord turns (agent-core + longer decode) are **~6–8 s** in sim — not these 0.9 s pings.

---

## 3. What the results mean

### 3.1 Why acceptance can be high but speed still loses

Speculative decoding pays off when:

```text
(time saved on target forwards) > (draft runs + verify batches + extra memory traffic)
```

On **RTX 1660 Ti**, the **fixed cost per speculative block** is large relative to 3B Q4 decode. At **~84% acceptance**, the draft is often right, but the card still spends more time on draft+verify than it saves.

### 3.2 Why earlier benches lied

| Mistake | Effect |
|---------|--------|
| Same server for both model tags | Both arms run spec (or same weights) |
| Short prompts | Model stops early; high timing noise |
| Two GPU servers at once | VRAM/compute contention skews spec arm |
| Ollama alias instead of real `DRAFT` | Spec arm is identical to baseline |

The **sequential target-only vs spec** bench fixes the first and third issues.

### 3.3 `/ready` says `"ollama": true` on llama.cpp

The gateway names its upstream client `ollama` regardless of URL. **`ollama: true` does not mean the Ollama container is running.** Check `OLLAMA_BASE_URL` on `seki-v2-inference`.

---

## 4. Best performance — practical guide

### For production (Discord on 1660 Ti)

| Priority | Choice |
|----------|--------|
| **Fastest chat** | **Ollama `qwen2.5:3b`** (`docker compose up`) or **`./scripts/up_llamacpp.sh`** → `llama-target` only |
| **Model tag** | `qwen2.5:3b` — not `seki-qwen-spec` |
| **Avoid** | `llama-spec`, Ollama `seki-qwen-spec` with failed `DRAFT`, running two llama.cpp servers in production |

### For embeddings

Ollama `nomic-embed-text` on the default compose path. llama.cpp overlay stops GPU Ollama — use `./scripts/up_llamacpp.sh down` when you need the full Ollama mesh.

### For load testing

- Serial soak: `N=2000 ./scripts/pressure_live.sh`
- Concurrent ceiling: `scripts/pressure_live_concurrent.py` (expect ~200–250/min short pings, not higher with one GPU job queue)

### For documenting spec research only

```bash
N=20 ./scripts/bench_speculative.sh   # sequential fair A/B; expect ~0.37× on this card
```

---

## 5. What to merge back to branches

### 5.1 `dev-optimize` → `Low_ends_6GB` (recommended)

Merge the **infrastructure and documentation**; keep **spec off by default** in production compose.

| Include | Rationale |
|---------|-----------|
| `docker-compose.llamacpp.yml` | Optional llama.cpp path; **gateway → `llama-target`** |
| `scripts/up_llamacpp.sh` | Target-only production overlay |
| `scripts/bench_speculative.sh` | Sequential fair A/B + draft metrics |
| `docs/SPEC_EXPERIMENT_REPORT.md` (this file) | Record of experiment |
| `docs/SPECULATIVE_DECODING.md` | How spec was attempted; conclusion |
| `docs/LIVE_PERFORMANCE.md` | Live throughput baselines |
| `scripts/pressure_live*.sh/py` | Load testing |
| Gateway fixes (`REQUEST_TIMEOUT=300`, `OLLAMA_DATA_PATH`, LF scripts, `.gitattributes`) | Stability on Linux Docker |
| `Execution_scripts/create_speculative_model.sh` | Still useful to export GGUF pair; optional if `OLLAMA_SPECULATIVE=0` |
| `app/config.py` (`num_ctx`, `draft_num_predict` options) | Harmless when not used |

| Set default / behaviour | Value |
|-------------------------|-------|
| `OLLAMA_SPECULATIVE` | **`0`** on `Low_ends_6GB` (skip broken Ollama `DRAFT` create) |
| `OLLAMA_MODEL` | **`qwen2.5:3b`** |
| Production llama path | **`llama-target` only**; `llama-spec` profile **`bench`** |

| Do **not** default on Low_ends | Reason |
|--------------------------------|--------|
| Gateway → `llama-spec` | ~2.7× slower than target |
| `OLLAMA_MODEL=seki-qwen-spec` in production | Implies spec stack |
| Expecting 1.25× speedup on 6 GB Turing | Not achieved |

### 5.2 `Low_ends_6GB` → `main`

`main` should stay **simple Ollama-only** unless you explicitly want the llama.cpp overlay documented as optional.

| Include on `main` | Rationale |
|-------------------|-----------|
| Gateway hardening (timeout, paths, verify scripts) | Production stability |
| `docs/LIVE_PERFORMANCE.md` | Operational expectations |
| Pressure / verify scripts | Ops tooling |
| README clarity (6 GB, no vLLM) | Onboarding |

| Exclude from `main` (or keep on a docs branch only) | Reason |
|-----------------------------------------------------|--------|
| `docker-compose.llamacpp.yml` as default path | Extra complexity; Ollama is enough for `main` |
| `OLLAMA_SPECULATIVE=1` | No benefit on this hardware |
| Spec-first README wording | Experiment failed speed goal |

### 5.3 `seki-discord-bot` (Production-grade root)

Already useful on `main` from earlier work; **not spec-specific**:

- Streamer flood policy (`DISCORD_STREAMER_*`) — reduces pile-up wait under load
- `docs/PRESSURE_TEST.md` — sim vs live expectations

No change required for spec conclusion; bot should point at **gateway + `qwen2.5:3b`**.

---

## 6. Conclusion

| Question | Answer |
|----------|--------|
| Did we hit the 25% speedup goal? | **No.** Fair speedup **~0.37×** (spec slower). |
| Was there ever a real ~20% win? | **No** — **1.21×** was a **benchmark artifact** (same spec server for both tags). |
| What is actually fastest? | **Target-only 3B** — Ollama or `llama-target`, **~3.4 s** long / **~1 s** short warm pings. |
| Should we ship speculative decoding on 1660 Ti? | **No** for production chat. |
| What did we gain from `dev-optimize`? | GGUF export path, llama.cpp overlay, fair bench methodology, live perf baselines, clear **no-go** on spec for this card. |

**Final production stack for 6 GB (this experiment):**

```text
Discord → seki-agent-core → seki-inference-engine (:9000) → Ollama qwen2.5:3b (:9114)
                              or, if using llama overlay:
                            → llama-target (:9082, target-only)
```

Spec stack (`llama-spec`) remains in repo **for reproducibility and future hardware**, not for Discord on RTX 1660 Ti.

---

## 7. References

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICLR 2023  
- In-repo: `docs/SPECULATIVE_DECODING.md`, `docs/LIVE_PERFORMANCE.md`, `docs/UBUNTU_6GB.md`  
- llama.cpp `draft-simple`, `--spec-draft-p-min` (default tuning used: `0.75`, `n_max=8`)
