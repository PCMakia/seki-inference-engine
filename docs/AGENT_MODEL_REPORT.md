# Agent-model branch report — `seki-qwen-3b` vs `qwen2.5:3b`

**Branch:** `agent-model` (from `Low_ends_6GB`)  
**Date:** 2026-08-29  
**Status:** **Decision — ship `seki-qwen-3b`**

---

## 1. Executive summary

We compared the **full Production-grade mesh** (agent-core graph + companion prompts → inference gateway → Ollama) using:

| Arm | Ollama tag | Role |
|-----|------------|------|
| **A (control)** | `qwen2.5:3b-instruct-q5_K_M` | Stock Qwen2.5-3B Instruct Q5 |
| **B (candidate)** | `seki-qwen-3b` | QLoRA fine-tune (checkpoint-250) exported to GGUF |

**Outcome:** Both arms passed all benchmark cases with **zero secretary-tone hits** under `AGENT_IDENTITY=companion`. **`seki-qwen-3b` is selected** for production: comparable or better latency, stronger companion voice, and better honesty when memory is thin. Re-fine-tuning on production `LINKED_CHAIN` prompt shape remains **optional** follow-up work, not a blocker.

---

## 2. Goals

1. Replace Seki-v1 **secretary** prompting with **companion** identity + fine-tuned voice.
2. Measure end-to-end behavior on the real agent path (not gateway-only pings).
3. Decide whether to keep base Qwen or adopt `seki-qwen-3b` on low-end (6 GB class) hardware.

**Non-goals:** Speculative decoding (stays bench-only on `dev-optimize`); vLLM on 6 GB.

---

## 3. Branch deliverables

### seki-inference-engine (`agent-model`)

| Item | Purpose |
|------|---------|
| `scripts/bench_agent_models.py` | A/B bench via `POST /agent/chat` (`tags: discord`) |
| `scripts/install_seki_qwen_ollama.sh` | Register exported GGUF as `seki-qwen-3b` |
| `scripts/up_agent_model_test.sh` | Start isolated test stack |
| `docker-compose.agent-model.test.yml` (Production-grade root) | `test-seki-*` containers, ports **10000 / 10080 / 11114** |
| `app/backends/failover.py` | Map client HF model IDs → `OLLAMA_MODEL` for Ollama |
| `docs/AGENT_MODEL_BENCHMARK.md` | Operator runbook |
| `docs/AGENT_MODEL_ENV.example` | Env templates |

### seki-agent-core (`agent-model`)

| Item | Purpose |
|------|---------|
| `AGENT_IDENTITY=companion` (default) | Drops secretary `IDENTITY_CORE_RULES` |
| `AGENT_DEFAULT_MODE=BANTERING` | Discord-style default mode |

### Fine-tune artifact (out of repo)

| Path | Notes |
|------|--------|
| `D:\AI_lab\Seki-Discord\agent-core\outputs-seki-qwen-3b\checkpoint-250\` | LoRA source |
| `...\seki-qwen-3b_gguf\qwen2.5-3b-instruct.Q5_K_M.gguf` | ~2.07 GiB Q5_K_M export (Unsloth naming) |

Export performed in **WSL** with **PyTorch cu128** (RTX 5070 Ti / sm_120). See `Seki-Discord/agent-core/training/WSL_EXPORT.md`.

---

## 4. Test methodology

- **Stack:** `docker-compose.agent-model.test.yml` (standalone; no `seki-v2-*` name/port clash).
- **Bench:** `python seki-inference-engine/scripts/bench_agent_models.py --expected-model <tag> --label <name>`
- **Cases (4):** `discord_short`, `discord_memory`, `discord_hedge`, `discord_followup`
- **Metrics:** pass rate, wall-clock latency (prep + LLM), `secretary_hits` heuristic
- **Prompting:** `AGENT_IDENTITY=companion`, no live control-plane persona override

Raw results: `Production-grade/bench_agent_models_results.jsonl` (local; not committed).

Early `base-qwen` runs failed with HTTP 404 (`Qwen/Qwen2.5-3B-Instruct` vs Ollama tag); fixed by gateway remap + env alignment. **Valid comparison uses the successful runs only** (timestamps below).

---

## 5. Quantitative results

### Summary (successful runs)

| Metric | base-qwen | seki-qwen | Δ |
|--------|-----------|-----------|---|
| Pass rate | 4/4 | 4/4 | — |
| mean_s | 39.88 | 32.45 | −19% (seki faster) |
| p50_s | 14.54 | 9.02 | −38% (seki faster) |
| max_s (1st turn, cold load) | 129.13 | 110.61 | −14% |
| secretary_hits | 0 | 0 | — |

**Interpretation:** Median latency favors **seki-qwen**; means are dominated by **first-turn model load** (~110–129 s). Warm turns for both arms were ~1–28 s. Performance is **production-viable** for either model; difference is not decisive alone.

### Per-case latency (seconds)

| Case | base-qwen | seki-qwen |
|------|-----------|-----------|
| discord_short (cold) | 129.13 | 110.61 |
| discord_memory | 27.71 | 16.79 |
| discord_hedge | 1.37 | 1.16 |
| discord_followup | 1.32 | 1.24 |

---

## 6. Qualitative comparison (reply previews)

### discord_short — “Hey Seki, you still awake?”

- **base:** Longer; “energy bank”, “chattime”, multiple emojis.
- **seki:** Tighter; “Still as alert as ever…”, one emoji.

**Edge:** Slight preference **seki** (less filler).

### discord_memory — “Remember pixel dreams last week?”

- **base:** Assumes prior chat (“How was that project…”).
- **seki:** Acknowledges topic but **admits gap** (“Can’t exactly recall the details”).

**Edge:** **seki** — better alignment with honest memory behavior.

### discord_hedge — rain on the window

- **base:** Fuller hedge line, invites opinion.
- **seki:** Short teasing beat.

**Edge:** Tie — both acceptable hedge behavior.

### discord_followup — stay up or sleep?

- **base:** Playful “flip a coin”.
- **seki:** Awkward “what time is it on Discord?”.

**Edge:** **base** — stronger banter on this turn.

### Voice / persona (overall)

Under companion identity, **seki-qwen** reads more naturally as a Discord companion without secretary scaffolding. **base-qwen** is usable but more generic-assistant in places.

---

## 7. Training vs runtime (known gap)

| | Fine-tune data | Production runtime |
|--|----------------|-------------------|
| Chain format | `Concept Chain: [Seed] -> [Mood] -> …` | `LINKED_CHAIN`, `GRAPH_LINKS`, graph steps |
| Task | Alpaca instruction → Seki reply | Translate external reasoning block + user message |

The fine-tune still helps **chain → prose** and **Seki voice** conceptually. Full format alignment was **not** required to pass this bench. Optional **phase-2 dataset** from `format_reasoning_block_text()` if graph-translation quality regresses in live Discord.

---

## 8. Decision

| Question | Answer |
|----------|--------|
| Ship which model? | **`seki-qwen-3b`** |
| Keep secretary prompts? | **No** — `AGENT_IDENTITY=companion` |
| Re-fine-tune now? | **No** — ship first; re-train only if live Discord shows gaps |
| Spec decoding on 6 GB? | **No** (per `dev-optimize` findings) |

**Rationale:** Equal reliability, better median latency, better memory honesty, preferred companion voice; one weaker banter turn is acceptable vs. re-train cost.

---

## 9. Production rollout

### Environment (root `.env` or mesh compose)

```env
OLLAMA_MODEL=seki-qwen-3b
OLLAMA_PULL_MODELS=nomic-embed-text
MODEL_NAME=seki-qwen-3b
AGENT_IDENTITY=companion
AGENT_DEFAULT_MODE=BANTERING
```

Ensure GGUF is registered in the Ollama volume used by compose (`install_seki_qwen_ollama.sh`).

### Branches to merge

1. **seki-inference-engine:** `agent-model` → `Low_ends_6GB`
2. **seki-agent-core:** `agent-model` → `main` (companion identity)

### Production compose (not test)

Use normal `docker-compose.yml` at Production-grade root (`seki-v2-*`, ports 9000/9080/9114). Copy `seki-qwen-3b` into prod Ollama volume or share model blob directory.

### Verify after deploy

```bash
curl -s http://127.0.0.1:9114/api/tags | grep seki-qwen
curl -s http://127.0.0.1:9080/agent/health
```

---

## 10. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Custom GGUF not on new machines | Document export path + `install_seki_qwen_ollama.sh`; keep checkpoint-250 |
| sm_120 / WSL export friction | `training/WSL_EXPORT.md`, PyTorch **cu128** |
| HF vs Ollama model name mismatch | Gateway forces `OLLAMA_MODEL` for Ollama backend |
| Thin bench (4 cases) | Monitor live Discord; expand bench or seed vault later |
| Banter weakness on some turns | Prompt tuning first; re-fine-tune later if needed |

---

## 11. Follow-up (optional)

- [ ] Merge `agent-model` branches after review
- [ ] Register `seki-qwen-3b` on **prod** Ollama (`seki-v2-ollama`)
- [ ] Update root `.env` on deployment hosts
- [ ] 1–2 weeks live Discord observation
- [ ] Phase-2 fine-tune on production reasoning-block format (only if needed)

---

## 12. Appendix — benchmark timestamps

| Label | UTC timestamp | expected_model |
|-------|---------------|----------------|
| base-qwen (valid) | 2026-08-29T01:34:47Z | `qwen2.5:3b-instruct-q5_K_M` |
| seki-qwen | 2026-08-29T07:27:05Z | `seki-qwen-3b` |

Failed base-qwen rows (404 model name) at `01:23:27Z` and `01:26:58Z` are excluded from analysis.

---

## Related docs

- `docs/AGENT_MODEL_BENCHMARK.md` — how to re-run A/B
- `docs/AGENT_MODEL_ENV.example` — env snippets
- `docs/QLORA_TRAINING.md` — training/export background
- `Seki-Discord/agent-core/training/WSL_EXPORT.md` — 5070 Ti export
