# Agent model A/B — `seki-qwen-3b` vs `qwen2.5:3b` (Low_ends_6GB mesh)

Branch **`agent-model`** (from `Low_ends_6GB`) compares the **full production line**:

```
Discord → agent-core (graph + reasoning chain + prompts) → inference gateway → Ollama
```

Goal: decide whether the existing fine-tune beats base Qwen **without** Seki-v1 secretary prompting, and whether you need a **re-fine-tune** on production prompt shape.

## Quick start

1. Checkout branches:
   - `seki-inference-engine`: `agent-model`
   - `seki-agent-core`: `agent-model`
2. Start the **isolated test stack** from Production-grade root (does not touch `seki-v2-*`):

   **Windows:**
   ```powershell
   .\scripts\up_agent_model_test.ps1 -Build
   ```

   **Linux / WSL:**
   ```bash
   chmod +x seki-inference-engine/scripts/up_agent_model_test.sh
   ./seki-inference-engine/scripts/up_agent_model_test.sh --build
   ```

   Or manually:
   ```bash
   docker compose -f docker-compose.agent-model.test.yml up -d --build
   ```

   Test containers: `test-seki-ollama`, `test-seki-inference`, `test-seki-agent-core`  
   Test ports: **10000** (gateway), **10080** (agent), **11114** (Ollama), **10002** (WS)

   Production (`seki-v2-*` on 9000/9080/9114) can keep running, but **only one Ollama should use the GPU** — stop prod Ollama if VRAM is tight:
   ```bash
   docker stop seki-v2-ollama
   ```

3. **Arm A (base)** — set model vars in `.env`, then recreate the test stack:
   ```bash
   docker compose -f docker-compose.agent-model.test.yml up -d --force-recreate
   ```
4. Run benchmark (defaults target test ports 10080 / 11114):
   ```bash
   python3 seki-inference-engine/scripts/bench_agent_models.py \
     --expected-model qwen2.5:3b-instruct-q5_K_M --label base-qwen
   ```
5. **Arm B (fine-tune)** — export/install GGUF (below), switch env to `seki-qwen-3b`, recreate test stack, bench again with `--label seki-qwen`.

Stop the test stack when done:
```bash
docker compose -f docker-compose.agent-model.test.yml down
```

Results append to `bench_agent_models_results.jsonl`.

**Final report:** `docs/AGENT_MODEL_REPORT.md` — decision to ship **`seki-qwen-3b`**.

## What changed on `agent-model`

| Component | Change |
|-----------|--------|
| **agent-core** | `AGENT_IDENTITY=companion` (default) replaces secretary `IDENTITY_CORE_RULES`; `AGENT_DEFAULT_MODE=BANTERING` for Discord-style turns |
| **inference** | Same Ollama-only stack; swap `OLLAMA_MODEL` / `MODEL_NAME` / `INFERENCE_MODEL` together |
| **bench** | `scripts/bench_agent_models.py` — end-to-end `/agent/chat` with `tags: ["discord"]` |

Set `AGENT_IDENTITY=secretary` to restore v1 workplace voice.

## Install `seki-qwen-3b` in Ollama

Training artifacts live under `agent-core/` (gitignored weights). On this machine the LoRA checkpoint exists at `D:\AI_lab\Seki-Discord\agent-core\outputs-seki-qwen-3b\checkpoint-250\` but **GGUF was not exported**.

### Export GGUF (one-time)

From a machine with Unsloth + GPU (see `docs/QLORA_TRAINING.md`):

```python
# Merge LoRA and export — paths adjusted to your checkpoint
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "outputs-seki-qwen-3b/checkpoint-250",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
model.save_pretrained_gguf("seki-qwen-3b", tokenizer, quantization_method="q5_k_m")
```

Copy `seki-qwen-3b/unsloth.Q5_K_M.gguf` into `seki-inference-engine/agent-core/seki-qwen-3b/` (or set `GGUF_PATH`).

### Register in Ollama

```bash
chmod +x scripts/install_seki_qwen_ollama.sh
GGUF_PATH=/path/to/unsloth.Q5_K_M.gguf ./scripts/install_seki_qwen_ollama.sh
```

Verify: `curl -s http://127.0.0.1:9114/api/tags | grep seki-qwen`

## Benchmark interpretation

| Metric | Meaning |
|--------|---------|
| `mean_s` / `p50_s` | Full turn latency (graph + memory + LLM). Expect similar VRAM; fine-tune Q5 ≈ base Q5 speed. |
| `secretary_hits` | Reply contained secretary markers — should be **0** with `AGENT_IDENTITY=companion`. |
| Reply preview | Subjective Seki voice / chain translation quality |

Compare JSONL records for the same case IDs across labels. Latency alone does not pick the winner; read replies for hedge + follow-up cases.

Gateway-only latency (no graph): `scripts/pressure_live.sh`.

## Training vs production prompt shape

| | **Fine-tune (`synthetic_seki_data.jsonl`)** | **Runtime (`format_reasoning_block_text`)** |
|--|---------------------------------------------|---------------------------------------------|
| Input | `Concept Chain: [Seed] -> [Mood] -> [Desire] -> ...` | `LINKED_CHAIN head-to-tail: [concept] ...` + `GRAPH_LINKS` |
| Task | Alpaca `### Instruction` / `### Response` | Layered system + user blocks; translate external CoT |
| Voice | VTuber Seki (synthetic) | Companion (env) + Discord instructions |

The fine-tune teaches **chain → Seki prose** on **synthetic** chains. Production builds **real** graph chains in a **different format**. Overlap is conceptual (structured thought → natural language), not format-identical.

## Do you need to re-fine-tune?

Use this decision tree after running the A/B bench:

```
Run A/B with AGENT_IDENTITY=companion
        │
        ├─ seki-qwen clearly better voice + chain translation, secretary_hits ≈ 0
        │     → Ship seki-qwen-3b. Re-fine-tune optional (diminishing returns).
        │
        ├─ seki-qwen better voice but weak on LINKED_CHAIN / memory turns
        │     → Re-fine-tune recommended: new dataset from production prompts
        │       (generate_dataset using format_reasoning_block_text samples).
        │
        ├─ base qwen similar quality, seki-qwen not worse
        │     → Prefer base qwen (simpler ops) unless you want baked-in persona.
        │
        └─ seki-qwen worse on abstention / hallucination cases
              → Stay on base; re-fine-tune with production-format data + safety rows.
```

**Re-fine-tune is worth it when:**

- Companion identity + production reasoning blocks are stable (prompt format frozen).
- A/B shows persona wins but **translation** of `LINKED_CHAIN` / memory context lags base model.
- You add rows where `instruction` = full user-layer block (or `LINKED_CHAIN` line) and `output` = good Discord replies from NIM or human edits.

**Re-fine-tune is not required when:**

- `seki-qwen-3b` already beats base on your bench cases with `AGENT_IDENTITY=companion`.
- You are still changing prompt layout often — train after the format settles.

## Env reference

| Variable | Arm A (base) | Arm B (fine-tune) |
|----------|--------------|-------------------|
| `OLLAMA_MODEL` | `qwen2.5:3b-instruct-q5_K_M` | `seki-qwen-3b` |
| `MODEL_NAME` / `INFERENCE_MODEL` | same as Ollama tag | `seki-qwen-3b` |
| `AGENT_IDENTITY` | `companion` | `companion` |
| `AGENT_DEFAULT_MODE` | `BANTERING` | `BANTERING` |

**Important:** set all three model vars to the same Ollama tag. The gateway remaps client HuggingFace IDs (e.g. `Qwen/Qwen2.5-3B-Instruct`) to `OLLAMA_MODEL` automatically, but `INFERENCE_MODEL` in compose should still match the Ollama tag for clarity.

### HTTP 404 `model 'Qwen/Qwen2.5-3B-Instruct' not found`

Not a port issue — agent-core reached the gateway, but Ollama only knows GGUF **tags** (`qwen2.5:3b-instruct-q5_K_M`), not HuggingFace names. Rebuild inference after pulling `agent-model`, or set in `.env`:

```
OLLAMA_MODEL=qwen2.5:3b-instruct-q5_K_M
MODEL_NAME=qwen2.5:3b-instruct-q5_K_M
```

Then `docker compose -f docker-compose.agent-model.test.yml up -d --build inference agent-core`.

## Related docs

- `docs/QLORA_TRAINING.md` — original QLoRA + export
- `docs/LIVE_PERFORMANCE.md` — gateway-only pressure
- `docs/UBUNTU_6GB.md` — 6 GB constraints
