# Phase 7 — Voice QLoRA Pipeline (Option A)

Synthetic dataset generation + Unsloth QLoRA fine-tune of `Qwen/Qwen2.5-3B-Instruct`
for Seki Amahara voice, then GGUF export for local Ollama.

## Hardware notes (GTX 1660 Ti)

| Setting | Recommendation |
|---|---|
| VRAM | 6 GB (tight) — keep `load_in_4bit=True`, batch size `1`, grad accum `8` |
| Driver | Recent NVIDIA Game Ready / Studio with CUDA 12.x toolkit matching Unsloth wheels |
| Close | Discord overlays, browsers with WebGL tabs, other CUDA jobs |
| Expect | Slow epochs; prefer overnight runs for the full 1k–2k row set |

If OOM: reduce `MAX_SEQ_LENGTH` to `1024` in `train_unsloth.py`, or cut the dataset with `--target 500` during generation.

## 1. Prerequisites

```bash
# Ollama (for inference after export) — pull base if needed
ollama pull qwen2.5:3b-instruct-q5_K_M

# Training env (separate from agent-core Discord venv recommended)
python -m venv .venv-unsloth
# Windows:
.venv-unsloth\Scripts\activate
# Linux/macOS:
source .venv-unsloth/bin/activate

pip install --upgrade pip
# Follow current Unsloth install docs for your CUDA version, e.g.:
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install datasets trl bitsandbytes accelerate
```

Also ensure `httpx` and `python-dotenv` are available for dataset generation
(already in `agent-core/requirements.txt`).

## 2. Generate synthetic dataset (NVIDIA NIM)

From `agent-core/`:

```bash
# PowerShell
$env:NVIDIA_API_KEY="nvapi-..."
python training/generate_dataset.py --target 1000 --concurrency 4

# bash
export NVIDIA_API_KEY=nvapi-...
python training/generate_dataset.py --target 1000 --concurrency 4
```

Output: `agent-core/training/synthetic_seki_data.jsonl`

Each line:

```json
{"instruction":"Concept Chain: ...","output":"Seki's voice response..."}
```

Aim for **1,000–2,000** rows (`--target 2000` for the upper bound). The generator
uses `deepseek-ai/deepseek-v4-flash` on `https://integrate.api.nvidia.com/v1`.

## 3. Fine-tune with Unsloth QLoRA

Still inside the Unsloth venv, from `agent-core/`:

```bash
python training/train_unsloth.py
```

**Windows + Triton:** plain `python` fails with `FileNotFoundError` / missing C compiler because MSVC is not on PATH. Use the launcher (loads `vcvars64`, sets `CC` to `cl.exe`, and points `CUDA_PATH` at your toolkit):

```powershell
powershell -ExecutionPolicy Bypass -File training/run_train_windows.ps1
```

Your machine’s CUDA toolkit root is `C:\Library\CUDA_Toolkits` (12.8). Triton needs `bin\ptxas.exe`, `include\cuda.h`, and `lib\x64\cuda.lib` under that path — CUDA alone is not enough without MSVC in the same shell.

> **Import order:** `train_unsloth.py` imports Unsloth **before** TRL on purpose. Importing `trl` first makes Unsloth leave `eos_token` as the placeholder `<EOS_TOKEN>`, which is not in Qwen2's vocabulary and crashes `SFTTrainer`.

Configured PEFT (locked by Phase 7 plan):

- Base: `Qwen/Qwen2.5-3B-Instruct`
- `r=16`, `lora_alpha=16`
- `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`
- `SFTTrainer` on Alpaca-style `### Instruction` / `### Response` text
- Final export: `model.save_pretrained_gguf("seki-qwen-3b", tokenizer, quantization_method="q5_k_m")`

Artifacts:

- Checkpoints → `agent-core/outputs-seki-qwen-3b/`
- GGUF folder → `agent-core/seki-qwen-3b/`

## 4. Load into Ollama

Create a Modelfile pointing at the exported GGUF (adjust path to the `.gguf` file Unsloth wrote):

```dockerfile
FROM ./seki-qwen-3b/unsloth.Q5_K_M.gguf
PARAMETER temperature 0.8
SYSTEM You are Seki Amahara, a reflective Discord companion and AI VTuber.
```

```bash
ollama create seki-qwen-3b -f Modelfile
ollama run seki-qwen-3b
```

Then point the agent hybrid router / Discord bot local fallback model name at
`seki-qwen-3b` when ready to swap off the stock `qwen2.5:3b-instruct-q5_K_M`.

## 5. Dual embedding reminder (control plane)

Training is independent of embeddings, but runtime memory search now defaults to
**LOCAL** Ollama `nomic-embed-text` (768-d → `embedding_local`). OpenAI
`text-embedding-3-small` (1536-d → `embedding`) remains optional via dashboard
**Embedding Provider** — `OPENAI_API_KEY` is only required when that toggle is
set to OpenAI.
