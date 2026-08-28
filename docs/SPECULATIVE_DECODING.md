# Speculative decoding on 6 GB (dev-optimize)

Goal: **≥25% faster** Discord-length chat on the RTX 1660 Ti without changing the reply distribution.

This branch (`dev-optimize`) packages **Qwen2.5-3B instruct** as the target and **Qwen2.5-0.5B** as the draft. Both share the Qwen2 tokenizer. Context is **2048** so the pair fits next to KV cache on 6 GB. `OLLAMA_MAX_LOADED_MODELS=1` — one serving artifact, not two resident library tags.

## Papers (what we implemented)

| Idea | Paper | Here |
|---|---|---|
| Draft model, target verifies a batch of tokens | Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICLR 2023 | 0.5B drafts, 3B verifies, `draft_num_predict` / `--spec-draft-n-max` = 8, `--spec-draft-p-min` = 0.75 |
| Extra decode heads on the target | Cai et al., Medusa | Not used (needs a Medusa-trained checkpoint) |
| Draft that reads target hidden states | Li et al., EAGLE / EAGLE-2 | llama.cpp `--spec-type draft-eagle3` if you later add an EAGLE GGUF |
| Lookahead / Jacobi | Fu et al., Lookahead Decoding, NeurIPS 2024 | llama.cpp n-gram types; not the default |
| Multi-token prediction trained in | Gloeckle et al., 2024 | Needs MTP weights; Ollama MTP is not the GGUF 3B+0.5B path |

Published draft-model speedups on chat are often **1.5–3×**. **1.25× is the success bar** for this card.

## What Compose does

1. Pull `qwen2.5:3b`, `qwen2.5:0.5b`, `nomic-embed-text`.
2. Copy the GGUF blobs to `/root/.ollama/gguf/{target,draft}.gguf`.
3. `ollama create seki-qwen-spec` with:

```
FROM .../target.gguf
DRAFT .../draft.gguf
PARAMETER num_ctx 2048
PARAMETER draft_num_predict 8
```

4. Gateway `OLLAMA_MODEL=seki-qwen-spec` and forwards `options.num_ctx` / `draft_num_predict`.

If `ollama create` rejects `DRAFT` (GGUF CUDA path on some Ollama builds), the entrypoint **aliases the 3B** as `seki-qwen-spec` so Discord still works. Then use the llama.cpp overlay.

## Measure

After `./scripts/up_llamacpp.sh`, production uses **target-only** `llama-target`. The bench script runs **sequentially** (one CUDA server on the GPU at a time):

```bash
N=20 ./scripts/bench_speculative.sh
```

1. Stops `llama-spec`, runs baseline on **`:9082`** (target-only, `draft_accept=n/a`)
2. Stops `llama-target`, runs spec on **`:9081`** (`llama-spec` profile `bench`)
3. Restores **`llama-target` + gateway** for Discord

Gateway-only (legacy, not fair on llama.cpp): `BENCH_VIA_GATEWAY=1`. Both servers at once (noisy on 6 GB): `BENCH_SEQUENTIAL=0`.

## Result on RTX 1660 Ti (6 GB)

Fair sequential bench (long prompt, `MAX_TOKENS=512`):

| Arm | Mean | Notes |
|---|---|---|
| Target-only `:9082` | ~3.6 s | Production choice |
| Spec `:9081` | ~10 s | ~83% draft acceptance |
| Speedup | **~0.35×** | Spec **slower**, not ≥1.25× |

High acceptance did not help: draft+verify overhead on Turing 6 GB exceeds any batching win. **Discord/production stays on target-only** (`llama-target` / `qwen2.5:3b`). `llama-spec` is bench-only (Compose profile `bench`).

On **`main`**, production is already target-only via Ollama `qwen2.5:3b` with no draft experiment.

## Ollama `DRAFT` failed (your logs)

If you see:

```text
[spec] WARNING: ollama create with DRAFT failed; aliasing qwen2.5:3b as seki-qwen-spec
copied 'qwen2.5:3b' to 'seki-qwen-spec'
```

then **`seki-qwen-spec` is not speculative** — it is a rename of the 3B. Ollama’s Modelfile `DRAFT` exists, but the **CUDA GGUF path on many Ollama builds does not apply it** (MTP/DRAFT is mainly exercised on other stacks). You cannot fix that with another `ollama pull`.

The GGUF pair is still written to `data/ollama/gguf/` before create fails. Use **llama.cpp** on the same files:

```bash
chmod +x scripts/up_llamacpp.sh
./scripts/up_llamacpp.sh
N=20 ./scripts/bench_speculative.sh
```

That stops GPU **Ollama**, starts **`seki-v2-llama-target`** (3B only, `:9082`) for the gateway, and keeps **`seki-v2-llama-spec`** off until the bench script enables it. Back to Ollama-only: `./scripts/up_llamacpp.sh down`.

If pull fails with `error from registry: denied`, stale GHCR login is the usual cause:

```bash
docker logout ghcr.io
./scripts/up_llamacpp.sh
```

Use the same `docker`/`sudo docker` you use for compose. Override image: `LLAMA_CPP_IMAGE=local/llama.cpp:server-cuda ./scripts/up_llamacpp.sh`.

Use **`MAX_TOKENS=256`** (or 512) for the bench — short 80-token pings hide spec gains.

## llama.cpp if Ollama does not speculate

Ollama’s `DRAFT` instruction is real (`parser.go`). Some GGUF/CUDA builds still decode the target only. Then:

```bash
# After a successful pull (gguf files exist even if create aliased)
docker compose stop ollama
./scripts/up_llamacpp.sh
```

Production overlay uses **target-only** `llama-target`. Spec stack is profile `bench` only. **Do not** run `seki-v2-ollama` and llama.cpp on this GPU at the same time.

## VRAM budget (approximate)

| Piece | Size |
|---|---|
| Qwen2.5-3B Q4 | ~1.9 GB |
| Qwen2.5-0.5B Q4 | ~0.4 GB |
| KV, both, `num_ctx=2048` | low hundreds of MB |
| Headroom | Keep TTS/vLLM off |

If the pair OOMs, drop `OLLAMA_NUM_CTX` to 1024 or use a smaller draft quant. Do not raise `OLLAMA_NUM_PARALLEL`.

## Disable

`OLLAMA_SPECULATIVE=0` skips `ollama create`. Point `OLLAMA_MODEL` at `qwen2.5:3b` and recreate the inference container.
