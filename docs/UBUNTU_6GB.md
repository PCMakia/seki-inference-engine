# Ubuntu 6 GB (dev-optimize)

Target: Ubuntu + NVIDIA driver + Container Toolkit, **RTX 1660 Ti 6 GB**, 64 GB RAM, laptop-class i7.

This branch **experimented** with **3B target + 0.5B draft** (`docs/SPECULATIVE_DECODING.md`). Fair bench on 1660 Ti showed spec is ~3× slower; **production on this branch uses target-only** (`./scripts/up_llamacpp.sh` → `llama-target`, or Ollama `qwen2.5:3b` on default compose). **`main`** ships Ollama-only with no draft path.

There is **no** separate hardware-allocation script. Discord’s gateway stays up; weights load on first chat, then unload after `OLLAMA_KEEP_ALIVE` (Ollama path).

## 1. Host prep

```bash
sudo mkdir -p /var/lib/seki/ollama
sudo chown "$USER:$USER" /var/lib/seki/ollama
```

NVIDIA Container Toolkit must be installed so the Ollama service’s GPU reservation actually sees the 1660 Ti.

## 2. Boot gateway + Ollama

```bash
git checkout dev-optimize
cp .env.example .env
# set API_KEY
docker compose up --build -d
```

Wait until `GET http://localhost:9000/ready` is 200 with `"ollama": true` and `"configured_primary": "ollama"`. First start pulls `qwen2.5:3b`, `qwen2.5:0.5b`, and `nomic-embed-text`, then `ollama create seki-qwen-spec`.

## 3. Prove a completion

```bash
chmod +x scripts/verify_gateway.sh
./scripts/verify_gateway.sh
```

Expect `x-seki-backend: ollama`. Spec experiment (optional): `N=20 ./scripts/bench_speculative.sh` after `./scripts/up_llamacpp.sh`.

Live serial (warm model, short pings):

```bash
N=20 ./scripts/pressure_live.sh
```

Concurrent batches (does not go through Discord; this is raw GPU pile-up):

```bash
MODE=flash BATCH=16 python3 scripts/pressure_live_concurrent.py
MODE=waves WAVE=8 DURATION=20 INTERVAL=2 python3 scripts/pressure_live_concurrent.py
```

## 4. Parent mesh (Discord)

From `Production-grade` (sibling repos):

- Clone `seki-discord-bot`, `seki-agent-core`, `seki-control-plane`.
- Root `docker compose up --build` (this branch’s compose has no vLLM).
- Pin MiniLM off the GPU: `CUDA_VISIBLE_DEVICES=` on `seki-v2-agent-core`.
- Linux Docker: `extra_hosts: ["host.docker.internal:host-gateway"]` so the bot can reach the control plane.
- Do not deploy `seki-tts` or `seki-gui` on this card.

## 5. Idle vs active

| Piece | 24/7 |
|---|---|
| Ollama **daemon** | yes (small) |
| Chat GGUF weights | no — `OLLAMA_KEEP_ALIVE=5m` |
| vLLM | not on this branch |
| Discord / agent-core / inference gateway | yes (CPU/RAM) |

User talk and autonomous announce share the same completion path.

## 6. Optional voice QLoRA (still 6 GB)

`docs/QLORA_TRAINING.md` + `agent-core/training/train_unsloth.py` are 4-bit 3B on this card (batch 1). Do not train while Discord is generating. Overnight is expected. Export GGUF and `ollama create seki-qwen-3b`.

Capacity numbers (queue, mention rate, vs MEE6 / Midjourney): sibling repo `seki-discord-bot/docs/PRESSURE_TEST.md`. Live Ollama timings: `docs/LIVE_PERFORMANCE.md`.
