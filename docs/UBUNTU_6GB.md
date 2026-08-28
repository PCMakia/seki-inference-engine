# Ubuntu 6 GB (Low_ends_6GB)

Target: Ubuntu + NVIDIA driver + Container Toolkit, **RTX 1660 Ti 6 GB**, 64 GB RAM, laptop-class i7.

This branch runs **Ollama only**. There is no vLLM service and no `--profile gpu`. vLLM OOMs Turing 6 GB and fights the chat GGUF for VRAM.

There is **no** separate hardware-allocation script. Discord’s gateway stays up; Ollama loads GGUF weights on the first chat or 2-hour announce, then unloads after `OLLAMA_KEEP_ALIVE`.

## 1. Host prep

```bash
sudo mkdir -p /var/lib/seki/ollama
sudo chown "$USER:$USER" /var/lib/seki/ollama
```

NVIDIA Container Toolkit must be installed so the Ollama service’s GPU reservation actually sees the 1660 Ti.

## 2. Boot gateway + Ollama

```bash
git checkout Low_ends_6GB
cp .env.example .env
# set API_KEY
docker compose up --build -d
```

Wait until `GET http://localhost:9000/ready` is 200 with `"ollama": true` and `"configured_primary": "ollama"`. First start pulls `qwen2.5:3b-instruct-q5_K_M` and `nomic-embed-text`.

If that Q5 tag is missing from the library, set `OLLAMA_MODEL=qwen2.5:3b` and matching `OLLAMA_PULL_MODELS`.

## 3. Prove a completion

```bash
chmod +x scripts/verify_gateway.sh
./scripts/verify_gateway.sh
```

Expect `x-seki-backend: ollama`.

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
