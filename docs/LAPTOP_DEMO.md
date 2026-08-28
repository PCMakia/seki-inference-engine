# 6 GB demo (Ollama only)

Target: a **6 GB Turing** card (RTX / GTX 1660 Ti). This branch never starts vLLM.

Ubuntu server walkthrough: `docs/UBUNTU_6GB.md`.

## 1. Boot gateway + Ollama

```bash
cp .env.example .env
docker compose up --build
```

Wait until `docker compose ps` shows `inference` as healthy. First Ollama start may pull `qwen2.5:3b-instruct-q5_K_M` and `nomic-embed-text`.

## 2. Record a successful completion

```bash
chmod +x scripts/verify_gateway.sh
./scripts/verify_gateway.sh
```

On Windows PowerShell you can still run `scripts/verify_gateway.ps1`.

You want:

- `/health` → `{"status":"ok"}`
- `/ready` → `"ollama": true`, `"configured_primary": "ollama"`
- `POST /v1/chat/completions` → HTTP 200 and header **`x-seki-backend: ollama`**

## 3. Full mesh (agent + Discord)

From `Production-grade` (parent folder), `docker compose up --build` with **no** GPU/vLLM profile. Pin MiniLM off the card (`CUDA_VISIBLE_DEVICES=` on agent-core). Do not deploy TTS on this GPU.
