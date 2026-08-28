# seki-inference-engine

OpenAI-compatible FastAPI gateway in front of **Ollama** (GGUF on a 6 GB Turing card). Other Seki services do not import this package; they call `/v1/chat/completions` and `/v1/embeddings`.

This git branch is **`Low_ends_6GB`**. It does not ship vLLM, Windows Unsloth launchers, or TTS.

## What this solves (and why it matters)

A 6 GB card cannot keep a vLLM server resident and still leave headroom for Discord. This repo is the serving contract for that box: one Bearer-auth `/v1` API, one Ollama GGUF, and a header (`x-seki-backend`) that shows Ollama answered.

Give it a try if you:

- Need **Ollama-first serving on an RTX 1660 Ti** without a vLLM connect miss on every Discord turn
- Want weights to **unload after idle** (`OLLAMA_KEEP_ALIVE`) while the Discord gateway stays connected
- Want **Prometheus** on gateway wall time (`GET /metrics`)
- Are showing **ML platform / inference** work: health vs ready, timeouts vs application 4xx, CI that builds the image

It does **not** host the Discord personality or the memory graph. Those live in `seki-agent-core`. It does **not** run vLLM.

| Method | Path | Auth | Role |
|---|---|---|---|
| `GET` | `/health` | no | Process is up |
| `GET` | `/ready` | no | Ollama can list models |
| `GET` | `/metrics` | no | Prometheus scrape |
| `POST` | `/v1/chat/completions` | Bearer | Chat (streaming supported) |
| `POST` | `/v1/embeddings` | Bearer | Embeddings |

## Hardware and prerequisites

**RTX / GTX 1660 Ti (6 GB Turing)** or similar. 64 GB system RAM is ample; VRAM is the limit.

- One quantized **3B** chat GGUF (`qwen2.5:3b-instruct-q5_K_M` or `qwen2.5:3b`) plus `nomic-embed-text`
- [Docker Engine](https://docs.docker.com/engine/install/) + Compose v2
- NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Python 3.11+ only if you run the gateway without Docker

Do **not** co-locate Qwen3-TTS, Whisper, `llava:7b`, or a second LLM on this GPU.

Optional overnight fine-tune: 4-bit QLoRA of Qwen2.5-3B (`docs/QLORA_TRAINING.md`). Stop Discord generation first.

## Fresh install

```bash
git clone https://github.com/PCMakia/seki-inference-engine.git
cd seki-inference-engine
git checkout Low_ends_6GB
cp .env.example .env
# set API_KEY
sudo mkdir -p /var/lib/seki/ollama
sudo chown "$USER:$USER" /var/lib/seki/ollama
docker compose up --build -d
```

Wait until `GET http://localhost:9000/ready` is 200. Completions should show `x-seki-backend: ollama`.

Host ports (so a v1 stack can keep 8000 / 11434): gateway **9000**, Ollama **9114**. Inside the mesh, containers still use 8000 / 11434.

**Without Docker** (Ollama already on the host):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## How to use it

1. Set `API_KEY` in `.env`. Clients send `Authorization: Bearer <API_KEY>`.
2. Prove a completion:

```bash
chmod +x scripts/verify_gateway.sh
./scripts/verify_gateway.sh
```

3. Chat:

```bash
curl -sS http://localhost:9000/v1/chat/completions \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

4. Scrape `http://localhost:9000/metrics`.
5. Point `seki-agent-core` at `INFERENCE_URL=http://localhost:9000/v1` on the host, or `http://seki-v2-inference:8000/v1` on the Compose network.

Weights unload after `OLLAMA_KEEP_ALIVE` (default 5m). Discord stays connected. Full host checklist: `docs/UBUNTU_6GB.md`.
