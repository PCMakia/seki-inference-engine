# seki-inference-engine

OpenAI-compatible FastAPI gateway that sits in front of **vLLM** (primary) and **Ollama** (failover). Other Seki services do not import this package; they call `/v1/chat/completions` and `/v1/embeddings`.

## What this solves (and why it matters)

Local stacks often hard-code one runtime. This repo is the serving contract: one Bearer-auth `/v1` API, one model name, and a header (`x-seki-backend`) that tells you whether the GPU engine or the laptop fallback answered.

Give it a try if you:

- Need **vLLM in production shape** with a documented failover instead of a crash when CUDA is busy
- Want **Prometheus** on gateway wall time, backend choice, and failover count (`GET /metrics`)
- Are showing **ML platform / inference** work: health vs ready, timeouts vs application 4xx, CI that builds the image

It does **not** host the Discord personality or the memory graph. Those live in `seki-agent-core`.

| Method | Path | Auth | Role |
|---|---|---|---|
| `GET` | `/health` | no | Process is up |
| `GET` | `/ready` | no | vLLM **or** Ollama can list models |
| `GET` | `/metrics` | no | Prometheus scrape |
| `POST` | `/v1/chat/completions` | Bearer | Chat (streaming supported) |
| `POST` | `/v1/embeddings` | Bearer | Embeddings |

## Hardware and prerequisites

**Laptop / demo (no vLLM)**  
CPU or any NVIDIA GPU that can run Ollama. A 6 GB card (for example GTX 1660 Ti) is enough for `qwen2.5:3b` in Ollama. Default Compose **does not** start vLLM.

**GPU box (vLLM primary)**  
- NVIDIA GPU with driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- About **11 GB free VRAM** for `Qwen/Qwen2.5-3B-Instruct` at `--gpu-memory-utilization 0.70` and `--max-model-len 4096` (tuned for RTX 5070 Ti-class 16 GB cards that already have a desktop / old container using ~1 GB)
- Docker Desktop or Docker Engine with Compose v2
- Hugging Face token if the model gated (`HF_TOKEN`)
- Optional host cache: `HF_CACHE_PATH` (Windows example `D:/data/huggingface`)

**Install before first run**

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Engine + Compose v2)
- [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx) if you use the `gpu` profile
- [Ollama](https://ollama.com/download) only if you run the gateway with `uvicorn` against a host Ollama; Compose already starts `seki-v2-ollama`
- Python 3.11+ only if you run the gateway without Docker

On Docker Desktop (WSL2), vLLM’s V2 runner needs `VLLM_WSL2_ENABLE_PIN_MEMORY=1` or it exits with `UVA is not available`. Compose sets that by default.

## Fresh install

```powershell
git clone https://github.com/PCMakia/seki-inference-engine.git
cd seki-inference-engine
Copy-Item .env.example .env
# set API_KEY, and HF_TOKEN if you will start vLLM
```

**Compose, laptop (Ollama only):**

```powershell
docker compose up --build
```

Wait until `GET http://localhost:9000/ready` is 200. Completions should show `x-seki-backend: ollama`.

**Compose, GPU box:**

```powershell
docker compose --profile gpu up --build
```

Host ports (so a v1 stack can keep 8000 / 11434): gateway **9000**, vLLM debug **9001**, Ollama **9114**. Inside the mesh, containers still use 8000 / 11434.

**Without Docker:**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-dev.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Point `VLLM_BASE_URL` / `OLLAMA_BASE_URL` at whatever is actually running.

## How to use it

1. Set `API_KEY` in `.env`. Clients send `Authorization: Bearer <API_KEY>`.
2. Chat:

```powershell
curl.exe -s http://localhost:9000/v1/chat/completions `
  -H "Authorization: Bearer change-me" `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"Qwen/Qwen2.5-3B-Instruct\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}"
```

3. Confirm the backend: response header `x-seki-backend` is `vllm` or `ollama`.
4. Scrape `http://localhost:9000/metrics` (or `:8000` if you used uvicorn).
5. From this repo: `powershell -File scripts/verify_gateway.ps1` after `/ready` is 200.

`seki-agent-core` should use `INFERENCE_URL=http://localhost:9000/v1` on the host, or `http://seki-v2-inference:8000/v1` on the Compose network.

Failover: stop vLLM (`docker stop seki-v2-vllm`); the same curl should keep working with `x-seki-backend: ollama`. Laptop walkthrough: `docs/LAPTOP_DEMO.md`.
