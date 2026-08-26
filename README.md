# seki-inference-engine

Production FastAPI gateway for LLM inference and embeddings.

The service exposes an OpenAI-compatible HTTP API. **vLLM** is the primary
backend. Requests fail over to **local Ollama** only when vLLM raises a
timeout or connection error. Other vLLM HTTP errors (4xx/5xx) are returned
to the client unchanged.

Do not import this package from other repos. Call the HTTP API.

## Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | no | Process liveness |
| `GET` | `/ready` | no | Ready when vLLM **or** Ollama answers `/models` |
| `POST` | `/v1/chat/completions` | Bearer | OpenAI Chat Completions (streaming supported) |
| `POST` | `/v1/embeddings` | Bearer | OpenAI Embeddings |

Successful `/v1` responses include `x-seki-backend: vllm` or `x-seki-backend: ollama`.

## Configuration

Copy `.env.example` to `.env`. Required / important variables:

| Variable | Purpose |
|---|---|
| `API_KEY` | Bearer token for `/v1/*` |
| `MODEL_NAME` | Default chat model (vLLM served name / Hugging Face id) |
| `EMBEDDING_MODEL_NAME` | Default embedding model sent to vLLM |
| `REQUEST_TIMEOUT` | Upstream timeout in seconds (vLLM and Ollama) |
| `VLLM_BASE_URL` | Primary OpenAI-compatible base, e.g. `http://localhost:8001/v1` |
| `OLLAMA_BASE_URL` | Fallback OpenAI-compatible base, e.g. `http://localhost:11434/v1` |
| `OLLAMA_MODEL` | Chat model used on failover |
| `OLLAMA_EMBEDDING_MODEL` | Embedding model used on failover |

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env   # set API_KEY and MODEL_NAME

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

```bash
curl -s http://localhost:8000/health

curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-3B-Instruct","messages":[{"role":"user","content":"hello"}]}'
```

## Docker Compose

**Laptop (default):** gateway + Ollama only. vLLM is *not* started — a GTX 1660 Ti (6 GB) will typically OOM if both engines load. The gateway still tries vLLM first; the connection fails and traffic fails over to Ollama (`x-seki-backend: ollama`).

```bash
cp .env.example .env   # API_KEY=change-me is fine for local
docker compose up --build
```

**GPU box:** add the `gpu` profile (vLLM on host port `8001`).

```bash
docker compose --profile gpu up --build
```

Ollama has no GPU reservation so it does not fight vLLM for VRAM.

Prove the contract (from the host, after `/ready` is 200):

```powershell
# liveness / readiness
curl.exe -s http://localhost:8000/health
curl.exe -s -w "`nHTTP %{http_code}`n" http://localhost:8000/ready

# one completion — laptop should print x-seki-backend: ollama
curl.exe -sD - http://localhost:8000/v1/chat/completions `
  -H "Authorization: Bearer change-me" `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"qwen2.5:3b\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in one word.\"}]}"
```

Or: `powershell -File scripts/verify_gateway.ps1`

GPU failover check: `docker stop seki-vllm`, then the same curl — header must be `x-seki-backend: ollama`. Start vLLM again with `docker compose --profile gpu up -d vllm`.

See [docs/LAPTOP_DEMO.md](docs/LAPTOP_DEMO.md).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Training (optional)

QLoRA / GGUF export scripts remain under `agent-core/training/`. They are not
part of the serving path. See [docs/QLORA_TRAINING.md](docs/QLORA_TRAINING.md).
