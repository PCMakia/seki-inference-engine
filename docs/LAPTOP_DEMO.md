# Laptop demo (no vLLM)

Target machine: Windows + Docker Desktop, ~6 GB GPU (e.g. GTX 1660 Ti).
Do **not** use `--profile gpu` on that box.

## 1. Boot gateway + Ollama

From `seki-inference-engine`:

```powershell
Copy-Item .env.example .env -ErrorAction SilentlyContinue
docker compose up --build
```

Wait until `docker compose ps` shows `inference` as healthy. First Ollama start may pull `qwen2.5:3b` and `nomic-embed-text` (several minutes).

Default `docker compose up` does **not** start `seki-v2-vllm`.

## 2. Record a successful completion (failover path)

```powershell
powershell -File scripts/verify_gateway.ps1
```

You want:

- `/health` → `{"status":"ok"}`
- `/ready` → `"ollama": true` (vLLM may be `false`)
- `POST /v1/chat/completions` → HTTP 200 and header **`x-seki-backend: ollama`**

That header is the failover proof on a laptop: vLLM is not running, so the gateway connects to Ollama.

## 3. GPU box only — primary then failover

```powershell
docker compose --profile gpu up --build
# ... wait until /ready has "vllm": true ...
# completion header should be x-seki-backend: vllm

docker stop seki-v2-vllm
# same curl — header must become x-seki-backend: ollama
```

## 4. Full mesh (agent + Discord)

From `Production-grade` (parent folder):

```powershell
Copy-Item .env.example .env -ErrorAction SilentlyContinue
docker compose up --build
# GPU:
docker compose --profile gpu up --build
```

Agent-core is on host port **9080**. GUI: `AGENT_BASE_URL=http://127.0.0.1:9080`.
Gateway is on **9000**; Ollama host publish is **9114**.
