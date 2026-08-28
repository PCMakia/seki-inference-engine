# Merging dev-optimize into Low_ends_6GB

Quick operator guide after the speculative decoding experiment. Full numbers: `docs/SPEC_EXPERIMENT_REPORT.md`.

## What to use in production (1660 Ti 6 GB)

| Component | Setting |
|-----------|---------|
| Chat model | **`qwen2.5:3b`** (target-only) |
| Default compose | `docker compose up` — `OLLAMA_SPECULATIVE=0` |
| Optional llama.cpp | `./scripts/up_llamacpp.sh` → **`llama-target`** on `:9082` |
| Do **not** use | `llama-spec`, `seki-qwen-spec` for Discord speed |

## Fresh deploy after merge

```bash
git checkout Low_ends_6GB
git pull
cp .env.example .env
# API_KEY=...
docker compose up --build -d
./scripts/verify_gateway.sh
```

## Optional: llama.cpp instead of Ollama GPU chat

When Ollama `DRAFT` fails or you want llama.cpp serving:

```bash
./scripts/up_llamacpp.sh
# Gateway -> llama-target (3B only). Embeddings need: ./scripts/up_llamacpp.sh down
```

## Optional: re-run the spec experiment

```bash
./scripts/up_llamacpp.sh   # target-only production
N=20 ./scripts/bench_speculative.sh   # sequential fair A/B; expect ~0.37× on 1660 Ti
```

Enable Ollama spec attempt only for debugging:

```bash
OLLAMA_SPECULATIVE=1 docker compose up -d --force-recreate ollama inference
```

## Discord / agent-core

```bash
INFERENCE_URL=http://localhost:9000/v1
# Model in requests can be omitted; gateway defaults to qwen2.5:3b
```

## What was intentionally excluded from merge

- `tests/test_ollama_manifest.py` — manifest helper tests only; helper kept if needed by create script
- Spec as default (`OLLAMA_MODEL=seki-qwen-spec`) — failed speed goal on this GPU
