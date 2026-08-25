# seki-inference-engine

Model serving, routing, and training pipeline for the Seki stack.

Extracted from:

- [Personal-Assistant](https://github.com/PCMakia/Personal-Assistant) — Ollama client, Docker Ollama worker
- `Seki-Discord` — hybrid NVIDIA NIM → Ollama router, Unsloth QLoRA / GGUF export

## What this repo is today

- OpenAI-compatible client wrapper over local Ollama (`src/LLM_handler/llm_client.py`)
- Hybrid primary/fallback router (`agent-core/core/llm_router.py`)
- QLoRA dataset generation + Unsloth train + GGUF Modelfile (`agent-core/training/`, `docs/QLORA_TRAINING.md`)
- Compose seed for an Ollama container (`docker-compose.yml`)

## Target (not implemented yet)

A standalone serving microservice with:

- vLLM / Triton Inference Server / ONNX Runtime backends
- A stable HTTP contract consumed by `seki-agent-core`
- Optional PySpark / Kafka ingestion worker for embedding backfill and training traces

Do not import this package from other repos. Other services should call an HTTP API once the gateway lands.

## Training

See [docs/QLORA_TRAINING.md](docs/QLORA_TRAINING.md).
