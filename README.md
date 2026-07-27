# Seki Discord

Control plane and Discord agent for **Seki Amahara** — authenticated Next.js dashboard, agent APIs, hybrid LLM routing, dual embeddings, and a fine-tuned local Qwen voice model.

## Stack

- **Control plane:** Next.js 14 (App Router), Auth.js, Prisma, PostgreSQL + pgvector
- **Agent core:** Python `discord.py` bot → `/api/v1/agent/*` with `x-agent-api-key`
- **LLMs:** NVIDIA NIM (primary) + local Ollama (fallback / fine-tuned Seki voice)
- **Embeddings:** Ollama `nomic-embed-text` (default) or OpenAI `text-embedding-3-small`

## Quick start

### 1. Control plane

```bash
cp .env.example .env
# set DATABASE_URL, AUTH secrets, SEKI_AGENT_API_KEY, optional NVIDIA/OpenAI keys
npm install
npx prisma migrate deploy
npx prisma db seed
npm run dev
```

### 2. Discord agent

```bash
cd agent-core
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# set DISCORD_TOKEN, SEKI_AGENT_API_KEY, CONTROL_PLANE_URL, channel IDs
python bot.py
```

### 3. Local Seki voice (Ollama)

After QLoRA export, place the `.gguf` next to the Modelfile (not committed — see `docs/QLORA_TRAINING.md`), then:

```bash
cd agent-core/seki-qwen-3b_gguf
ollama create seki-qwen-3b -f Modelfile
ollama run seki-qwen-3b
```

Point the bot local fallback / Ollama model name at `seki-qwen-3b` when ready.

## Training (optional)

See [docs/QLORA_TRAINING.md](docs/QLORA_TRAINING.md). On Windows, use:

```powershell
cd agent-core
powershell -ExecutionPolicy Bypass -File training/run_train_windows.ps1
```

## License

Private lab project.
