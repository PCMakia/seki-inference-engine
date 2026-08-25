"""Generate Alpaca-style synthetic Seki voice data via NVIDIA NIM.

Option A pipeline (Phase 7): concept-chain → Seki Amahara voice pairs.
Target: 1,000–2,000 JSONL rows for QLoRA fine-tuning.

Usage:
  set NVIDIA_API_KEY=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
  python -m training.generate_dataset
  # or: python training/generate_dataset.py

Optional env:
  DATASET_TARGET=1000
  DATASET_CONCURRENCY=4
  DATASET_OUTPUT=training/synthetic_seki_data.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

logger = logging.getLogger("seki.training.generate_dataset")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_TARGET = 1000
DEFAULT_CONCURRENCY = 4
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "synthetic_seki_data.jsonl"

SYSTEM_PROMPT = (
    "You are a dataset writer producing training examples for an AI VTuber named "
    "Seki Amahara. Reply with ONLY the spoken/chat response text — no labels, "
    "no JSON, no meta commentary."
)

USER_PROMPT_TEMPLATE = (
    "Given the following conceptual memory chain, write a response precisely in "
    "the voice of Seki Amahara (AI VTuber). Include typical mannerisms and emojis.\n\n"
    "Concept Chain:\n{concept_chain}"
)

MOODS = [
    "Calm",
    "Curious",
    "Playful",
    "Wistful",
    "Focused",
    "Sleepy",
    "Mischievous",
    "Warm",
]

DESIRES = [
    "Quiet observation",
    "Gentle teasing",
    "Deep reflection",
    "Soft encouragement",
    "Chaotic curiosity",
    "Late-night pondering",
    "Sharing a tiny secret",
    "Watching the chat glow",
]

INTENTS = [
    "Listen",
    "Reply softly",
    "Ask a small question",
    "Offer tea-adjacent comfort",
    "Note a pattern",
    "Spark a concept chain",
    "Reassure without smothering",
    "Drift into metaphor",
]

TOPICS = [
    "discord midnight chat",
    "streaming nerves",
    "memory shards",
    "fan art",
    "rain on the window",
    "new song fragments",
    "forgotten tabs",
    "quiet friend energy",
    "pixel dreams",
    "voice crack humor",
]


def _build_concept_chain(index: int) -> str:
    mood = random.choice(MOODS)
    desire = random.choice(DESIRES)
    intent = random.choice(INTENTS)
    topic = random.choice(TOPICS)
    return (
        f"[Seed:{index}] -> [Mood: {mood}] -> [Desire: {desire}] -> "
        f"[Intent: {intent}] -> [Topic: {topic}]"
    )


async def _generate_one(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    concept_chain: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, str]:
    payload: dict[str, Any] = {
        "model": NVIDIA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(concept_chain=concept_chain),
            },
        ],
        "temperature": 0.85,
        "max_tokens": 320,
    }

    max_retries = 5

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    f"{NVIDIA_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60.0,
                )
                
                # Check for rate limit BEFORE raising other errors
                if response.status_code == 429:
                    print(f"429 Rate limit hit. Waiting 5s before retry (attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(5.0)
                    continue
                    
                response.raise_for_status()
                
                # SUCCESS! Pace the requests to stay under 40 RPM
                await asyncio.sleep(1.6)

                data = response.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                text = (content or "").strip()
                if not text:
                    raise RuntimeError("Empty completion from NVIDIA NIM")

                return {
                    "instruction": f"Concept Chain: {concept_chain}",
                    "output": text,
                }
                
            except httpx.HTTPStatusError as err:
                # Catch it here as well just in case raise_for_status triggers it
                if err.response.status_code == 429:
                    print(f"429 Rate limit hit. Waiting 5s before retry...")
                    await asyncio.sleep(5.0)
                    continue
                # If it's a 401 Unauthorized or 500 Server Error, crash immediately
                raise err
                
        # If we loop 5 times and get 429s every time, fail out
        raise RuntimeError(f"Failed to generate after {max_retries} attempts due to NVIDIA API rate limits.")


async def generate_dataset(
    *,
    target: int,
    concurrency: int,
    output_path: Path,
    api_key: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    written = 0

    async with httpx.AsyncClient() as client:
        with output_path.open("w", encoding="utf-8") as handle:
            batch_size = max(1, concurrency * 2)
            next_index = 0
            while written < target:
                batch_n = min(batch_size, target - written)
                chains = [
                    _build_concept_chain(next_index + offset)
                    for offset in range(batch_n)
                ]
                next_index += batch_n
                
                results = await asyncio.gather(
                    *[
                        _generate_one(
                            client,
                            api_key=api_key,
                            concept_chain=chain,
                            semaphore=semaphore,
                        )
                        for chain in chains
                    ],
                    return_exceptions=True,
                )

                for result in results:
                    if isinstance(result, BaseException):
                        logger.warning("Generation failed: %s", result)
                        continue
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    written += 1
                    if written % 25 == 0 or written == target:
                        logger.info("Wrote %s / %s rows", written, target)
                    
                
                
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Seki Alpaca JSONL via NVIDIA NIM",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=int(os.getenv("DATASET_TARGET", str(DEFAULT_TARGET))),
        help="Number of JSONL rows to generate (default 1000)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("DATASET_CONCURRENCY", str(DEFAULT_CONCURRENCY))),
        help="Max in-flight NIM requests",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv("DATASET_OUTPUT", str(DEFAULT_OUTPUT))),
        help="Output JSONL path",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    args = parse_args()
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "NVIDIA_API_KEY is required to generate the synthetic dataset.",
        )

    if args.target < 1:
        raise SystemExit("--target must be >= 1")

    written = asyncio.run(
        generate_dataset(
            target=args.target,
            concurrency=args.concurrency,
            output_path=args.output,
            api_key=api_key,
        ),
    )
    logger.info("Done. Wrote %s rows to %s", written, args.output)


if __name__ == "__main__":
    main()
