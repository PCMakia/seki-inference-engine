"""QLoRA fine-tune Qwen2.5-3B-Instruct on synthetic Seki voice data (Unsloth).

Requires a CUDA GPU (1660 Ti class is supported with 4-bit QLoRA; keep batch
size small). See docs/QLORA_TRAINING.md for the full runbook.

Usage (from a GPU-enabled Python env with Unsloth installed):
  python training/train_unsloth.py
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

# Unsloth MUST be imported before trl — otherwise its SFTTrainer patch leaves
# eos_token as the placeholder "<EOS_TOKEN>", which is not in Qwen's vocab.
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DATA_PATH = Path(__file__).resolve().parent / "synthetic_seki_data.jsonl"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
QWEN_EOS_TOKEN = "<|im_end|>"


def configure_qwen_tokenizer(tokenizer):  # type: ignore[no-untyped-def]
    """Map Unsloth's <EOS_TOKEN> placeholder to a real Qwen2 vocab token."""
    eos_id = tokenizer.convert_tokens_to_ids(QWEN_EOS_TOKEN)
    if eos_id is None or eos_id == tokenizer.unk_token_id:
        raise ValueError(
            f"{QWEN_EOS_TOKEN!r} is missing from the tokenizer vocabulary.",
        )
    tokenizer.eos_token = QWEN_EOS_TOKEN
    tokenizer.pad_token = QWEN_EOS_TOKEN
    tokenizer.eos_token_id = eos_id
    tokenizer.pad_token_id = eos_id
    return tokenizer


def load_alpaca_jsonl(path: Path) -> Dataset:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            instruction = str(item.get("instruction", "")).strip()
            output = str(item.get("output", "")).strip()
            if not instruction or not output:
                continue
            rows.append(
                {
                    "text": (
                        f"### Instruction:\n{instruction}\n\n"
                        f"### Response:\n{output}"
                    ),
                },
            )
    if not rows:
        raise FileNotFoundError(
            f"No usable Alpaca rows in {path}. Run generate_dataset.py first.",
        )
    return Dataset.from_list(rows)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset at {DATA_PATH}. "
            "Run: python training/generate_dataset.py",
        )

    dataset = load_alpaca_jsonl(DATA_PATH)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    tokenizer = configure_qwen_tokenizer(tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    # Re-assert after PEFT wrap in case Unsloth patching rewrote special tokens.
    tokenizer = configure_qwen_tokenizer(tokenizer)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            packing=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=20,
            num_train_epochs=2,
            learning_rate=2e-4,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs-seki-qwen-3b",
            report_to="none",
            eos_token=QWEN_EOS_TOKEN,
            pad_token=QWEN_EOS_TOKEN,
        ),
    )

    # Prefer response-only loss when chat template helpers are available.
    try:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="### Instruction:\n",
            response_part="### Response:\n",
        )
    except Exception:
        pass

    trainer.train()

    # Export GGUF (q5_k_m) for local Ollama / llama.cpp inference.
    model.save_pretrained_gguf(
        "seki-qwen-3b",
        tokenizer,
        quantization_method="q5_k_m",
    )
    print("Training complete. GGUF exported to ./seki-qwen-3b")


if __name__ == "__main__":
    main()
