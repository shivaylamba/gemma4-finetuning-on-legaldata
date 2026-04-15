"""
Fine-tune Gemma on a JSONL legal dataset using HF transformers + PEFT + TRL.
No unsloth. Assumes a single H100 (80GB) but works on any >=24GB card with
4-bit loading.

Input format: one JSON object per line. This script looks for a `text` field
(the format produced by scrape_legislation.py). If your records use a
different key, pass --text_field.
"""

import argparse
import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", required=True)
    p.add_argument("--data_path", required=True, help="path to .jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--text_field", default="text")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--no_4bit", dest="load_in_4bit", action="store_false")
    return p.parse_args()


def build_dataset(path: str, text_field: str):
    ds = load_dataset("json", data_files=path, split="train")
    # Keep only non-empty text rows. scrape_legislation.py produced some
    # metadata-only rows with empty `text` — drop those.
    if text_field not in ds.column_names:
        raise ValueError(
            f"field '{text_field}' not in dataset columns {ds.column_names}"
        )
    ds = ds.filter(lambda r: isinstance(r[text_field], str) and len(r[text_field]) > 50)

    # Rename to 'text' so SFTTrainer's dataset_text_field='text' just works.
    if text_field != "text":
        ds = ds.rename_column(text_field, "text")

    # Keep only the text column — SFTTrainer does not need the rest.
    drop = [c for c in ds.column_names if c != "text"]
    if drop:
        ds = ds.remove_columns(drop)
    print(f"dataset rows after filtering: {len(ds)}")
    return ds


def main():
    args = parse_args()

    # ---- tokenizer ----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- model (4-bit, nf4) -------------------------------------------------
    bnb = None
    if args.load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except Exception:
        attn_impl = "sdpa"
    print(f"attention impl: {attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # ---- LoRA ---------------------------------------------------------------
    # Target the attention + MLP linear layers. Names below are the Gemma
    # family convention and also match most Llama-style models.
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # ---- data ---------------------------------------------------------------
    train_ds = build_dataset(args.data_path, args.text_field)

    # ---- trainer ------------------------------------------------------------
    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        max_seq_length=args.max_seq_length,
        packing=True,              # pack multiple samples per seq — good for long docs
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=sft_cfg,
        peft_config=lora,
    )

    trainer.train()

    # ---- save ---------------------------------------------------------------
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved adapters + tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
