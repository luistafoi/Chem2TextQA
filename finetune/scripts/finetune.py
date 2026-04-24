"""LoRA SFT for ChemQA, following LlaSMol (arXiv:2402.09391) hyperparameters.

LoRA:        r=16, alpha=16, dropout=0.05, targets={q,k,v,o,gate,down,up}_proj
Optim:       adamw_bnb_8bit, lr=1e-4, cosine schedule
Epochs:      3
Global BS:   512 (per-device 16 x 8 GPUs x grad_accum 4)
Max seq len: 512 (paper: covers 99.7% of samples)
Warmup:      ratio 0.05 (paper used 1000 steps over ~19k; we scale to our ~790 steps)
Precision:   bf16

Training is single-process-per-GPU DDP via torchrun. We load the base model
without quantization to maximize use of B200 HBM and keep LoRA updates in bf16.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

LLASMOL_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]


@dataclass
class Args:
    model_name: str
    output_dir: str
    run_name: str
    train_file: str = "/mnt/data_lab/ChemQA/data/processed/train.jsonl"
    eval_file: str = "/mnt/data_lab/ChemQA/data/processed/val.jsonl"
    max_seq_len: int = 512
    per_device_batch_size: int = 16
    grad_accum_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    seed: int = 42
    eval_samples: int = 1000
    grad_ckpt: bool = True
    ddp_find_unused: bool = False


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--per_device_batch_size", type=int, default=16)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--eval_samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_ckpt", type=int, default=1, help="1=enable gradient checkpointing")
    p.add_argument("--ddp_find_unused", type=int, default=0, help="1=set DDP find_unused_parameters (needed for multimodal heads)")
    ns = p.parse_args()
    d = vars(ns)
    d["grad_ckpt"] = bool(d["grad_ckpt"])
    d["ddp_find_unused"] = bool(d["ddp_find_unused"])
    return Args(**d)


def build_dataset(tokenizer, train_file: str, eval_file: str, max_len: int, eval_samples: int):
    raw = load_dataset(
        "json",
        data_files={"train": train_file, "eval": eval_file},
    )
    if eval_samples and eval_samples < len(raw["eval"]):
        raw["eval"] = raw["eval"].shuffle(seed=0).select(range(eval_samples))

    eos = tokenizer.eos_token or ""

    def format_and_tokenize(example):
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
        # Render to strings first, then tokenize with plain __call__. In
        # transformers 5.x, apply_chat_template(tokenize=True) returns a
        # BatchEncoding, which complicates length math; strings are simpler.
        prompt_text = tokenizer.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if eos and not full_text.endswith(eos):
            full_text = full_text + eos
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        full_ids = full_ids[:max_len]
        labels = list(full_ids)
        # Mask the prompt region so loss is computed only on the assistant response
        n_mask = min(len(prompt_ids), len(labels))
        for i in range(n_mask):
            labels[i] = -100
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    cols = raw["train"].column_names
    tokenized = raw.map(
        format_and_tokenize,
        remove_columns=cols,
        num_proc=8,
        desc="tokenize",
    )
    return tokenized


class DataCollatorSFT:
    def __init__(self, pad_id: int, max_len: int):
        self.pad_id = pad_id
        self.max_len = max_len

    def __call__(self, features):
        max_len = min(self.max_len, max(len(f["input_ids"]) for f in features))
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            ids = f["input_ids"][:max_len]
            lbl = f["labels"][:max_len]
            am = f["attention_mask"][:max_len]
            pad = max_len - len(ids)
            batch["input_ids"].append(ids + [self.pad_id] * pad)
            batch["attention_mask"].append(am + [0] * pad)
            batch["labels"].append(lbl + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def main():
    args = parse_args()
    set_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0

    if is_main:
        print(f"[args] {args}")
        print(f"[world] local_rank={local_rank} world_size={world_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if args.grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Only wire LoRA on modules actually present in the model (some archs don't
    # have all of {gate_proj, up_proj, down_proj} named identically).
    present = set()
    for n, _ in model.named_modules():
        leaf = n.rsplit(".", 1)[-1]
        if leaf in LLASMOL_TARGETS:
            present.add(leaf)
    target_modules = sorted(present) or LLASMOL_TARGETS
    if is_main:
        print(f"[lora] target_modules={target_modules}")

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)
    if is_main:
        model.print_trainable_parameters()

    ds = build_dataset(tokenizer, args.train_file, args.eval_file, args.max_seq_len, args.eval_samples)
    collator = DataCollatorSFT(pad_id=tokenizer.pad_token_id, max_len=args.max_seq_len)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim="adamw_bnb_8bit",
        bf16=True,
        tf32=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to=["tensorboard"],
        logging_dir=os.path.join(args.output_dir, "tb"),
        ddp_find_unused_parameters=args.ddp_find_unused,
        gradient_checkpointing=args.grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.grad_ckpt else None,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save final LoRA adapter and tokenizer from rank 0
    if is_main:
        final_dir = os.path.join(args.output_dir, "final")
        Path(final_dir).mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        with open(os.path.join(final_dir, "training_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"[done] adapter saved -> {final_dir}")


if __name__ == "__main__":
    main()
