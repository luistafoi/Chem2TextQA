"""Generate responses for an eval set, optionally applying a LoRA adapter.

Usage:
  python generate.py \
    --base_model Qwen/Qwen2.5-14B-Instruct \
    --adapter_dir /path/to/final \            # optional; omit for base-model eval
    --input_file /path/to/test.jsonl \        # jsonl with {prompt, response}
    --output_file /path/to/preds.jsonl \
    --num_samples 500 \
    --max_new_tokens 256 \
    --gpu 0
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max_seq_len", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = f"cuda:{args.gpu}"
    os.environ.setdefault("HF_HOME", "/mnt/data_lab/ChemQA/hf_cache")

    # Load examples and take a reproducible subsample
    records = [json.loads(l) for l in open(args.input_file)]
    import random
    random.Random(args.seed).shuffle(records)
    records = records[: args.num_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # needed for causal batch generation

    print(f"[load] base={args.base_model} adapter={args.adapter_dir or '(none)'}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device,
    )
    model.config.use_cache = True
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        # Merge for faster inference if possible; skip on multimodal archs that
        # can refuse to merge cleanly.
        try:
            model = model.merge_and_unload()
            print("[load] adapter merged", flush=True)
        except Exception as e:
            print(f"[load] merge skipped: {e}", flush=True)
    model.eval()

    Path(os.path.dirname(args.output_file)).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(args.output_file, "w") as fout:
        for i in range(0, len(records), args.batch_size):
            batch = records[i : i + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": r["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for r in batch
            ]
            enc = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=args.max_seq_len,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            in_len = enc["input_ids"].shape[1]
            gen = out[:, in_len:]
            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for r, t in zip(batch, texts):
                fout.write(
                    json.dumps(
                        {
                            "id": r.get("cid", r.get("sample_id", "")),
                            "prompt": r["prompt"],
                            "reference": r.get("response", r.get("reference", "")),
                            "prediction": t.strip(),
                            "topic": r.get("topic"),
                            "task": r.get("task"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            fout.flush()
            dt = time.time() - t0
            done = i + len(batch)
            print(
                f"[gen] {done}/{len(records)}  {dt:.1f}s  {done/max(dt,1):.2f} ex/s",
                flush=True,
            )
    print(f"[done] wrote -> {args.output_file} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
