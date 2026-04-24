"""vLLM-based batch generation for ChemQA eval. Uses PagedAttention +
continuous batching to max out GPU memory and throughput.

Greedy decoding (temperature=0). LoRA adapters applied via vLLM's LoRA engine
when --adapter_dir is provided.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/data_lab/ChemQA/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Avoid vLLM's engine-side multi-process launcher grabbing stdio; use v0 engine
# mode for simpler single-process behaviour inside our launcher.
os.environ.setdefault("VLLM_USE_V1", "1")

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--num_samples", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_model_len", type=int, default=1024)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--lora_rank", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    Path(os.path.dirname(args.output_file)).mkdir(parents=True, exist_ok=True)

    # Load, shuffle deterministically, truncate
    records = [json.loads(l) for l in open(args.input_file)]
    import random
    random.Random(args.seed).shuffle(records)
    records = records[: args.num_samples]
    print(f"[data] n={len(records)} from {args.input_file}", flush=True)

    # Tokenizer for chat templating (vLLM also has a chat API but this keeps
    # identical templating to our HF eval)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in records
    ]

    llm_kwargs = dict(
        model=args.base_model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=False,
    )
    lora_req = None
    if args.adapter_dir:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.lora_rank
        lora_req = LoRARequest("adapter", 1, args.adapter_dir)

    print(f"[load] base={args.base_model} adapter={args.adapter_dir or '(none)'}", flush=True)
    llm = LLM(**llm_kwargs)

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=None,
    )

    t0 = time.time()
    outputs = llm.generate(
        prompts,
        sp,
        lora_request=lora_req,
        use_tqdm=True,
    )
    dt = time.time() - t0
    print(f"[gen] {len(outputs)} samples in {dt:.1f}s  ({len(outputs)/dt:.2f} ex/s)", flush=True)

    with open(args.output_file, "w") as fout:
        for r, o in zip(records, outputs):
            text = o.outputs[0].text
            fout.write(
                json.dumps(
                    {
                        "id": r.get("cid", r.get("sample_id", "")),
                        "prompt": r["prompt"],
                        "reference": r.get("response", r.get("reference", "")),
                        "prediction": text.strip(),
                        "topic": r.get("topic"),
                        "task": r.get("task"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[done] wrote -> {args.output_file}", flush=True)


if __name__ == "__main__":
    main()
