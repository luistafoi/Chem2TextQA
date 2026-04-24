# Fine-tuning on Chem2TextQA

Scripts and headline results from fine-tuning open base models on the
Chem2TextQA gold subset. Three base models tried: **Gemma 3 12B**,
**Llama 3.1 8B**, **Qwen 2.5 14B**.

## Headline results

Full holdout evaluation on the 26,205-compound test split (scaffold-split
`test`), scored with CIDEr via `pycocoevalcap`.

| Model | Variant | n | CIDEr |
|---|---|---|---|
| gemma3_12b | base | 26,205 | 0.0055 |
| gemma3_12b | **finetuned** | 26,205 | **0.2667** |
| llama3_1_8b | base | 26,205 | 0.0059 |
| llama3_1_8b | **finetuned** | 26,205 | **0.2387** |
| qwen2_5_14b | base | 26,205 | 0.0017 |
| qwen2_5_14b | **finetuned** | 26,205 | **0.2758** |

All three bases go from near-zero CIDEr on the held-out test set to
~0.24–0.28 after fine-tuning — ~45–160× improvement depending on base.

Full details in `eval/results_full/summary.md` and the per-model
`*.metrics.json` files.

## Contents

```
finetune/
├── scripts/                      training + setup
│   ├── finetune.py               the actual SFT training loop (TRL + FSDP)
│   ├── preprocess.py             converts dataset_gold.jsonl to SFT chat format
│   ├── download_models.py        pulls base weights from HF
│   ├── train_all.sh              full sweep: 3 models × 1 run each
│   ├── train_one.sh              single-model driver
│   ├── bs_probe.py               batch-size probe for a given GPU config
│   └── make_bundle.py            zips everything for transfer
│
└── eval/
    ├── scripts/                  evaluation pipeline
    │   ├── generate.py           run inference on the test split
    │   ├── generate_vllm.py      vLLM variant (faster)
    │   ├── score.py              compute CIDEr / BLEU / ROUGE
    │   ├── summarize.py          aggregate per-model results
    │   ├── prep_smolinstruct.py  optional: compare against SMolInstruct
    │   ├── run_all.sh            small-holdout eval driver
    │   ├── run_full.sh           full-holdout (26,205) eval driver
    │   ├── run_full_vllm.sh      full-holdout via vLLM
    │   ├── run_full_vllm_retry.sh  resumable variant
    │   └── run_smol_extras.sh    additional benchmarks
    │
    └── results_full/
        ├── summary.md            headline CIDEr table
        ├── gemma3_12b/*.metrics.json
        ├── llama3_1_8b/*.metrics.json
        └── qwen2_5_14b/*.metrics.json
```

## What's NOT in this directory (gitignored; kept locally)

- `finetune/data/` — the training data (comes from
  `data/qa_pipeline/full_premium_kimi/dataset_gold.jsonl`; 542 MB).
- `finetune/checkpoints/` — fine-tuned LoRA / full-FT weights
  (~260 MB per model).
- `finetune/eval/results_full/*/base_chemqa.jsonl` and
  `ft_chemqa.jsonl` — raw prediction outputs (~49 MB + 36 MB per model).
- `finetune/logs/` — training and eval logs.

If you need the fine-tuned checkpoints or raw prediction files, they
live at `/data/macaulay/ChemQA/ChemQA/` on the lab server. Ask Macaulay.

## Reproducing the fine-tune from scratch

1. Install deps (TRL, Accelerate, Transformers, pycocoevalcap, vLLM).
2. Point `scripts/preprocess.py` at
   `data/qa_pipeline/full_premium_kimi/dataset_gold.jsonl`; it emits
   chat-format SFT JSONL using the `split` field to partition into
   train/val/test.
3. Download base weights: `bash scripts/download_models.py`.
4. Train: `bash scripts/train_all.sh` (4×H100 FSDP default; see
   `train_one.sh` for single-model).
5. Evaluate: `bash eval/scripts/run_full.sh <model>` — produces
   `eval/results_full/<model>/{base,ft}_chemqa.jsonl` plus scored
   metrics.

Approximate cost per model: ~3–5 H100-hours for the fine-tune, plus
~2 hours for the full-holdout eval.

## Using the dataset for fine-tuning

See `../USAGE_FINETUNING.md` for the dataset schema, split semantics,
and recommended chat-format conversion. Everything in that guide
applies here.
