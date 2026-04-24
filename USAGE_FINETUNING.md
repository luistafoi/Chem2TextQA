# Chem2TextQA — fine-tuning usage guide

For the lab-mate running fine-tunes. Everything you need to load the
data, respect the splits, and avoid common pitfalls.

## TL;DR

- Use `dataset_gold.jsonl` for training. It's the high-agreement subset
  (Phase 1 and Phase 2 LLMs independently produced the same answer).
- Each row is **one compound** with a nested array of Q&A pairs.
- Filter by the `split` field: `train` / `val` / `test`. Scaffold-based
  (MoleculeNet standard). Already assigned — don't re-split.
- Canary is a **separate file** (`canary/dataset_final.jsonl`) — use it
  only for held-out contamination stress-tests, never for training.

## Files

All paths relative to `data/qa_pipeline/full_premium_kimi/`.

| File | Compounds | Q&A | Purpose |
|---|---|---|---|
| `dataset_gold.jsonl` | 15,509 | ~188K | **Use this for fine-tuning.** Only Q&A where Phase 1 and Phase 2 agreed. |
| `dataset_final.jsonl` | 15,509 | ~210K | Superset — includes `disagree` and `unclear` verdicts. Use only if you want to study label noise or run ablations on verdict filtering. |
| `canary/dataset_final.jsonl` | 119 | ~1,517 | Held-out contamination test set. Compounds first indexed after 2024-01-01 (minimal LLM-pretraining exposure). Never train on this. |
| `cid_to_split.json` | — | — | CID → split mapping. The `split` field on each record is already populated from this; included for transparency. |
| `scaffold_split_report.json` | — | — | Statistics on the split (scaffold count, group sizes). Cite this in the paper. |

## Schema

### Top-level compound record

```json
{
  "cid": 71,
  "split": "train",                   // "train" | "val" | "test" | "canary"
  "name": "2-Oxoadipic acid",
  "iupac_name": "2-oxohexanedioic acid",
  "smiles": "C(CC(=O)C(=O)O)CC(=O)O",
  "molecular_formula": "C6H8O5",
  "molecular_weight": 160.037,
  "inchi_key": "InChI=1S/...",
  "num_pmids": 1,
  "num_synonyms": 58,
  "num_evidence_sentences": 2,
  "evidence_sentences": [...],        // redacted PubMed/PMC sentences, one per object
  "qa_pairs": [...]                   // the actual Q&A list
}
```

### Per-Q&A record

```json
{
  "qa_index": 1,
  "topic": "mechanism",               // freeform tag; see DATASHEET.md for taxonomy
  "question": "What is the molecular target of the compound?",
  "phase1_answer": "The compound is a ...",
  "phase2_answer": "This compound acts as ...",
  "verdict": "agree",                 // agree | disagree | unclear
  "judge_reasoning": "Both answers identify the same target ...",
  "evidence_ids": [1, 3]              // which evidence sentence IDs the claim draws on
}
```

For fine-tuning you typically use `question` + `phase1_answer` (or `phase2_answer`; both are valid — see "Which answer to train on" below).

## Split details

The split follows **MoleculeNet-standard strict scaffold splitting**:

1. For each compound, compute its Murcko scaffold via RDKit
   (`MurckoScaffold.GetScaffoldForMol`).
2. Group compounds by scaffold.
3. Sort groups by size, descending.
4. Greedy-pack groups into train (70%), val (15%), test (15%). Whole
   scaffold groups always stay together.

**Why this matters:** compounds sharing a scaffold can trivially leak
information between splits. Scaffold splitting guarantees val/test
compounds have structurally distinct cores from anything in train. This
is standard practice in chemistry ML.

**Properties:**
- 5,916 unique scaffolds across the non-canary set.
- Largest scaffold group: 1,299 compounds (in train).
- **Zero scaffold leakage** between train/val/test (verified
  programmatically).
- Val/test are harder than a random split — they consist mostly of
  singleton scaffolds, so performance there tests generalization, not
  memorization.

**Canary** is a separate held-out split (not mixed into train/val/test)
containing 119 compounds whose earliest PubMed article is ≥2024-01-01 —
i.e., largely outside the LLM training corpus. See `CONTAMINATION.md`.

## Loading the data

### Plain Python

```python
import json

def load(path, split=None):
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if split is None or rec.get("split") == split:
                yield rec

# Training set: all compounds tagged "train", flatten to Q&A pairs
train_pairs = []
for compound in load("dataset_gold.jsonl", split="train"):
    for qa in compound["qa_pairs"]:
        train_pairs.append({
            "cid": compound["cid"],
            "smiles": compound["smiles"],
            "formula": compound["molecular_formula"],
            "mw": compound["molecular_weight"],
            "question": qa["question"],
            "answer": qa["phase1_answer"],
            "topic": qa["topic"],
        })

print(f"Training Q&A pairs: {len(train_pairs):,}")
```

### Hugging Face Datasets

```python
from datasets import load_dataset

ds = load_dataset(
    "json",
    data_files={
        "train": "dataset_gold.jsonl",
        "canary": "canary/dataset_final.jsonl",
    },
)
# `train` here contains ALL splits in the main file — filter further:
train = ds["train"].filter(lambda r: r["split"] == "train")
val   = ds["train"].filter(lambda r: r["split"] == "val")
test  = ds["train"].filter(lambda r: r["split"] == "test")
```

### Flattening to one Q&A per row (standard SFT format)

```python
def flatten(records):
    for rec in records:
        for qa in rec["qa_pairs"]:
            yield {
                "messages": [
                    {"role": "system",
                     "content": "You are an expert medicinal chemist. Answer questions about the compound given only its SMILES and provided identifiers."},
                    {"role": "user",
                     "content": f"SMILES: {rec['smiles']}\nMolecular formula: {rec['molecular_formula']}\nMolecular weight: {rec['molecular_weight']}\n\nQUESTION: {qa['question']}"},
                    {"role": "assistant",
                     "content": qa["phase1_answer"]},
                ],
                "cid": rec["cid"],
                "split": rec["split"],
                "topic": qa["topic"],
            }
```

## Which answer to train on

Each Q&A has two candidate answers:
- `phase1_answer` — Gemini 3 Flash preview (dataset generator).
- `phase2_answer` — Kimi K2.5 (blind independent re-answer).

Both were scored `agree` by the judge (in the gold subset), so either is
a valid target. Pragmatic choices:

- **Train on `phase1_answer` only** — simplest. ~188K Q&A in gold.
  Recommended default.
- **Train on both, duplicated per Q&A** — effectively 2× the training
  signal. May help for a large model. Risk: the model sees two slightly
  different wordings for the same question, which can be a regularizer
  OR a confuser depending on base model.
- **Train on `phase1_answer`, use `phase2_answer` as a preference
  sample** for DPO-style training. Advanced; probably out of scope
  for the first round.

For first pass: **`phase1_answer` only**.

## Recommended training stack

- **Base model:** Qwen3-4B-Instruct-2507 (general) or LlaSMol-Mistral-7B
  (chemistry-pretrained). See main project docs for rationale.
- **Framework:** Axolotl or Hugging Face TRL with `SFTTrainer`.
  Axolotl's config-driven approach is cleanest for multi-GPU.
- **Hardware:** fits on 1×A100 40GB (LoRA) or scales across 4×H100
  (full fine-tune + FSDP).
- **LoRA config (if not full FT):** r=16, α=16, target_modules="all-linear",
  DoRA on, 2–3 epochs, lr=2e-4. Matches 2026 best-practice benchmarks.

## Evaluation protocol

1. **Dev**: tune hyperparameters on `split == "val"` only. Never touch test.
2. **Test**: final headline numbers on `split == "test"`. Report
   per-topic and per-verdict-source accuracy if possible.
3. **Canary**: run the final checkpoint on `canary/dataset_final.jsonl`
   separately. The delta between test accuracy and canary accuracy is
   the memorization footprint signal (see `CONTAMINATION.md`).

## Things to watch out for

- **Topic imbalance.** The `topic` field is freeform. Structural topics
  (composition, scaffold, functional_groups) dominate in volume;
  functional topics (mechanism, metabolism, therapeutic_use) are fewer
  per compound but exist on most. Stratify evaluation by topic bucket if
  you want clean per-category metrics. See `scripts/topic_bucket.py`.
- **Evidence concentration.** ~65% of total evidence sentences come
  from the top 10% of compounds (well-studied drugs). Q&A count per
  compound scales with evidence, so training signal is weighted toward
  those. If you want a balanced fine-tune, downsample compounds with
  >100 Q&A.
- **Compound identity is redacted.** In `evidence_sentences[].text`,
  the compound name is replaced with `[COMPOUND]`. Do not un-redact
  during training — it would defeat the blind-answer premise.
- **The `evidence_ids` field** on each Q&A tells you which evidence
  sentences the Phase 1 model cited as supporting the answer. Useful
  for grounding verification; can be empty for purely structural claims.
- **Stochasticity.** Two runs of the dataset pipeline with different
  seeds produce ~34% token-Jaccard on the same Q&A (counts are stable).
  The shipped dataset is a single seed — note this as "fixed-seed
  release" in any paper.

## What NOT to do

- Don't re-split. The scaffold split has zero leakage by construction;
  any re-split will be worse.
- Don't train on `dataset_final.jsonl` disagree/unclear verdicts unless
  you're specifically studying label noise. The gold subset is the
  intended training signal.
- Don't train on canary. It's the contamination test set. If you want
  more training data, use the standard or broad tiers of the raw
  dataset instead.
- Don't claim clinical validity of any trained model. See
  `RESPONSIBLE_AI.md`. This is an instruction-tuning research
  resource, not a clinical tool.

## Pointers to deeper docs

- `DATASHEET.md` — Gebru-style datasheet
- `LIMITATIONS.md` — every known weakness, with measurements
- `CONTAMINATION.md` — why the canary exists and how it's used
- `RESPONSIBLE_AI.md` — intended use, bias, misuse
- `EVALUATIVE_ROLE.md` — what claims the dataset supports
- `scaffold_split_report.json` — the machine-readable split stats
