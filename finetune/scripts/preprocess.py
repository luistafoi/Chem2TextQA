"""Expand ChemQA dataset_gold.jsonl into per-QA SFT records.

Each molecule row contains a `qa_pairs` list. We expand into individual
(question, answer) rows, prepending the SMILES string to the question so the
model is conditioned on the molecule. We prefer `phase2_answer` and fall back
to `phase1_answer` (dropping rows that have neither).
"""
import json
import os
from pathlib import Path

SRC = Path("/mnt/data_lab/ChemQA/data/dataset_gold.jsonl")
OUT_DIR = Path("/mnt/data_lab/ChemQA/data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_prompt(smiles: str, question: str) -> str:
    # Prepend the SMILES string (no label) to the question.
    return f"{smiles}\n{question.strip()}"


def main():
    buckets = {"train": [], "val": [], "test": []}
    with SRC.open() as f:
        for line in f:
            row = json.loads(line)
            split = row.get("split")
            if split not in buckets:
                continue
            smiles = row.get("smiles", "")
            for qa in row.get("qa_pairs") or []:
                q = (qa.get("question") or "").strip()
                a = qa.get("phase2_answer") or qa.get("phase1_answer")
                if not q or not a:
                    continue
                buckets[split].append(
                    {
                        "cid": row.get("cid"),
                        "smiles": smiles,
                        "qa_index": qa.get("qa_index"),
                        "topic": qa.get("topic"),
                        "prompt": make_prompt(smiles, q),
                        "response": a.strip(),
                    }
                )

    for split, records in buckets.items():
        out = OUT_DIR / f"{split}.jsonl"
        with out.open("w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(records)} records -> {out}")


if __name__ == "__main__":
    main()
