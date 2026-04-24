"""Aggregate all .metrics.json files into a markdown table."""
import json
from pathlib import Path

RESULTS = Path("/mnt/data_lab/ChemQA/eval/results")


def main():
    rows = []
    for p in sorted(RESULTS.rglob("*.metrics.json")):
        m = json.loads(p.read_text())
        # tag is the relative path without extension, e.g. "qwen2_5_14b/base_chemqa"
        tag = p.relative_to(RESULTS).with_suffix("").as_posix().replace(".metrics", "")
        model, cond = tag.split("/", 1)
        dataset = "ChemQA" if "chemqa" in cond else "SMolInstruct-mcap"
        variant = "base" if cond.startswith("base") else "finetuned"
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "variant": variant,
                "n": m.get("n"),
                "CIDEr": m.get("CIDEr"),
                "BLEU-1": m.get("BLEU-1"),
                "BLEU-4": m.get("BLEU-4"),
                "ROUGE-L": m.get("ROUGE-L"),
            }
        )

    print("# ChemQA Evaluation Results\n")
    print("CIDEr computed with pycocoevalcap.\n")
    print("## ChemQA holdout test set\n")
    _print_table([r for r in rows if r["dataset"] == "ChemQA"])
    print("\n## SMolInstruct — molecule_captioning (LlaSMol test subset)\n")
    _print_table([r for r in rows if r["dataset"] == "SMolInstruct-mcap"])


def _print_table(rows):
    if not rows:
        print("_(no results yet)_")
        return
    header = ["model", "variant", "n", "CIDEr"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    # Sort: model then variant (base before finetuned)
    for r in sorted(rows, key=lambda x: (x["model"], x["variant"])):
        vals = [
            r["model"],
            r["variant"],
            str(r["n"]),
            f"{r['CIDEr']:.4f}",
        ]
        print("| " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main()
