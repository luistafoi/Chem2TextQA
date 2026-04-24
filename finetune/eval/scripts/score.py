"""Score a predictions jsonl against its references using CIDEr (+BLEU, ROUGE-L).

Input jsonl must have `prediction` and `reference` fields.
Outputs a JSON with aggregate metrics. Uses pycocoevalcap for CIDEr.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

# Silence NLTK download spam
import os
os.environ.setdefault("NLTK_DATA", "/tmp/nltk_data")
os.makedirs(os.environ["NLTK_DATA"], exist_ok=True)

import nltk
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", download_dir=os.environ["NLTK_DATA"], quiet=True)

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge


WS = re.compile(r"\s+")


def _normalize(s: str) -> str:
    return WS.sub(" ", s.strip().lower())


def score(pred_file: Path) -> dict:
    items = [json.loads(l) for l in open(pred_file)]
    gts, res = {}, {}
    for i, r in enumerate(items):
        pred = _normalize(r.get("prediction", ""))
        ref = _normalize(r.get("reference", ""))
        if not ref:
            continue
        gts[i] = [ref]
        res[i] = [pred if pred else "<empty>"]
    n = len(gts)
    out = {"n": n, "file": str(pred_file)}

    cider, _ = Cider().compute_score(gts, res)
    out["CIDEr"] = float(cider)

    bleu, _ = Bleu(4).compute_score(gts, res)  # returns list [B1,B2,B3,B4]
    out["BLEU-1"] = float(bleu[0])
    out["BLEU-4"] = float(bleu[3])

    rouge, _ = Rouge().compute_score(gts, res)
    out["ROUGE-L"] = float(rouge)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_file", required=True)
    p.add_argument("--out_file", default=None)
    args = p.parse_args()

    metrics = score(Path(args.pred_file))
    out_file = args.out_file or args.pred_file.replace(".jsonl", ".metrics.json")
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
