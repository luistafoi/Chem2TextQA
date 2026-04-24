"""Measure seed-to-seed variance of the Phase 1 Q&A generator.

Runs Phase 1 on a fixed small compound sample K times with different
`temperature` random seeds (by relying on a temperature > 0 and
repeating the call). Reports:
  - Per-compound variance in Q&A count
  - Mean answer-Jaccard across runs (content stability)
  - Topic distribution stability

Cheap to run: K × N compounds × Phase 1 call. For K=3, N=30, ~$1.

Usage:
    python3 scripts/measure_seed_variance.py \\
        --evidence data/qa_pipeline/phase0_full_premium_v3/evidence_per_cid.jsonl \\
        --n 30 --seeds 3 \\
        --out-dir data/qa_pipeline/experiments/variance3x30

Note: this is a measurement script that delegates the generation to the
regular `chem2textqa qa-generate` CLI invoked K times on the same input,
writing each run's output to a numbered subdirectory. The comparison
phase loads all K outputs and emits a report.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "with", "which", "what", "how", "when", "where", "this",
    "these", "those", "their", "they", "them", "compound", "molecule",
})


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower())
            if t not in _STOPWORDS and len(t) > 2}


def _sample_evidence(src: Path, n: int, seed: int, out: Path):
    random.seed(seed)
    recs = [json.loads(l) for l in src.open("r", encoding="utf-8")]
    elig = [r for r in recs if len(r.get("evidence_sentences") or []) >= 3]
    sample = random.sample(elig, min(n, len(elig)))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return [int(r["cid"]) for r in sample]


def _load_qa(path: Path) -> dict[int, list[dict]]:
    out = {}
    if not path.exists():
        return out
    for line in path.open("r", encoding="utf-8"):
        rec = json.loads(line)
        out[int(rec["cid"])] = rec.get("qa_pairs") or []
    return out


def _pairwise_jaccard(a: list[dict], b: list[dict]) -> float:
    if not a or not b:
        return 0.0
    tokens_b = [_tokens(q.get("answer", "")) for q in b]
    scores = []
    for qa in a:
        ta = _tokens(qa.get("answer", ""))
        best = 0.0
        for tb in tokens_b:
            u = ta | tb
            j = len(ta & tb) / len(u) if u else 0.0
            best = max(best, j)
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0.0


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sample once, reuse for all seeds
    sample_path = out_dir / "evidence_sample.jsonl"
    _sample_evidence(Path(args.evidence), args.n, args.sample_seed, sample_path)
    print(f"  Sampled {args.n} compounds → {sample_path}")

    qa_paths = []
    for k in range(args.seeds):
        run_dir = out_dir / f"seed_{k}"
        run_dir.mkdir(parents=True, exist_ok=True)
        qa_path = run_dir / "qa_pairs.jsonl"
        # Clear any stale output so Phase 1 re-runs fresh
        if qa_path.exists():
            qa_path.unlink()
        print(f"  Running Phase 1 seed {k} →  {qa_path}")
        cmd = [
            "chem2textqa", "qa-generate",
            "--input", str(sample_path),
            "--output", str(qa_path),
            "--errors", str(run_dir / "errors.jsonl"),
            "--model", args.model,
            "--workers", str(args.workers),
        ]
        subprocess.run(cmd, check=True)
        qa_paths.append(qa_path)

    # Analyze
    runs = [_load_qa(p) for p in qa_paths]
    cids = sorted(set.intersection(*(set(r.keys()) for r in runs)))
    print(f"\n  Compounds present in all {args.seeds} runs: {len(cids)}")

    counts_per_compound = []
    pairwise_jaccards = []
    topic_counts_per_run = [Counter() for _ in runs]

    for cid in cids:
        counts = [len(r[cid]) for r in runs]
        counts_per_compound.append({"cid": cid, "counts": counts,
                                    "range": max(counts) - min(counts)})
        for i, r in enumerate(runs):
            for qa in r[cid]:
                topic_counts_per_run[i][qa.get("topic", "other")] += 1
        for i, j in combinations(range(len(runs)), 2):
            pairwise_jaccards.append(_pairwise_jaccard(runs[i][cid], runs[j][cid]))

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    count_range_mean = _mean([c["range"] for c in counts_per_compound])
    answer_jaccard_mean = _mean(pairwise_jaccards)

    report = {
        "n_seeds": args.seeds,
        "n_compounds": len(cids),
        "qa_count_range_per_compound_mean": count_range_mean,
        "answer_pairwise_jaccard_mean": answer_jaccard_mean,
        "topic_distribution_per_run": [dict(c.most_common(10)) for c in topic_counts_per_run],
    }
    (out_dir / "variance_report.json").write_text(json.dumps(report, indent=2))

    print()
    print("  Seed-variance report:")
    print(f"    Mean (max-min) Q&A count per compound: {count_range_mean:.2f}")
    print(f"    Mean pairwise answer Jaccard:          {answer_jaccard_mean:.3f}")
    print()
    print(f"  Wrote: {out_dir / 'variance_report.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", required=True)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--sample-seed", type=int, default=41)
    p.add_argument("--model", default="google/gemini-3-flash-preview")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
