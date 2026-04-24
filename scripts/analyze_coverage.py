"""Quantify compound-coverage bias in the Phase 0 evidence pool.

Surfaces the distributions reviewers will ask for:
- Molecular-weight bucket distribution
- Heavy-atom count distribution
- Evidence volume vs compound (power-law tail check)
- PMID count per compound (literature weight)
- Whether a compound has full-text vs abstract-only articles

Also emits a "top-studied" vs "long-tail" split: compounds above the 90th
percentile of evidence-sentence count are expected to dominate the Q&A
population; we quantify by how much.

Usage:
    python3 scripts/analyze_coverage.py \\
        --input data/qa_pipeline/phase0_full_premium_v3/evidence_per_cid.jsonl \\
        --output data/qa_pipeline/coverage_report.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def _mw_bucket(mw):
    if mw is None:
        return "unknown"
    try:
        mw = float(mw)
    except (TypeError, ValueError):
        return "unknown"
    if mw < 150:
        return "<150 (fragments, ions)"
    if mw < 300:
        return "150-299 (Lipinski small)"
    if mw < 500:
        return "300-499 (drug-like)"
    if mw < 800:
        return "500-799 (large drug-like)"
    if mw < 1500:
        return "800-1499 (macrocyclic / peptide)"
    return ">=1500 (biologic / large peptide)"


def _heavy_atoms(smiles: str) -> int:
    """Very rough: count non-H, non-bracket-non-digit characters."""
    import re
    if not smiles:
        return 0
    # Remove brackets + their contents (treating each as one atom)
    cleaned = re.sub(r"\[[^\]]*\]", "A", smiles)
    n = 0
    for ch in cleaned:
        if ch.isalpha() and ch.upper() in "CONSPBFICLRIA":
            n += 1
    return n


def _heavy_bucket(n):
    if n <= 5:
        return "<=5 (tiny)"
    if n <= 15:
        return "6-15 (small)"
    if n <= 30:
        return "16-30 (drug-like)"
    if n <= 60:
        return "31-60 (large)"
    return ">60 (macro)"


def analyze(input_path: Path, output_path: Path):
    mw_buckets = Counter()
    heavy_buckets = Counter()
    sentence_counts = []
    pmid_counts = []
    n = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n += 1
            mw_buckets[_mw_bucket(rec.get("molecular_weight"))] += 1
            heavy_buckets[_heavy_bucket(_heavy_atoms(rec.get("smiles") or ""))] += 1
            sentence_counts.append(len(rec.get("evidence_sentences") or []))
            pmid_counts.append(rec.get("num_pmids") or 0)

    def _pctile(xs, q):
        if not xs:
            return 0
        xs = sorted(xs)
        k = max(0, min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1)))))
        return xs[k]

    # Concentration: top 10% of compounds' share of total evidence
    if sentence_counts:
        sorted_desc = sorted(sentence_counts, reverse=True)
        top10_cutoff = max(1, len(sorted_desc) // 10)
        top10_share = sum(sorted_desc[:top10_cutoff]) / sum(sorted_desc)
    else:
        top10_share = 0.0

    report = {
        "n_compounds": n,
        "molecular_weight_bucket": dict(mw_buckets.most_common()),
        "heavy_atom_bucket": dict(heavy_buckets.most_common()),
        "evidence_sentence_stats": {
            "mean": statistics.mean(sentence_counts) if sentence_counts else 0,
            "median": statistics.median(sentence_counts) if sentence_counts else 0,
            "p90": _pctile(sentence_counts, 90),
            "p99": _pctile(sentence_counts, 99),
            "max": max(sentence_counts, default=0),
            "top_10pct_share_of_total_evidence": top10_share,
        },
        "pmid_stats": {
            "mean": statistics.mean(pmid_counts) if pmid_counts else 0,
            "median": statistics.median(pmid_counts) if pmid_counts else 0,
            "p90": _pctile(pmid_counts, 90),
            "max": max(pmid_counts, default=0),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    print(f"  Compounds analyzed: {n:,}")
    print()
    print("  Molecular-weight distribution:")
    for k, v in report["molecular_weight_bucket"].items():
        print(f"    {k:<32} {v:>6,}  ({100*v/n:.1f}%)")
    print()
    print("  Heavy-atom distribution:")
    for k, v in report["heavy_atom_bucket"].items():
        print(f"    {k:<20} {v:>6,}  ({100*v/n:.1f}%)")
    print()
    e = report["evidence_sentence_stats"]
    print(f"  Evidence sentences per compound: mean={e['mean']:.1f}, median={e['median']:.0f}, "
          f"p90={e['p90']:.0f}, p99={e['p99']:.0f}, max={e['max']:.0f}")
    print(f"  Top 10% compounds hold {100*e['top_10pct_share_of_total_evidence']:.1f}% of total evidence.")
    print()
    print(f"  Wrote: {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    analyze(Path(args.input), Path(args.output))


if __name__ == "__main__":
    sys.exit(main() or 0)
