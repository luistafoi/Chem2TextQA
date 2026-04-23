"""Empty-evidence probe — does the model invent functional claims when deprived
of evidence hints?

For each test compound we run Phase 1 twice:
  1. Real evidence (baseline)
  2. Empty evidence list

Then compare. Under the soft-rule design:
  • Structural Q&A should still appear in the empty run (derivable from SMILES).
  • Functional Q&A SHOULD shrink drastically or disappear — if it persists
    and looks similar to the real run, the model is writing functional claims
    from training recall, not from evidence. This is the identity-leak signal.

Subcommands:
  prepare  → sample N compounds, write real + empty evidence files
  compare  → diff two Phase 1 outputs, bucketed by structural vs functional
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from topic_bucket import bucket_topic  # noqa: E402


def prepare(args):
    random.seed(args.seed)
    all_records = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_records.append(json.loads(line))

    eligible = [r for r in all_records if len(r.get("evidence_sentences", [])) >= 5]
    sampled = random.sample(eligible, min(args.n, len(eligible)))
    print(f"  Eligible (>=5 sents): {len(eligible):,}")
    print(f"  Sampled:              {len(sampled):,}")

    real_path = Path(args.output_real)
    empty_path = Path(args.output_empty)
    real_path.parent.mkdir(parents=True, exist_ok=True)
    empty_path.parent.mkdir(parents=True, exist_ok=True)

    with real_path.open("w", encoding="utf-8") as real_f, \
         empty_path.open("w", encoding="utf-8") as empty_f:
        for rec in sampled:
            real_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stripped = dict(rec)
            stripped["evidence_sentences"] = []
            stripped["num_pmids"] = 0
            stripped["pmids"] = []
            empty_f.write(json.dumps(stripped, ensure_ascii=False) + "\n")

    print(f"  Wrote real:  {real_path}")
    print(f"  Wrote empty: {empty_path}")


def _load_qa(path: Path) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[int(rec["cid"])] = rec.get("qa_pairs", []) or []
    return out


def compare(args):
    real = _load_qa(Path(args.real))
    empty = _load_qa(Path(args.empty))
    common = sorted(set(real.keys()) & set(empty.keys()))
    print(f"  Real records:            {len(real):,}")
    print(f"  Empty records:           {len(empty):,}")
    print(f"  Compounds in both runs:  {len(common):,}")

    # Phase 1 refusal: compounds that failed in the empty run but succeeded in real
    refused = sorted(set(real.keys()) - set(empty.keys()))

    # Per-compound: Q&A counts by bucket
    per_compound = []
    total_real = Counter()
    total_empty = Counter()

    for cid in common:
        r = real[cid]
        e = empty[cid]
        r_bucket = Counter(bucket_topic(q.get("topic")) for q in r)
        e_bucket = Counter(bucket_topic(q.get("topic")) for q in e)
        total_real.update(r_bucket)
        total_empty.update(e_bucket)
        per_compound.append({
            "cid": cid,
            "n_real_total": len(r),
            "n_empty_total": len(e),
            "real_by_bucket": dict(r_bucket),
            "empty_by_bucket": dict(e_bucket),
        })

    def _share(c, b): return c.get(b, 0) / sum(c.values()) if sum(c.values()) else 0.0

    report = {
        "n_compounds": len(common),
        "n_refused_in_empty_run": len(refused),
        "refused_cids": refused,
        "total_real_by_bucket": dict(total_real),
        "total_empty_by_bucket": dict(total_empty),
        "real_bucket_shares": {b: _share(total_real, b) for b in ("structural", "functional", "other")},
        "empty_bucket_shares": {b: _share(total_empty, b) for b in ("structural", "functional", "other")},
        "per_compound": per_compound,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"  Wrote report: {args.output}")

    print("")
    print(f"  Refused / failed in empty run: {len(refused)}")
    print("")
    print("  Q&A counts by bucket (totals across all compounds):")
    print(f"    {'':<12} {'REAL':>8} {'EMPTY':>8}  Δ (empty - real)")
    for b in ("structural", "functional", "other"):
        r = total_real.get(b, 0)
        e = total_empty.get(b, 0)
        print(f"    {b:<12} {r:>8} {e:>8}  {e - r:+d}")
    print("")
    print("  Share of Q&A pairs per bucket:")
    for b in ("structural", "functional", "other"):
        print(f"    {b:<12} real={100*report['real_bucket_shares'][b]:.1f}%   "
              f"empty={100*report['empty_bucket_shares'][b]:.1f}%")
    print("")
    print("  INTERPRETATION:")
    print("    Empty run with ~only structural Q&A → model respects 'functional needs evidence'.")
    print("    Empty run still producing lots of functional Q&A → hallucination from SMILES recall.")
    print("    Empty run reducing total Q&A count → model self-limits when hints absent.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare",
        help="Sample N compounds; write real and empty-evidence files")
    pp.add_argument("--input", required=True,
        help="Cached Phase 0 evidence (full or pilot)")
    pp.add_argument("-n", type=int, default=30)
    pp.add_argument("--seed", type=int, default=29)
    pp.add_argument("--output-real", required=True)
    pp.add_argument("--output-empty", required=True)
    pp.set_defaults(func=prepare)

    cp = sub.add_parser("compare",
        help="Diff real vs empty Phase 1 outputs, bucketed")
    cp.add_argument("--real", required=True)
    cp.add_argument("--empty", required=True)
    cp.add_argument("--output", default=None)
    cp.set_defaults(func=compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
