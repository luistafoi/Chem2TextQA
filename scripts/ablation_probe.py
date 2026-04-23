"""Ablation probe — does the private topic hint actually drive Phase 1?

We take N compounds and build two evidence files:
  1. real      — the original redacted evidence for each compound.
  2. scrambled — same N compounds (same SMILES/formula/MW), but every
                 evidence sentence is replaced with a random sentence drawn
                 from a DIFFERENT compound's evidence pool.

We then run Phase 1 on both and compare outputs per-compound. If the
scrambled run produces the same topic distribution and similar questions,
the hints aren't steering the model — it's answering from internal recall
of the SMILES. If the runs diverge, the hints are doing their job.

Subcommands:
  prepare  → produce ablation_real/*.jsonl and ablation_scrambled/*.jsonl
  compare  → diff the two Phase 1 outputs and report overlap metrics
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from topic_bucket import bucket_topic  # noqa: E402


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "he", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "to", "was", "were", "will", "with", "which", "what", "how", "when",
    "where", "why", "this", "these", "those", "their", "they", "them",
    "compound", "molecule", "structure", "the",
})


def _content_tokens(text: str) -> set[str]:
    toks = _TOKEN_RE.findall(text.lower())
    return {t for t in toks if t not in _STOPWORDS and len(t) > 2}


# ------------------------------------------------------------------
# prepare
# ------------------------------------------------------------------


def prepare(args):
    random.seed(args.seed)
    evidence_path = Path(args.input)

    all_records = []
    with evidence_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_records.append(json.loads(line))

    eligible = [r for r in all_records if len(r.get("evidence_sentences", [])) >= 5]
    print(f"  Total records:          {len(all_records):,}")
    print(f"  Eligible (>=5 sents):   {len(eligible):,}")
    sampled = random.sample(eligible, min(args.n, len(eligible)))
    sampled_cids = {int(r["cid"]) for r in sampled}
    print(f"  Sampled:                {len(sampled):,}")

    # Build a pool of sentences from compounds NOT in our sample.
    pool = []
    for r in all_records:
        if int(r["cid"]) in sampled_cids:
            continue
        for s in r.get("evidence_sentences", []) or []:
            text = s.get("text") or ""
            if text:
                pool.append(text)
    print(f"  Scramble pool (sents):  {len(pool):,}")

    real_dir = Path(args.output_real).parent
    scr_dir = Path(args.output_scrambled).parent
    real_dir.mkdir(parents=True, exist_ok=True)
    scr_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.output_real).open("w", encoding="utf-8") as real_f, \
         Path(args.output_scrambled).open("w", encoding="utf-8") as scr_f:
        for rec in sampled:
            real_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            n = len(rec.get("evidence_sentences", []))
            scrambled = dict(rec)
            picks = random.sample(pool, n) if n <= len(pool) else random.choices(pool, k=n)
            scrambled["evidence_sentences"] = [
                {"id": i + 1, "pmid": "SCRAMBLED", "source": "scramble", "text": picks[i]}
                for i in range(n)
            ]
            scrambled["num_pmids"] = 0
            scrambled["pmids"] = []
            scr_f.write(json.dumps(scrambled, ensure_ascii=False) + "\n")

    print(f"  Wrote real:      {args.output_real}")
    print(f"  Wrote scrambled: {args.output_scrambled}")


# ------------------------------------------------------------------
# compare
# ------------------------------------------------------------------


def _load_qa(path: Path) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[int(rec["cid"])] = rec.get("qa_pairs", []) or []
    return out


def _topic_jaccard(a: list[dict], b: list[dict]) -> float:
    ta = {q.get("topic", "other") for q in a}
    tb = {q.get("topic", "other") for q in b}
    u = ta | tb
    return len(ta & tb) / len(u) if u else 0.0


def _best_pair_jaccard(a: list[dict], b: list[dict], field: str) -> float:
    """For each question in a, find the best-matching question in b by
    content-token Jaccard; return the average over a."""
    if not a or not b:
        return 0.0
    b_tokens = [_content_tokens(q.get(field, "")) for q in b]
    total = 0.0
    for qa in a:
        ta = _content_tokens(qa.get(field, ""))
        best = 0.0
        for tb in b_tokens:
            u = ta | tb
            j = len(ta & tb) / len(u) if u else 0.0
            if j > best:
                best = j
        total += best
    return total / len(a)


def _by_bucket(pairs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for q in pairs:
        out[bucket_topic(q.get("topic"))].append(q)
    return dict(out)


def compare(args):
    real = _load_qa(Path(args.real))
    scr = _load_qa(Path(args.scrambled))

    common = sorted(set(real.keys()) & set(scr.keys()))
    print(f"  Compounds in both runs: {len(common):,}")

    # Overall metrics
    topic_jacc, q_jacc, a_jacc = [], [], []
    # Per-bucket metrics — answer Jaccard by structural vs functional
    bucket_answer_jacc: dict[str, list[float]] = defaultdict(list)
    bucket_counts_real: dict[str, int] = Counter()
    bucket_counts_scr: dict[str, int] = Counter()

    topic_counts_real = Counter()
    topic_counts_scr = Counter()
    per_compound = []

    for cid in common:
        r = real[cid]
        s = scr[cid]
        tj = _topic_jaccard(r, s)
        qj = _best_pair_jaccard(r, s, "question")
        aj = _best_pair_jaccard(r, s, "answer")
        topic_jacc.append(tj); q_jacc.append(qj); a_jacc.append(aj)

        for q in r:
            topic_counts_real[q.get("topic", "other")] += 1
            bucket_counts_real[bucket_topic(q.get("topic"))] += 1
        for q in s:
            topic_counts_scr[q.get("topic", "other")] += 1
            bucket_counts_scr[bucket_topic(q.get("topic"))] += 1

        # Bucketed comparison: for each bucket, answer-Jaccard of real-bucket-pairs
        # against any scrambled pair (best-match across the full scrambled list).
        r_by = _by_bucket(r)
        for bucket, r_pairs in r_by.items():
            if not r_pairs or not s:
                continue
            aj_bucket = _best_pair_jaccard(r_pairs, s, "answer")
            bucket_answer_jacc[bucket].append(aj_bucket)

        per_compound.append({"cid": cid, "topic_jaccard": tj,
                              "question_jaccard": qj, "answer_jaccard": aj,
                              "n_real": len(r), "n_scrambled": len(s)})

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    bucket_report = {
        b: {"mean_answer_jaccard": _mean(xs), "n_compounds_with_bucket": len(xs)}
        for b, xs in bucket_answer_jacc.items()
    }

    report = {
        "n_compounds": len(common),
        "mean_topic_jaccard": _mean(topic_jacc),
        "mean_question_jaccard": _mean(q_jacc),
        "mean_answer_jaccard": _mean(a_jacc),
        "per_bucket_answer_jaccard": bucket_report,
        "bucket_counts_real": dict(bucket_counts_real),
        "bucket_counts_scrambled": dict(bucket_counts_scr),
        "topic_distribution_real": dict(topic_counts_real.most_common()),
        "topic_distribution_scrambled": dict(topic_counts_scr.most_common()),
        "per_compound": per_compound,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"  Wrote report: {args.output}")

    print("")
    print("  OVERALL (real vs scrambled)")
    print(f"    Topic Jaccard:    {report['mean_topic_jaccard']:.3f}")
    print(f"    Question Jaccard: {report['mean_question_jaccard']:.3f}")
    print(f"    Answer Jaccard:   {report['mean_answer_jaccard']:.3f}")
    print("")
    print("  PER-BUCKET ANSWER JACCARD (soft-rule expectations):")
    print("    Structural → high Jaccard = OK (SMILES-driven, not hint-dependent)")
    print("    Functional → low  Jaccard = OK (hints drive functional claims)")
    print("    Functional → high Jaccard = RED FLAG (functional claims from recall, not evidence)")
    for b in ("structural", "functional", "other"):
        r = bucket_report.get(b)
        if r:
            n_real = bucket_counts_real.get(b, 0)
            n_scr = bucket_counts_scr.get(b, 0)
            print(f"    {b:<12} {r['mean_answer_jaccard']:.3f}   "
                  f"(real_pairs={n_real}, scrambled_pairs={n_scr})")
    print("")
    print("  Top 5 topics (real):     ",
          list(topic_counts_real.most_common(5)))
    print("  Top 5 topics (scrambled):",
          list(topic_counts_scr.most_common(5)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare",
        help="Sample N compounds, write real + scrambled evidence files")
    pp.add_argument("--input", required=True,
        help="Cached Phase 0 evidence (e.g. phase0_full_premium/evidence_per_cid.jsonl)")
    pp.add_argument("-n", type=int, default=50)
    pp.add_argument("--seed", type=int, default=13)
    pp.add_argument("--output-real", required=True)
    pp.add_argument("--output-scrambled", required=True)
    pp.set_defaults(func=prepare)

    cp = sub.add_parser("compare",
        help="Diff two Phase 1 qa_pairs.jsonl files")
    cp.add_argument("--real", required=True,
        help="Phase 1 output on real evidence")
    cp.add_argument("--scrambled", required=True,
        help="Phase 1 output on scrambled evidence")
    cp.add_argument("--output", default=None,
        help="Write full JSON report here")
    cp.set_defaults(func=compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
