"""SMILES-swap probe — does the model actually parse each unique SMILES?

For each compound A in our test set we pick a different compound B and
build a hybrid evidence record: A's redacted evidence sentences + B's
SMILES, formula, and MW. We then run Phase 1 on the hybrid.

Two baselines to compare against (already produced by the ablation probe):
  • real_A = Phase 1 on (evidence_A, smiles_A)
  • real_B = Phase 1 on (evidence_B, smiles_B)

For each hybrid output we compute two answer-level Jaccard similarities:
  • sim_to_evidence_owner = sim(hybrid_A, real_A)   # "model tracked the evidence"
  • sim_to_smiles_donor   = sim(hybrid_A, real_B)   # "model tracked the SMILES"

Interpretation:
  sim_to_smiles_donor >> sim_to_evidence_owner
      → model parses each unique SMILES (good; no identity leak through evidence)
  sim_to_evidence_owner >> sim_to_smiles_donor
      → model mostly ignored SMILES and answered from evidence topic signal
  roughly equal (both low)
      → answers are generic structural boilerplate not specific to either
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from topic_bucket import bucket_topic  # noqa: E402


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "with", "which", "what", "how", "when", "where", "this",
    "these", "those", "their", "they", "them", "compound", "molecule",
    "structure", "structural",
})


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower())
            if t not in _STOPWORDS and len(t) > 2}


def _answers_blob(qa_pairs) -> str:
    return " ".join(q.get("answer", "") for q in (qa_pairs or []))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


# ------------------------------------------------------------------
# prepare — swap SMILES between pairs
# ------------------------------------------------------------------


def prepare(args):
    random.seed(args.seed)
    real_evidence = []
    with Path(args.real_evidence).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            real_evidence.append(json.loads(line))

    n = len(real_evidence)
    print(f"  Real evidence records: {n}")

    # Derangement: every compound gets a partner that is NOT itself.
    indices = list(range(n))
    partners = list(range(n))
    random.shuffle(partners)
    for i in range(n):
        if partners[i] == i:
            j = (i + 1) % n
            partners[i], partners[j] = partners[j], partners[i]

    # Hybrid CIDs are synthesized so they don't collide with real baseline CIDs
    # in Phase 1's resume check. Mapping (hybrid_cid → evidence_owner, smiles_donor)
    # is persisted in a sidecar so the compare step can look up both baselines —
    # Phase 1 strips non-standard fields from its output records.
    hybrid = []
    mapping = {}
    for i, p in zip(indices, partners):
        evidence_owner = real_evidence[i]
        smiles_donor = real_evidence[p]
        ev_cid = int(evidence_owner["cid"])
        sm_cid = int(smiles_donor["cid"])
        hybrid_cid = sm_cid * 10_000_000 + ev_cid

        rec = dict(evidence_owner)
        rec["cid"] = hybrid_cid
        rec["smiles"] = smiles_donor.get("smiles") or ""
        rec["molecular_formula"] = smiles_donor.get("molecular_formula") or ""
        rec["molecular_weight"] = smiles_donor.get("molecular_weight")
        hybrid.append(rec)

        mapping[str(hybrid_cid)] = {
            "evidence_owner_cid": ev_cid,
            "smiles_donor_cid": sm_cid,
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in hybrid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sidecar = Path(args.mapping)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print(f"  Wrote {len(hybrid)} hybrid records: {out}")
    print(f"  Wrote mapping sidecar:              {sidecar}")


# ------------------------------------------------------------------
# compare — diff hybrid output against both baselines
# ------------------------------------------------------------------


def compare(args):
    def _load(path):
        out = {}
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out[int(rec["cid"])] = rec
        return out

    real = _load(args.real)
    hybrid = _load(args.hybrid)
    with Path(args.mapping).open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    print(f"  Real baseline records: {len(real)}")
    print(f"  Hybrid records:        {len(hybrid)}")
    print(f"  Mapping entries:       {len(mapping)}")

    def _bucketed_blob(pairs, bucket):
        return " ".join(
            q.get("answer", "") for q in (pairs or [])
            if bucket_topic(q.get("topic")) == bucket
        )

    sims_evidence = []
    sims_smiles = []
    # Per-bucket tracking
    bucket_tracks: dict[str, list[dict]] = defaultdict(list)
    per = []
    for cid, rec in hybrid.items():
        m = mapping.get(str(cid))
        if m is None:
            continue
        ev_cid = int(m["evidence_owner_cid"])
        sm_cid = int(m["smiles_donor_cid"])
        if ev_cid not in real or sm_cid not in real:
            continue
        hybrid_tokens = _tokens(_answers_blob(rec.get("qa_pairs")))
        evidence_tokens = _tokens(_answers_blob(real[ev_cid].get("qa_pairs")))
        smiles_tokens = _tokens(_answers_blob(real[sm_cid].get("qa_pairs")))
        s_ev = _jaccard(hybrid_tokens, evidence_tokens)
        s_sm = _jaccard(hybrid_tokens, smiles_tokens)
        sims_evidence.append(s_ev)
        sims_smiles.append(s_sm)

        # Per-bucket: compare only structural-bucket answers across runs,
        # then functional-bucket separately.
        for bucket in ("structural", "functional"):
            h_blob = _bucketed_blob(rec.get("qa_pairs"), bucket)
            ev_blob = _bucketed_blob(real[ev_cid].get("qa_pairs"), bucket)
            sm_blob = _bucketed_blob(real[sm_cid].get("qa_pairs"), bucket)
            if not h_blob or (not ev_blob and not sm_blob):
                continue
            s_ev_b = _jaccard(_tokens(h_blob), _tokens(ev_blob))
            s_sm_b = _jaccard(_tokens(h_blob), _tokens(sm_blob))
            bucket_tracks[bucket].append({
                "sim_to_evidence_owner": s_ev_b,
                "sim_to_smiles_donor": s_sm_b,
            })

        per.append({
            "hybrid_cid": cid,
            "evidence_owner_cid": ev_cid,
            "smiles_donor_cid": sm_cid,
            "sim_to_evidence_owner": s_ev,
            "sim_to_smiles_donor": s_sm,
            "tracks": ("smiles" if s_sm > s_ev + 0.05
                      else "evidence" if s_ev > s_sm + 0.05
                      else "tie"),
        })

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    tracks_count = {"smiles": 0, "evidence": 0, "tie": 0}
    for p in per:
        tracks_count[p["tracks"]] += 1

    # Bucketed summary
    bucket_summary = {}
    for bucket, rows in bucket_tracks.items():
        ev_means = [r["sim_to_evidence_owner"] for r in rows]
        sm_means = [r["sim_to_smiles_donor"] for r in rows]
        tracks = {"smiles": 0, "evidence": 0, "tie": 0}
        for r in rows:
            if r["sim_to_smiles_donor"] > r["sim_to_evidence_owner"] + 0.05:
                tracks["smiles"] += 1
            elif r["sim_to_evidence_owner"] > r["sim_to_smiles_donor"] + 0.05:
                tracks["evidence"] += 1
            else:
                tracks["tie"] += 1
        bucket_summary[bucket] = {
            "n_compounds": len(rows),
            "mean_sim_to_smiles_donor": _mean(sm_means),
            "mean_sim_to_evidence_owner": _mean(ev_means),
            **{f"tracks_{k}": v for k, v in tracks.items()},
        }

    report = {
        "n_pairs": len(per),
        "mean_sim_to_smiles_donor": _mean(sims_smiles),
        "mean_sim_to_evidence_owner": _mean(sims_evidence),
        "tracks_smiles": tracks_count["smiles"],
        "tracks_evidence": tracks_count["evidence"],
        "tracks_tie": tracks_count["tie"],
        "per_bucket": bucket_summary,
        "per_pair": per,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"  Wrote report: {args.output}")

    print("")
    print(f"  Pairs analyzed:                         {len(per)}")
    print(f"  Mean answer-Jaccard to SMILES donor:    "
          f"{report['mean_sim_to_smiles_donor']:.3f}")
    print(f"  Mean answer-Jaccard to evidence owner:  "
          f"{report['mean_sim_to_evidence_owner']:.3f}")
    print("")
    print(f"  Compounds whose answers track SMILES:   "
          f"{tracks_count['smiles']} ({100*tracks_count['smiles']/len(per):.0f}%)")
    print(f"  Compounds tracking evidence owner:      "
          f"{tracks_count['evidence']} ({100*tracks_count['evidence']/len(per):.0f}%)")
    print(f"  Ties (±0.05):                           "
          f"{tracks_count['tie']} ({100*tracks_count['tie']/len(per):.0f}%)")
    print("")
    print("  PER-BUCKET (soft-rule expectations):")
    print("    Structural: should track SMILES donor (answers derive from SMILES)")
    print("    Functional: should track EVIDENCE owner (answers derive from evidence)")
    for b in ("structural", "functional"):
        bs = bucket_summary.get(b)
        if not bs:
            continue
        n = bs["n_compounds"]
        if n == 0:
            continue
        pct_sm = 100 * bs["tracks_smiles"] / n
        pct_ev = 100 * bs["tracks_evidence"] / n
        print(f"    {b:<10} n={n:>3}  "
              f"sim_smiles={bs['mean_sim_to_smiles_donor']:.3f} "
              f"sim_evidence={bs['mean_sim_to_evidence_owner']:.3f}  "
              f"tracks: smiles={pct_sm:.0f}% evidence={pct_ev:.0f}%")
    print("")
    print("  OVERALL INTERPRETATION:")
    print("    Tracks SMILES > evidence  → model parses each unique SMILES.")
    print("    Tracks evidence > SMILES  → answers derived from evidence content.")
    print("    Under soft rule: structural=tracks_SMILES AND functional=tracks_EVIDENCE is the success pattern.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare",
        help="Build hybrid evidence: evidence_A + SMILES_B (pairwise swap)")
    pp.add_argument("--real-evidence", required=True,
        help="Original real evidence file (from ablation run, e.g. ablation50/real/evidence.jsonl)")
    pp.add_argument("--seed", type=int, default=17)
    pp.add_argument("--output", required=True)
    pp.add_argument("--mapping", required=True,
        help="Sidecar JSON mapping hybrid_cid → (evidence_owner_cid, smiles_donor_cid)")
    pp.set_defaults(func=prepare)

    cp = sub.add_parser("compare",
        help="Diff hybrid Phase 1 output against the baseline real Phase 1 output")
    cp.add_argument("--real", required=True,
        help="Phase 1 output on real (unswapped) evidence")
    cp.add_argument("--hybrid", required=True,
        help="Phase 1 output on hybrid (swapped) evidence")
    cp.add_argument("--mapping", required=True,
        help="Sidecar JSON produced by `prepare`")
    cp.add_argument("--output", default=None,
        help="Write full JSON report here")
    cp.set_defaults(func=compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
