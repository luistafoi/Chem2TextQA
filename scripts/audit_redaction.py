"""Audit Phase 0 redaction for false-negative compound-identity leaks.

Redaction is built from PubChem/MeSH/IUPAC/primary-name synonyms (see
chem2textqa/qa_pipeline/phase_0_evidence/synonyms.py). Compound mentions
via trade names, abbreviations, or non-English names that fall outside
that synonym union will leak through. This script measures how often
that happens against a held-out reference.

Two approaches implemented:

1. Synonym-holdout leave-one-source-out audit:
   For each CID, rebuild the redaction regex with ONE synonym source
   omitted (e.g., omit IUPAC). Re-redact the compound's evidence
   sentences. Count how many sentences contain the omitted names in
   unredacted form — that's the false-negative upper bound for that
   source.

2. External-lexicon spot check:
   Supply a newline-separated file of additional known names (trade
   names etc.). For each evidence sentence, check whether any external
   name appears unredacted. Report per-compound leakage rate.

Both modes emit a JSON report with per-compound and overall stats.

Usage:
    python3 scripts/audit_redaction.py holdout \\
        --evidence data/qa_pipeline/phase0_full_premium_v3/evidence_per_cid.jsonl \\
        --source iupac \\
        --output data/qa_pipeline/redaction_audit_iupac.json

    python3 scripts/audit_redaction.py external \\
        --evidence data/qa_pipeline/phase0_full_premium_v3/evidence_per_cid.jsonl \\
        --lexicon external_trade_names.txt \\
        --output data/qa_pipeline/redaction_audit_external.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _sentence_contains_whole_word(text: str, term: str) -> bool:
    if len(term) < 3:
        return False
    pat = r"\b" + re.escape(term) + r"\b"
    return bool(re.search(pat, text, re.IGNORECASE))


def audit_external(args):
    """Check evidence text for unredacted occurrences of external names."""
    lexicon_terms: list[str] = []
    with Path(args.lexicon).open("r", encoding="utf-8") as f:
        for line in f:
            term = line.strip()
            if len(term) >= 3:
                lexicon_terms.append(term)
    print(f"  External lexicon terms: {len(lexicon_terms)}")

    per_compound = []
    total_sentences = 0
    total_leaks = 0
    with Path(args.evidence).open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cid = int(rec["cid"])
            sents = rec.get("evidence_sentences") or []
            leaks = 0
            for s in sents:
                txt = s.get("text", "") or ""
                for term in lexicon_terms:
                    if _sentence_contains_whole_word(txt, term):
                        leaks += 1
                        break
            n = len(sents)
            if n:
                per_compound.append({"cid": cid, "n_sentences": n, "leaked_sentences": leaks})
                total_sentences += n
                total_leaks += leaks

    report = {
        "mode": "external_lexicon",
        "lexicon_size": len(lexicon_terms),
        "n_compounds": len(per_compound),
        "n_sentences_total": total_sentences,
        "n_sentences_leaked": total_leaks,
        "leak_rate_sentence_level": total_leaks / total_sentences if total_sentences else 0.0,
        "per_compound": per_compound[:1000],  # cap for readability
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))

    print(f"\n  Total compounds audited:   {report['n_compounds']:,}")
    print(f"  Sentences audited:         {report['n_sentences_total']:,}")
    print(f"  Sentences with leaks:      {report['n_sentences_leaked']:,}")
    print(f"  Sentence-level leak rate:  {100*report['leak_rate_sentence_level']:.2f}%")
    print(f"\n  Wrote: {args.output}")


def audit_holdout(args):
    """Per-compound leave-one-source-out audit.

    For each evidence record we inspect, we pretend the named synonym
    source (iupac / mesh_terms / pubchem_synonyms / primary_name) was
    missing, and ask: is any string from that source present in the
    evidence text? If yes, the production regex relied on the other
    sources to mask it — implying that if the named source were missing
    in the future, those mentions would leak.

    This is an upper bound on per-source leakage risk.
    """
    # Load the per-compound synonym sources from the Phase 0 output.
    # evidence_per_cid.jsonl records keep `synonyms_sample` (first 10)
    # but not the full synonym set. For a full audit we would re-run
    # Phase 0's synonym collection. For now, this holdout check uses the
    # primary name + iupac_name fields preserved on each record.

    source = args.source.lower().strip()
    assert source in {"iupac", "name"}, f"Unsupported source: {source}"

    per_compound = []
    total_sentences = 0
    total_leaks = 0
    with Path(args.evidence).open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cid = int(rec["cid"])
            if source == "iupac":
                holdout = [(rec.get("iupac_name") or "").strip()]
            else:
                holdout = [(rec.get("name") or "").strip()]
            holdout = [h for h in holdout if len(h) >= 3]
            if not holdout:
                continue
            sents = rec.get("evidence_sentences") or []
            leaks = 0
            for s in sents:
                txt = s.get("text", "") or ""
                if any(_sentence_contains_whole_word(txt, h) for h in holdout):
                    leaks += 1
            n = len(sents)
            if n:
                per_compound.append({"cid": cid, "n_sentences": n, "leaked_sentences": leaks})
                total_sentences += n
                total_leaks += leaks

    report = {
        "mode": f"holdout:{source}",
        "n_compounds": len(per_compound),
        "n_sentences_total": total_sentences,
        "n_sentences_leaked": total_leaks,
        "leak_rate_sentence_level": total_leaks / total_sentences if total_sentences else 0.0,
        "per_compound": per_compound[:1000],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))

    print(f"\n  Held-out source:           {source}")
    print(f"  Compounds audited:         {report['n_compounds']:,}")
    print(f"  Sentences audited:         {report['n_sentences_total']:,}")
    print(f"  Sentences with leaks:      {report['n_sentences_leaked']:,}")
    print(f"  Sentence-level leak rate:  {100*report['leak_rate_sentence_level']:.2f}%")
    print(f"\n  Wrote: {args.output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("holdout", help="Leave-one-source-out leakage audit")
    h.add_argument("--evidence", required=True)
    h.add_argument("--source", required=True, choices=["iupac", "name"])
    h.add_argument("--output", required=True)
    h.set_defaults(func=audit_holdout)

    e = sub.add_parser("external", help="Audit against an external lexicon file (one term per line)")
    e.add_argument("--evidence", required=True)
    e.add_argument("--lexicon", required=True)
    e.add_argument("--output", required=True)
    e.set_defaults(func=audit_external)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
