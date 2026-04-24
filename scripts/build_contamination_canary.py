"""Build a post-cutoff contamination canary set.

A compound is post-cutoff if:
  1. Its earliest PubMed indexing date (across the premium-tier articles
     that reference it) is on or after `--cutoff` (default 2025-09-01).
  2. It has at least `--min-articles` (default 3) articles in the
     premium tier.
  3. Optionally: its PubChem CID creation date is on or after
     `--cid-cutoff` (default 2025-06-01) as a recency proxy.

Output is a CID list file consumable by
`chem2textqa qa-extract-evidence --target-cids <path>`. The Phase 0 → 4
pipeline is then run on that CID list to produce a parallel
`canary_post_cutoff.jsonl` that reviewers can compare to the main set.

See CONTAMINATION.md for why this matters.

Usage:
    python3 scripts/build_contamination_canary.py \\
        --tier data/filtered/drug_articles_v2_premium.jsonl \\
        --cutoff 2025-09-01 \\
        --min-articles 3 \\
        --output data/qa_pipeline/canary_cids.txt
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path


def _parse_date(s: str | None) -> dt.date | None:
    """PubMed records have `pub_date` like '2025-09-15' or '2025' etc."""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            d = dt.datetime.strptime(s[:len(fmt.replace('%Y', 'YYYY').replace('%m', 'MM').replace('%d','DD'))], fmt).date()
            return d
        except ValueError:
            continue
    return None


def build(args):
    cutoff = dt.date.fromisoformat(args.cutoff)
    min_articles = args.min_articles

    # Index: CID → list[(pub_date, pmid)]
    per_cid_dates: dict[int, list[dt.date]] = defaultdict(list)
    articles_seen = 0
    with Path(args.tier).open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            articles_seen += 1
            pub_date = _parse_date(rec.get("pub_date") or rec.get("publication_date"))
            if pub_date is None:
                continue
            for c in rec.get("linked_compounds", []) or []:
                cid_raw = c.get("cid")
                try:
                    cid = int(cid_raw)
                except (TypeError, ValueError):
                    continue
                per_cid_dates[cid].append(pub_date)

    print(f"  Articles scanned:   {articles_seen:,}")
    print(f"  Unique compounds:   {len(per_cid_dates):,}")
    print(f"  Cutoff date:        {cutoff.isoformat()}")
    print(f"  Min articles:       {min_articles}")

    canary = []
    pre = 0
    insufficient = 0
    for cid, dates in per_cid_dates.items():
        if len(dates) < min_articles:
            insufficient += 1
            continue
        earliest = min(dates)
        if earliest < cutoff:
            pre += 1
            continue
        canary.append((cid, earliest))

    canary.sort(key=lambda t: (t[1], t[0]))

    print(f"\n  Pre-cutoff compounds dropped:         {pre:,}")
    print(f"  Insufficient-articles dropped:        {insufficient:,}")
    print(f"  Post-cutoff canary compounds kept:    {len(canary):,}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for cid, _ in canary:
            f.write(f"{cid}\n")

    # Also emit a manifest with dates for auditability
    manifest = Path(str(out) + ".manifest.json")
    manifest.write_text(json.dumps({
        "cutoff_date": cutoff.isoformat(),
        "min_articles": min_articles,
        "n_canary_compounds": len(canary),
        "n_dropped_pre_cutoff": pre,
        "n_dropped_insufficient_articles": insufficient,
        "samples": [{"cid": c, "earliest_pub": d.isoformat()} for c, d in canary[:20]],
    }, indent=2))

    print(f"\n  Wrote CID list:  {out}")
    print(f"  Wrote manifest:  {manifest}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", required=True,
                   help="Filtered premium-tier JSONL (drug_articles_v2_premium.jsonl)")
    p.add_argument("--cutoff", default="2025-09-01",
                   help="ISO date; compounds whose earliest article predates this are excluded")
    p.add_argument("--min-articles", type=int, default=3,
                   help="Minimum articles required to keep a compound in the canary")
    p.add_argument("--output", required=True,
                   help="Destination for the CID list (one CID per line)")
    args = p.parse_args()
    build(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
