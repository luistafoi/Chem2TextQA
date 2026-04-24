"""Split a Phase 0 evidence file into main + canary shards, tagging each
canary record with `is_canary: true` so assembly correctly assigns the
`split: "canary"` label.

Idempotent: safe to re-run after Phase 0 regeneration.

Usage:
    python3 scripts/split_main_canary.py \\
        --evidence data/qa_pipeline/phase0_full_premium_v3/evidence_per_cid.jsonl \\
        --canary-cids data/qa_pipeline/canary_cids.txt \\
        --out-main data/qa_pipeline/phase0_full_premium_v3/evidence_main.jsonl \\
        --out-canary data/qa_pipeline/phase0_full_premium_v3/evidence_canary.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", required=True)
    p.add_argument("--canary-cids", required=True)
    p.add_argument("--out-main", required=True)
    p.add_argument("--out-canary", required=True)
    args = p.parse_args()

    canary_ids: set[int] = set()
    with Path(args.canary_cids).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                canary_ids.add(int(line))
    print(f"  Canary CIDs loaded: {len(canary_ids)}")

    main_out = Path(args.out_main)
    can_out = Path(args.out_canary)
    main_out.parent.mkdir(parents=True, exist_ok=True)
    can_out.parent.mkdir(parents=True, exist_ok=True)

    main_n = can_n = 0
    with Path(args.evidence).open("r", encoding="utf-8") as src, \
         main_out.open("w", encoding="utf-8") as m, \
         can_out.open("w", encoding="utf-8") as c:
        for line in src:
            rec = json.loads(line)
            cid = int(rec["cid"])
            if cid in canary_ids:
                rec["is_canary"] = True
                c.write(json.dumps(rec, ensure_ascii=False) + "\n")
                can_n += 1
            else:
                rec.pop("is_canary", None)
                m.write(json.dumps(rec, ensure_ascii=False) + "\n")
                main_n += 1

    print(f"  Main:   {main_n:,} records → {main_out}")
    print(f"  Canary: {can_n:,} records → {can_out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
