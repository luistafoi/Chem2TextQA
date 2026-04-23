#!/usr/bin/env bash
# ============================================================
# Run Phase 0 on the full premium tier — v3 regeneration.
#
# Extracts evidence for all 22,438 premium-tier compounds under
# the current pipeline:
#   - 500-sentence cap per compound (config.py)
#   - random sampling across all the compound's articles
#     (deduplicated, per-CID-seeded RNG for reproducibility)
#
# No LLM calls — $0 cost. ~1–2 hours.
#
# Outputs (v3 path, leaves v1 cache at phase0_full_premium/ intact):
#   data/qa_pipeline/phase0_full_premium_v3/evidence_per_cid.jsonl
#   data/qa_pipeline/phase0_full_premium_v3/retention_stats.json
#
# Usage:
#   tmux new -s p0full
#   bash run_phase0_full_premium.sh
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="data/qa_pipeline/phase0_full_premium_v3"
OUTPUT="$OUT_DIR/evidence_per_cid.jsonl"
STATS="$OUT_DIR/retention_stats.json"
LOG="$OUT_DIR/phase0_full.log"

mkdir -p "$OUT_DIR"

echo "Phase 0 on full premium tier" | tee "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "  Output:  $OUTPUT" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

chem2textqa qa-extract-evidence \
    --input data/filtered/drug_articles_v2_premium.jsonl \
    --output "$OUTPUT" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "Computing retention statistics..." | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"

python3 << PYEOF 2>&1 | tee -a "$LOG"
import json
from collections import Counter
from pathlib import Path

evidence_path = Path("$OUTPUT")
premium_path = Path("data/filtered/drug_articles_v2_premium.jsonl")

# Count compounds in premium tier
premium_cids = set()
article_count_by_cid = Counter()
with premium_path.open() as f:
    for line in f:
        rec = json.loads(line)
        for c in rec.get("linked_compounds", []):
            cid = c.get("cid")
            if cid is not None:
                premium_cids.add(int(cid))
                article_count_by_cid[int(cid)] += 1

# Count compounds with evidence
evidence_cids = []
sentence_count_by_cid = {}
synonym_count_by_cid = {}
pmids_by_cid = {}
with evidence_path.open() as f:
    for line in f:
        rec = json.loads(line)
        cid = int(rec["cid"])
        evidence_cids.append(cid)
        sentence_count_by_cid[cid] = len(rec.get("evidence_sentences", []))
        synonym_count_by_cid[cid] = rec.get("num_synonyms", 0)
        pmids_by_cid[cid] = rec.get("num_pmids", 0)

kept = set(evidence_cids)
dropped = premium_cids - kept

# Why dropped? Count the article distribution for dropped compounds
dropped_articles = Counter()
for cid in dropped:
    dropped_articles[article_count_by_cid[cid]] += 1

# Stats by sentence bucket — aligned with the current cap from config.
from chem2textqa.qa_pipeline.config import MAX_EVIDENCE_SENTENCES_PER_COMPOUND as CAP
# Ordered so the cap bucket appears at the top end regardless of value.
bucket_edges = [1, 5, 10, 20, 50, 100, 200, 300, CAP]
bucket_edges = sorted(set(bucket_edges))
sentence_buckets = Counter()
at_cap = 0
for cnt in sentence_count_by_cid.values():
    if cnt >= CAP:
        sentence_buckets[f"{CAP} (cap)"] += 1
        at_cap += 1
        continue
    # Find the largest lower edge <= cnt
    placed = False
    for i in range(len(bucket_edges) - 1, -1, -1):
        lo = bucket_edges[i]
        if lo == CAP:
            continue
        if cnt >= lo:
            hi = bucket_edges[i + 1] - 1 if i + 1 < len(bucket_edges) else CAP - 1
            # Cap the upper edge at CAP-1
            hi = min(hi, CAP - 1)
            sentence_buckets[f"{lo}-{hi}"] += 1
            placed = True
            break
    if not placed:
        sentence_buckets["1-4"] += 1

stats = {
    "premium_compounds_total": len(premium_cids),
    "compounds_with_evidence": len(kept),
    "compounds_dropped": len(dropped),
    "retention_rate": len(kept) / len(premium_cids) if premium_cids else 0,
    "avg_evidence_sentences_per_compound": (
        sum(sentence_count_by_cid.values()) / len(sentence_count_by_cid)
        if sentence_count_by_cid else 0
    ),
    "total_evidence_sentences": sum(sentence_count_by_cid.values()),
    "dropped_compound_article_distribution": dict(dropped_articles.most_common()),
    "kept_compound_sentence_buckets": dict(sentence_buckets.most_common()),
}

with open("$STATS", "w") as f:
    json.dump(stats, f, indent=2)

pct = stats["retention_rate"] * 100
print()
print(f"  Premium compounds total:      {stats['premium_compounds_total']:>8,}")
print(f"  Kept (have evidence):         {stats['compounds_with_evidence']:>8,}  ({pct:.1f}%)")
print(f"  Dropped (no evidence):        {stats['compounds_dropped']:>8,}  ({100-pct:.1f}%)")
print(f"  Total evidence sentences:     {stats['total_evidence_sentences']:>8,}")
print(f"  Avg sentences/kept compound:  {stats['avg_evidence_sentences_per_compound']:>8.1f}")
print()
print("  Dropped compounds by article count (top 10):")
for k, v in list(stats["dropped_compound_article_distribution"].items())[:10]:
    print(f"    {k:>3} articles: {v:>6,} dropped")
print()
print("  Kept compounds by evidence-sentence bucket:")
# Print in ascending-lower-edge order so the table reads naturally.
def _sort_key(item):
    label, _ = item
    if label.endswith("(cap)"):
        return (10**9,)
    lo = int(label.split("-", 1)[0])
    return (lo,)
for k, v in sorted(stats["kept_compound_sentence_buckets"].items(), key=_sort_key):
    print(f"    {k:<14} {v:>8,}")
print(f"  At-cap compounds (cap={CAP}):  {at_cap:>8,}")
PYEOF

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
echo "Stats written to: $STATS" | tee -a "$LOG"
