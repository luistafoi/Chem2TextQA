#!/usr/bin/env bash
# ============================================================
# Filter v2 dataset (same cleanup rules as v1)
#
# Input:  data/drug_articles_v2.jsonl
# Output: data/filtered/drug_articles_v2_filtered.jsonl
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p data/filtered
LOG="data/filtered/cleanup_v2.log"

echo "Filtering v2 drug/metabolite dataset" | tee "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

chem2textqa cleanup-dataset \
    --input data/drug_articles_v2.jsonl \
    --output data/filtered/drug_articles_v2_filtered.jsonl \
    --stats data/filtered/filter_stats_v2.json \
    2>&1 | tee -a "$LOG"

echo "---" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
if [ -f data/filtered/drug_articles_v2_filtered.jsonl ]; then
    echo "Output: $(wc -l < data/filtered/drug_articles_v2_filtered.jsonl) articles" | tee -a "$LOG"
    echo "Size:   $(du -h data/filtered/drug_articles_v2_filtered.jsonl | cut -f1)" | tee -a "$LOG"
fi
