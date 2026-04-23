#!/usr/bin/env bash
# ============================================================
# Build v2 dataset: expanded compound set
# (MeSH + DrugBank + HMDB + KEGG + ChEBI = ~437K compounds)
#
# Usage:
#   tmux new -s build2
#   bash run_build_dataset_v2.sh
#
# Monitor:
#   tail -f data/build_v2.log
#   wc -l data/drug_articles_v2.jsonl
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

LOG="data/build_v2.log"
OUTPUT="data/drug_articles_v2.jsonl"

echo "Building v2 drug/metabolite dataset" | tee "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "  Output:  $OUTPUT" | tee -a "$LOG"
echo "  Compound set: MeSH + DrugBank + HMDB + KEGG + ChEBI (~437K)" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

chem2textqa build-dataset \
    --bulk-dir data/bulk \
    --output "$OUTPUT" \
    --fulltext data/bulk/pmc_fulltext.jsonl \
    --source DrugBank \
    --source HMDB \
    --source KEGG \
    --source ChEBI \
    2>&1 | tee -a "$LOG"

echo "---" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
if [ -f "$OUTPUT" ]; then
    echo "Articles: $(wc -l < "$OUTPUT")" | tee -a "$LOG"
    echo "Size: $(du -h "$OUTPUT" | cut -f1)" | tee -a "$LOG"
fi
