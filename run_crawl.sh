#!/usr/bin/env bash
# ============================================================
# Full crawl: DrugBank + HMDB + ChEBI + KEGG + BindingDB
#
# Usage:
#   bash run_crawl.sh --api-key YOUR_NCBI_KEY --email you@example.com
#   bash run_crawl.sh  # runs without API key (slower, 3 req/sec)
#
# Run in tmux:
#   tmux new -s crawl
#   bash run_crawl.sh --api-key YOUR_KEY --email you@example.com
#
# Resume after interruption (state is saved automatically):
#   bash run_crawl.sh --api-key YOUR_KEY --email you@example.com
#
# Monitor progress:
#   tail -f data/crawl.log
#   cat data/crawl_state.json | python3 -m json.tool
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

# Parse flags
NCBI_API_KEY=""
NCBI_EMAIL=""
MAX_DISK_GB=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-key)  NCBI_API_KEY="$2"; shift 2 ;;
        --email)    NCBI_EMAIL="$2"; shift 2 ;;
        --disk-gb)  MAX_DISK_GB="$2"; shift 2 ;;
        *)          echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Export as env vars (pydantic-settings picks these up automatically)
if [[ -n "$NCBI_API_KEY" ]]; then
    export NCBI_API_KEY
fi
if [[ -n "$NCBI_EMAIL" ]]; then
    export NCBI_EMAIL
fi

LOG="data/crawl.log"
mkdir -p data

echo "Starting crawl at $(date)" | tee -a "$LOG"
echo "Disk limit: ${MAX_DISK_GB} GB" | tee -a "$LOG"
if [[ -n "$NCBI_API_KEY" ]]; then
    echo "NCBI API key: configured (10 req/sec)" | tee -a "$LOG"
else
    echo "NCBI API key: NOT SET (3 req/sec — pass --api-key for 3x speed)" | tee -a "$LOG"
fi
if [[ -n "$NCBI_EMAIL" ]]; then
    echo "NCBI email: $NCBI_EMAIL" | tee -a "$LOG"
fi
echo "Log: $LOG" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

chem2textqa crawl \
    -s DrugBank \
    -s HMDB \
    -s ChEBI \
    -s KEGG \
    -s BindingDB \
    -n 999999 \
    -a 50000 \
    -t 500 \
    -r 50 \
    --full-text \
    --max-disk-gb "$MAX_DISK_GB" \
    2>&1 | tee -a "$LOG"

echo "---" | tee -a "$LOG"
echo "Finished at $(date)" | tee -a "$LOG"
echo "Output sizes:" | tee -a "$LOG"
ls -lh data/*.jsonl 2>/dev/null | tee -a "$LOG"
du -sh data/ | tee -a "$LOG"
