#!/usr/bin/env bash
# ============================================================
# Master runner — runs Phase 0 → 1 → 2 → 3 → assembly.
#
# Each phase is gated on the previous one succeeding, so a
# failure in Phase 1 won't burn money on Phase 2.
#
# Usage:
#   # Full premium tier (22K compounds, ~$1,100, ~6 hours)
#   bash run_qa_all.sh
#
#   # Pilot (20 compounds, ~$1, ~10 min)
#   bash run_qa_all.sh --cid-file data/qa_pipeline/pilot_cids.txt
#
#   # Agree-only final dataset (gold subset)
#   bash run_qa_all.sh --agree-only
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

# Load .env if present (OPENROUTER_API_KEY etc.)
if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

CID_FILE=""
AGREE_ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cid-file)   CID_FILE="$2"; shift 2 ;;
        --agree-only) AGREE_ONLY="--agree-only"; shift ;;
        *)            echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set (add to .env)"
    exit 1
fi

MASTER_LOG="data/qa_pipeline/master.log"
mkdir -p data/qa_pipeline

{
    echo "QA master pipeline"
    echo "  Started: $(date)"
    if [[ -n "$CID_FILE" ]]; then
        echo "  CID file: $CID_FILE"
    fi
    echo "==========================================="
} | tee "$MASTER_LOG"

run_phase() {
    local name="$1"
    shift
    echo "" | tee -a "$MASTER_LOG"
    echo ">>> $name ($(date))" | tee -a "$MASTER_LOG"
    if ! "$@" 2>&1 | tee -a "$MASTER_LOG"; then
        echo "" | tee -a "$MASTER_LOG"
        echo "!!! $name FAILED — aborting pipeline" | tee -a "$MASTER_LOG"
        exit 1
    fi
}

# Phase 0 — evidence extraction (no API cost)
if [[ -n "$CID_FILE" ]]; then
    run_phase "Phase 0" bash run_qa_phase0.sh --cid-file "$CID_FILE"
else
    run_phase "Phase 0" bash run_qa_phase0.sh
fi

# Phase 1 — QA generation (Claude Sonnet 4.6)
run_phase "Phase 1" bash run_qa_phase1.sh

# Phase 2 — independent answers (Kimi K2.5, reasoning disabled)
run_phase "Phase 2" bash run_qa_phase2.sh

# Phase 3 — cross-validation judge (Gemma 4 31B)
run_phase "Phase 3" bash run_qa_phase3.sh

# Assembly — merge all phases into one dataset
echo "" | tee -a "$MASTER_LOG"
echo ">>> Assembly ($(date))" | tee -a "$MASTER_LOG"
# shellcheck disable=SC2086
chem2textqa qa-assemble $AGREE_ONLY 2>&1 | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "===========================================" | tee -a "$MASTER_LOG"
echo "Pipeline finished at $(date)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Final outputs:" | tee -a "$MASTER_LOG"
echo "  data/qa_pipeline/dataset_final.jsonl   (streaming-friendly)" | tee -a "$MASTER_LOG"
echo "  data/qa_pipeline/dataset_final.json    (pretty-printed, <=5K compounds)" | tee -a "$MASTER_LOG"
echo "  data/qa_pipeline/dataset_summary.json  (stats)" | tee -a "$MASTER_LOG"
echo "  data/qa_pipeline/master.log            (full run log)" | tee -a "$MASTER_LOG"
