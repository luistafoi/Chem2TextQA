#!/usr/bin/env bash
# ============================================================
# Phase 0: build per-compound evidence bundles.
#
# Input:  data/filtered/drug_articles_v2_premium.jsonl (320K articles, 22K compounds)
# Output: data/qa_pipeline/phase_0_evidence/evidence_per_cid.jsonl
#
# Optional: pass --cid-file to restrict to a pilot subset.
#
# Usage:
#   tmux new -s qa0
#   bash run_qa_phase0.sh                            # full premium tier
#   bash run_qa_phase0.sh --cid-file pilot_cids.txt  # pilot
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

INPUT="data/filtered/drug_articles_v2_premium.jsonl"
OUTPUT="data/qa_pipeline/phase_0_evidence/evidence_per_cid.jsonl"
LOG="data/qa_pipeline/phase_0_evidence/phase0.log"
CID_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cid-file)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "ERROR: --cid-file requires a path argument"; exit 1
            fi
            CID_FILE="$2"; shift 2 ;;
        --input)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "ERROR: --input requires a path argument"; exit 1
            fi
            INPUT="$2"; shift 2 ;;
        --output)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "ERROR: --output requires a path argument"; exit 1
            fi
            OUTPUT="$2"; shift 2 ;;
        *)  echo "Unknown flag: $1"; exit 1 ;;
    esac
done

mkdir -p "$(dirname "$OUTPUT")"

echo "QA Phase 0 — evidence extraction" | tee "$LOG"
echo "  Input:   $INPUT" | tee -a "$LOG"
echo "  Output:  $OUTPUT" | tee -a "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

ARGS=(--input "$INPUT" --output "$OUTPUT")
if [[ -n "$CID_FILE" ]]; then
    ARGS+=(--target-cids "$CID_FILE")
fi

chem2textqa qa-extract-evidence "${ARGS[@]}" 2>&1 | tee -a "$LOG"

echo "---" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
