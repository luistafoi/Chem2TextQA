#!/usr/bin/env bash
# ============================================================
# Phase 1: structured Q&A generation via LLM1 (Gemini 2.5 Flash).
#
# Input:  data/qa_pipeline/phase_0_evidence/evidence_per_cid.jsonl
# Output: data/qa_pipeline/phase_1_qa/qa_pairs.jsonl
#
# Resumable — append-only output; re-running skips already-processed CIDs.
#
# Usage:
#   bash run_qa_phase1.sh --api-key YOUR_OPENROUTER_KEY
#   bash run_qa_phase1.sh --api-key ... --workers 40
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

# Load .env if present (sets OPENROUTER_API_KEY, etc.)
if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

API_KEY=""
WORKERS=20
MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-key)  API_KEY="$2"; shift 2 ;;
        --workers)  WORKERS="$2"; shift 2 ;;
        --model)    MODEL="$2"; shift 2 ;;
        *)          echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ -n "$API_KEY" ]]; then
    export OPENROUTER_API_KEY="$API_KEY"
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set. Either:"
    echo "  - add to .env file as OPENROUTER_API_KEY=sk-or-..."
    echo "  - export OPENROUTER_API_KEY=sk-or-..."
    echo "  - pass --api-key sk-or-... to this script"
    exit 1
fi

LOG="data/qa_pipeline/phase_1_qa/phase1.log"
mkdir -p "$(dirname "$LOG")"

echo "QA Phase 1 — generation" | tee "$LOG"
echo "  Workers: $WORKERS" | tee -a "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

ARGS=(--workers "$WORKERS")
if [[ -n "$MODEL" ]]; then
    ARGS+=(--model "$MODEL")
fi

chem2textqa qa-generate "${ARGS[@]}" 2>&1 | tee -a "$LOG"

echo "---" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
