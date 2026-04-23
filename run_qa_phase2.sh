#!/usr/bin/env bash
# ============================================================
# Phase 2: independent answers via LLM2 (DeepSeek Chat).
#
# LLM2 answers each question blind to LLM1's answer, using only the
# redacted evidence. This enables Phase 3 cross-validation.
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

# Load .env if present
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
    echo "ERROR: OPENROUTER_API_KEY not set (add to .env or pass --api-key)"
    exit 1
fi

LOG="data/qa_pipeline/phase_2_independent/phase2.log"
mkdir -p "$(dirname "$LOG")"

echo "QA Phase 2 — independent answers" | tee "$LOG"
echo "  Workers: $WORKERS" | tee -a "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

ARGS=(--workers "$WORKERS")
if [[ -n "$MODEL" ]]; then
    ARGS+=(--model "$MODEL")
fi

chem2textqa qa-independent "${ARGS[@]}" 2>&1 | tee -a "$LOG"

echo "---" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
