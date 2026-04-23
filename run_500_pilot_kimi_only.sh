#!/usr/bin/env bash
# ============================================================
# 500-compound Kimi-only pilot (cost validation run)
#
# Uses a different random 500 CIDs (seed=7, disjoint from the
# first pilot) to get a second independent data point on cost
# and agree rate.
#
# Models: Gemini 3 Flash (P1) + Kimi K2.5 (P2) + Gemma 4 31B (P3)
# Workers: 50
#
# Usage:
#   tmux new -s pilot500b
#   bash run_500_pilot_kimi_only.sh
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set"
    exit 1
fi

CID_FILE="data/qa_pipeline/pilot_500b_cids.txt"
if [[ ! -f "$CID_FILE" ]]; then
    echo "ERROR: $CID_FILE not found"
    exit 1
fi

LOG="data/qa_pipeline/pilot500b.log"
SHARED_DIR="data/qa_pipeline/experiments/pilot500b_shared"
SHARED_EVIDENCE="$SHARED_DIR/phase_0_evidence/evidence_per_cid.jsonl"
SHARED_P1="$SHARED_DIR/phase_1_qa/qa_pairs.jsonl"
SHARED_P1_ERR="$SHARED_DIR/phase_1_qa/errors.jsonl"
EXP_DIR="data/qa_pipeline/experiments/pilot500b_kimi"
mkdir -p "$SHARED_DIR/phase_0_evidence" "$SHARED_DIR/phase_1_qa"
mkdir -p "$EXP_DIR/phase_2_independent" "$EXP_DIR/phase_3_validate"

echo "500b Kimi-only pilot started at $(date)" | tee "$LOG"
echo "Seed: 7 | CIDs: $(wc -l < "$CID_FILE")" | tee -a "$LOG"

# Phase 0
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "PHASE 0 — evidence for 500 drugs (seed=7)" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-extract-evidence \
    --input data/filtered/drug_articles_v2_premium.jsonl \
    --output "$SHARED_EVIDENCE" \
    --target-cids "$CID_FILE" 2>&1 | tee -a "$LOG"

# Phase 1 — Gemini 3 Flash
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "PHASE 1 — Gemini 3 Flash preview (50 workers)" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-generate \
    --input "$SHARED_EVIDENCE" \
    --output "$SHARED_P1" \
    --errors "$SHARED_P1_ERR" \
    --model google/gemini-3-flash-preview \
    --workers 50 2>&1 | tee -a "$LOG"

# Phase 2 — Kimi K2.5 only
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "PHASE 2 — Kimi K2.5 (50 workers)" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-independent \
    --input "$SHARED_P1" \
    --output "$EXP_DIR/phase_2_independent/qa_independent.jsonl" \
    --errors "$EXP_DIR/phase_2_independent/errors.jsonl" \
    --model moonshotai/kimi-k2.5 \
    --workers 50 2>&1 | tee -a "$LOG"

# Phase 3 — Gemma judge
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "PHASE 3 — Gemma 4 31B judge (50 workers)" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-judge \
    --input "$EXP_DIR/phase_2_independent/qa_independent.jsonl" \
    --output "$EXP_DIR/phase_3_validate/validated.jsonl" \
    --model google/gemma-4-31b-it \
    --workers 50 2>&1 | tee -a "$LOG"

# Assembly
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "ASSEMBLY" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-assemble \
    --phase0 "$SHARED_EVIDENCE" \
    --phase1 "$SHARED_P1" \
    --phase2 "$EXP_DIR/phase_2_independent/qa_independent.jsonl" \
    --phase3 "$EXP_DIR/phase_3_validate/validated.jsonl" \
    --output-jsonl "$EXP_DIR/dataset_final.jsonl" \
    --output-json  "$EXP_DIR/dataset_final.json" \
    --summary      "$EXP_DIR/dataset_summary.json" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Pilot finished at $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Outputs:" | tee -a "$LOG"
echo "  Final:   $EXP_DIR/dataset_final.jsonl" | tee -a "$LOG"
echo "  Summary: $EXP_DIR/dataset_summary.json" | tee -a "$LOG"
echo "  Log:     $LOG" | tee -a "$LOG"
