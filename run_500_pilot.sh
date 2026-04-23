#!/usr/bin/env bash
# ============================================================
# 500-compound pilot — Kimi vs Llama head-to-head (serial)
#
# SHARED across both runs (computed once):
#   - Phase 0: evidence bundles for 500 random premium drugs
#   - Phase 1: Gemini 3 Flash preview generates Q&A
#
# DIFFERENT per run:
#   - Phase 2: Kimi K2.5  vs  Llama 3.3 70B
#   - Phase 3: Gemma 4 31B judges each P2's answers vs the shared P1 answers
#   - Assembly
#
# This makes the comparison clean: same compounds, same questions, same
# Phase 1 answers. Only Phase 2 differs. Also saves ~$3 on duplicate P1 work.
#
# Expected wall time: ~1 hour
# Expected cost:      ~$12-15
#
# Usage:
#   tmux new -s pilot500
#   bash run_500_pilot.sh
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

# Load OPENROUTER_API_KEY from .env
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

CID_FILE="data/qa_pipeline/pilot_500_cids.txt"
if [[ ! -f "$CID_FILE" ]]; then
    echo "ERROR: $CID_FILE not found"
    exit 1
fi

LOG="data/qa_pipeline/pilot500.log"
mkdir -p data/qa_pipeline
echo "500-compound pilot started at $(date)" | tee "$LOG"
echo "Seed: 42 | CIDs: $(wc -l < "$CID_FILE")" | tee -a "$LOG"

# Output structure:
#   data/qa_pipeline/experiments/pilot500_shared/
#     phase_0_evidence/evidence_per_cid.jsonl
#     phase_1_qa/qa_pairs.jsonl           ← shared Phase 1 output
#
#   data/qa_pipeline/experiments/pilot500_kimi/
#     phase_2_independent/qa_independent.jsonl
#     phase_3_validate/validated.jsonl
#     dataset_final.jsonl
#
#   data/qa_pipeline/experiments/pilot500_llama/
#     (same structure)

SHARED_DIR="data/qa_pipeline/experiments/pilot500_shared"
SHARED_EVIDENCE="$SHARED_DIR/phase_0_evidence/evidence_per_cid.jsonl"
SHARED_P1="$SHARED_DIR/phase_1_qa/qa_pairs.jsonl"
SHARED_P1_ERR="$SHARED_DIR/phase_1_qa/errors.jsonl"
mkdir -p "$SHARED_DIR/phase_0_evidence" "$SHARED_DIR/phase_1_qa"

# ------------------------------------------------------------------
# Phase 0 (shared)
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "PHASE 0 — evidence for 500 drugs (shared)" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
if [[ -f "$SHARED_EVIDENCE" ]] && [[ $(wc -l < "$SHARED_EVIDENCE") -ge 450 ]]; then
    echo "  Evidence already exists ($(wc -l < "$SHARED_EVIDENCE") records) — skipping" | tee -a "$LOG"
else
    chem2textqa qa-extract-evidence \
        --input data/filtered/drug_articles_v2_premium.jsonl \
        --output "$SHARED_EVIDENCE" \
        --target-cids "$CID_FILE" 2>&1 | tee -a "$LOG"
fi

# ------------------------------------------------------------------
# Phase 1 (shared) — Gemini 3 Flash preview
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "PHASE 1 — Gemini 3 Flash preview (shared, 50 workers)" | tee -a "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
if [[ -f "$SHARED_P1" ]] && [[ $(wc -l < "$SHARED_P1") -ge 450 ]]; then
    echo "  Phase 1 already exists ($(wc -l < "$SHARED_P1") records) — skipping" | tee -a "$LOG"
else
    chem2textqa qa-generate \
        --input "$SHARED_EVIDENCE" \
        --output "$SHARED_P1" \
        --errors "$SHARED_P1_ERR" \
        --model google/gemini-3-flash-preview \
        --workers 50 2>&1 | tee -a "$LOG"
fi

# ------------------------------------------------------------------
# Helper: run Phase 2 → 3 → assembly for a named P2 model
# ------------------------------------------------------------------
run_p2_p3_assembly() {
    local name="$1"
    local p2_model="$2"

    local exp_dir="data/qa_pipeline/experiments/pilot500_${name}"
    local p2_out="$exp_dir/phase_2_independent/qa_independent.jsonl"
    local p2_err="$exp_dir/phase_2_independent/errors.jsonl"
    local p3_out="$exp_dir/phase_3_validate/validated.jsonl"
    mkdir -p "$exp_dir/phase_2_independent" "$exp_dir/phase_3_validate"

    echo "" | tee -a "$LOG"
    echo "===========================================" | tee -a "$LOG"
    echo "RUN: $name (Phase 2 = $p2_model)" | tee -a "$LOG"
    echo "  Started: $(date)" | tee -a "$LOG"
    echo "===========================================" | tee -a "$LOG"

    chem2textqa qa-independent \
        --input "$SHARED_P1" \
        --output "$p2_out" \
        --errors "$p2_err" \
        --model "$p2_model" \
        --workers 50 2>&1 | tee -a "$LOG"

    chem2textqa qa-judge \
        --input "$p2_out" \
        --output "$p3_out" \
        --model google/gemma-4-31b-it \
        --workers 50 2>&1 | tee -a "$LOG"

    chem2textqa qa-assemble \
        --phase0 "$SHARED_EVIDENCE" \
        --phase1 "$SHARED_P1" \
        --phase2 "$p2_out" \
        --phase3 "$p3_out" \
        --output-jsonl "$exp_dir/dataset_final.jsonl" \
        --output-json  "$exp_dir/dataset_final.json" \
        --summary      "$exp_dir/dataset_summary.json" 2>&1 | tee -a "$LOG"
}

# ------------------------------------------------------------------
# Run A: Kimi K2.5 as Phase 2
# ------------------------------------------------------------------
run_p2_p3_assembly "kimi" "moonshotai/kimi-k2.5"

# ------------------------------------------------------------------
# Run B: Llama 3.3 70B as Phase 2
# ------------------------------------------------------------------
run_p2_p3_assembly "llama" "meta-llama/llama-3.3-70b-instruct"

# ------------------------------------------------------------------
# Head-to-head comparison
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "HEAD-TO-HEAD COMPARISON" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-compare \
    --baseline   data/qa_pipeline/experiments/pilot500_kimi/dataset_final.jsonl \
    --experiment data/qa_pipeline/experiments/pilot500_llama/dataset_final.jsonl \
    --label-a kimi --label-b llama \
    --output data/qa_pipeline/pilot500_comparison.json 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Pilot finished at $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Shared inputs:" | tee -a "$LOG"
echo "  Evidence: $SHARED_EVIDENCE" | tee -a "$LOG"
echo "  Phase 1:  $SHARED_P1" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Per-run outputs:" | tee -a "$LOG"
echo "  Kimi:  data/qa_pipeline/experiments/pilot500_kimi/dataset_final.jsonl" | tee -a "$LOG"
echo "  Llama: data/qa_pipeline/experiments/pilot500_llama/dataset_final.jsonl" | tee -a "$LOG"
echo "  Comparison: data/qa_pipeline/pilot500_comparison.json" | tee -a "$LOG"
echo "  Full log:   $LOG" | tee -a "$LOG"
