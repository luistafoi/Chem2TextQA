#!/usr/bin/env bash
# ============================================================
# Ablation probe — does the private topic hint actually steer Phase 1,
# or does the model answer from internal recall of the SMILES?
#
# Pipeline:
#   1. Sample 50 compounds from the cached Phase 0 evidence (seed=13).
#   2. Build TWO evidence files for them:
#        real       — original redacted evidence
#        scrambled  — same compounds, but every evidence sentence is
#                     replaced with a random sentence from a DIFFERENT
#                     compound's pool.
#   3. Run Phase 1 on both (Gemini 3 Flash preview, 50 workers).
#   4. Compare outputs per-compound (topic Jaccard, best-question Jaccard,
#      best-answer Jaccard).
#
# Cost: ~$0.50 (2 × 50 Phase 1 calls on Gemini 3 Flash).
# Time: ~1 minute.
#
# Usage:
#   tmux new -s ablation
#   bash run_ablation_probe.sh
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

SRC_EVIDENCE="data/qa_pipeline/phase0_full_premium/evidence_per_cid.jsonl"
if [[ ! -f "$SRC_EVIDENCE" ]]; then
    echo "ERROR: cached Phase 0 evidence missing at $SRC_EVIDENCE"
    exit 1
fi

BASE="data/qa_pipeline/experiments/ablation50"
REAL_DIR="$BASE/real"
SCR_DIR="$BASE/scrambled"
REAL_EV="$REAL_DIR/evidence.jsonl"
SCR_EV="$SCR_DIR/evidence.jsonl"
REAL_QA="$REAL_DIR/qa_pairs.jsonl"
SCR_QA="$SCR_DIR/qa_pairs.jsonl"
REPORT="$BASE/ablation_report.json"
LOG="$BASE/ablation.log"

mkdir -p "$REAL_DIR" "$SCR_DIR"
echo "Ablation probe started at $(date)" | tee "$LOG"

# ------------------------------------------------------------------
# Step 1 — sample + scramble
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "STEP 1 — prepare real + scrambled evidence" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
python3 scripts/ablation_probe.py prepare \
    --input "$SRC_EVIDENCE" \
    -n 50 --seed 13 \
    --output-real "$REAL_EV" \
    --output-scrambled "$SCR_EV" 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------------
# Step 2 — Phase 1 on real
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "STEP 2 — Phase 1 on REAL evidence" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-generate \
    --input "$REAL_EV" \
    --output "$REAL_QA" \
    --errors "$REAL_DIR/errors.jsonl" \
    --model google/gemini-3-flash-preview \
    --workers 50 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------------
# Step 3 — Phase 1 on scrambled
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "STEP 3 — Phase 1 on SCRAMBLED evidence" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-generate \
    --input "$SCR_EV" \
    --output "$SCR_QA" \
    --errors "$SCR_DIR/errors.jsonl" \
    --model google/gemini-3-flash-preview \
    --workers 50 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------------
# Step 4 — compare
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "STEP 4 — compare" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
python3 scripts/ablation_probe.py compare \
    --real "$REAL_QA" \
    --scrambled "$SCR_QA" \
    --output "$REPORT" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Ablation finished at $(date)" | tee -a "$LOG"
echo "Report: $REPORT" | tee -a "$LOG"
echo "Log:    $LOG" | tee -a "$LOG"
