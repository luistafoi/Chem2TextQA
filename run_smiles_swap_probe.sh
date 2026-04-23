#!/usr/bin/env bash
# ============================================================
# SMILES-swap probe — does the model actually read each unique SMILES?
#
# For each of the 50 ablation compounds, we pair it with a random other
# compound and build a hybrid evidence record:
#   evidence of A + SMILES/formula/MW of B
#
# Run Phase 1 on these hybrids, then compare each output to two
# baselines from the ablation run:
#   • Phase 1 on (evidence_A + SMILES_A) — the evidence-owner's real run
#   • Phase 1 on (evidence_B + SMILES_B) — the SMILES-donor's real run
#
# If hybrid outputs are closer to the SMILES donor's real run, the model
# is reading each unique SMILES (clean). If closer to the evidence
# owner's real run, identity is leaking through the redacted evidence.
#
# Depends on run_ablation_probe.sh already having completed:
#   ablation50/real/evidence.jsonl
#   ablation50/real/qa_pairs.jsonl
#
# Cost: ~$0.25 (50 Phase 1 calls on Gemini 3 Flash).
# Time: ~30 seconds.
#
# Usage:
#   bash run_smiles_swap_probe.sh
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

ABLATION_DIR="data/qa_pipeline/experiments/ablation50"
REAL_EV="$ABLATION_DIR/real/evidence.jsonl"
REAL_QA="$ABLATION_DIR/real/qa_pairs.jsonl"

if [[ ! -f "$REAL_EV" || ! -f "$REAL_QA" ]]; then
    echo "ERROR: ablation baseline missing. Run run_ablation_probe.sh first."
    exit 1
fi

BASE="data/qa_pipeline/experiments/smiles_swap50"
HYBRID_EV="$BASE/hybrid_evidence.jsonl"
HYBRID_QA="$BASE/qa_pairs.jsonl"
HYBRID_MAP="$BASE/hybrid_map.json"
REPORT="$BASE/swap_report.json"
LOG="$BASE/swap.log"

mkdir -p "$BASE"
echo "SMILES-swap probe started at $(date)" | tee "$LOG"

# Clear stale Phase 1 output from any prior run so we regenerate fresh.
rm -f "$HYBRID_QA" "$BASE/errors.jsonl"

# ------------------------------------------------------------------
# Step 1 — build hybrid evidence (evidence_A + SMILES_B per pair)
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "STEP 1 — build hybrid evidence" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
python3 scripts/smiles_swap_probe.py prepare \
    --real-evidence "$REAL_EV" \
    --seed 17 \
    --output "$HYBRID_EV" \
    --mapping "$HYBRID_MAP" 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------------
# Step 2 — Phase 1 on hybrid
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "STEP 2 — Phase 1 on hybrid evidence" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-generate \
    --input "$HYBRID_EV" \
    --output "$HYBRID_QA" \
    --errors "$BASE/errors.jsonl" \
    --model google/gemini-3-flash-preview \
    --workers 50 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------------
# Step 3 — compare to real baseline
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "STEP 3 — compare" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
python3 scripts/smiles_swap_probe.py compare \
    --real "$REAL_QA" \
    --hybrid "$HYBRID_QA" \
    --mapping "$HYBRID_MAP" \
    --output "$REPORT" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "SMILES-swap probe finished at $(date)" | tee -a "$LOG"
echo "Report: $REPORT" | tee -a "$LOG"
echo "Log:    $LOG" | tee -a "$LOG"
