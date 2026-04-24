#!/usr/bin/env bash
# ============================================================
# Full premium-tier QA generation (15,667 compounds).
#
# REUSES the cached Phase 0 evidence at:
#   data/qa_pipeline/phase0_full_premium_v3/evidence_per_cid.jsonl
# produced by run_phase0_full_premium.sh (500-cap, random sampling).
# Does NOT recompute it.
#
# Models:
#   Phase 1: Gemini 3 Flash preview
#   Phase 2: Kimi K2.5 (reasoning disabled)
#   Phase 3: Gemma 4 31B (paid, no rate limits)
#
# Pilot-extrapolated projections:
#   Raw QA:   ~152,000
#   Gold QA:  ~127,000  (at 83.5% agree rate from pilot A)
#   Cost:     ~$290
#   Time:     ~22 hours at 50 workers
#
# All phases are resumable — safe to kill and restart.
#
# Usage:
#   tmux new -s full
#   bash run_qa_full_premium.sh
#   # optionally: bash run_qa_full_premium.sh --workers 100
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
    echo "ERROR: OPENROUTER_API_KEY not set (add to .env)"
    exit 1
fi

WORKERS=50
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "ERROR: --workers requires a value"; exit 1
            fi
            WORKERS="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

EVIDENCE_DIR="data/qa_pipeline/phase0_full_premium_v3"
# The canary split is the contamination-sensitive held-out test.
# Processing it separately (but through the same pipeline in the same
# invocation) keeps its output traceable while amortising fixed costs.
MAIN_EVIDENCE="$EVIDENCE_DIR/evidence_main.jsonl"
CANARY_EVIDENCE="$EVIDENCE_DIR/evidence_canary.jsonl"
# Back-compat fallback: if the split files don't exist, use the combined file.
SHARED_EVIDENCE="${MAIN_EVIDENCE:-$EVIDENCE_DIR/evidence_per_cid.jsonl}"
if [[ ! -f "$SHARED_EVIDENCE" ]]; then
    SHARED_EVIDENCE="$EVIDENCE_DIR/evidence_per_cid.jsonl"
fi
if [[ ! -f "$SHARED_EVIDENCE" ]]; then
    echo "ERROR: Phase 0 evidence not found at $SHARED_EVIDENCE"
    echo "Run run_phase0_full_premium.sh first (writes to phase0_full_premium_v3/)."
    exit 1
fi

OUT_DIR="data/qa_pipeline/full_premium_kimi"
P1_OUT="$OUT_DIR/phase_1_qa/qa_pairs.jsonl"
P1_ERR="$OUT_DIR/phase_1_qa/errors.jsonl"
P2_OUT="$OUT_DIR/phase_2_independent/qa_independent.jsonl"
P2_ERR="$OUT_DIR/phase_2_independent/errors.jsonl"
P3_OUT="$OUT_DIR/phase_3_validate/validated.jsonl"
FINAL_JSONL="$OUT_DIR/dataset_final.jsonl"
FINAL_JSON="$OUT_DIR/dataset_final.json"
SUMMARY="$OUT_DIR/dataset_summary.json"
LOG="$OUT_DIR/run.log"

# Canary (separate outputs so reviewers can evaluate contamination impact
# independently — see CONTAMINATION.md)
CANARY_DIR="$OUT_DIR/canary"
CANARY_P1_OUT="$CANARY_DIR/phase_1_qa/qa_pairs.jsonl"
CANARY_P1_ERR="$CANARY_DIR/phase_1_qa/errors.jsonl"
CANARY_P2_OUT="$CANARY_DIR/phase_2_independent/qa_independent.jsonl"
CANARY_P2_ERR="$CANARY_DIR/phase_2_independent/errors.jsonl"
CANARY_P3_OUT="$CANARY_DIR/phase_3_validate/validated.jsonl"
CANARY_FINAL_JSONL="$CANARY_DIR/dataset_final.jsonl"

mkdir -p "$OUT_DIR/phase_1_qa" "$OUT_DIR/phase_2_independent" "$OUT_DIR/phase_3_validate"
mkdir -p "$CANARY_DIR/phase_1_qa" "$CANARY_DIR/phase_2_independent" "$CANARY_DIR/phase_3_validate"

EVIDENCE_COUNT=$(wc -l < "$SHARED_EVIDENCE")
echo "Full premium QA pipeline" | tee "$LOG"
echo "  Started:      $(date)" | tee -a "$LOG"
echo "  Main ev.:     $SHARED_EVIDENCE ($EVIDENCE_COUNT compounds)" | tee -a "$LOG"
if [[ -f "$CANARY_EVIDENCE" ]]; then
    CANARY_COUNT=$(wc -l < "$CANARY_EVIDENCE")
    echo "  Canary ev.:   $CANARY_EVIDENCE ($CANARY_COUNT compounds)" | tee -a "$LOG"
else
    echo "  Canary ev.:   (missing — skipping canary pass)" | tee -a "$LOG"
fi
echo "  Output dir:   $OUT_DIR" | tee -a "$LOG"
echo "  Workers:      $WORKERS" | tee -a "$LOG"
echo "  P1 model:     google/gemini-3-flash-preview" | tee -a "$LOG"
echo "  P2 model:     moonshotai/kimi-k2.5 (reasoning disabled)" | tee -a "$LOG"
echo "  P3 model:     google/gemma-4-31b-it (paid)" | tee -a "$LOG"

run_step() {
    local name="$1"
    shift
    echo "" | tee -a "$LOG"
    echo "===========================================" | tee -a "$LOG"
    echo ">>> $name ($(date))" | tee -a "$LOG"
    echo "===========================================" | tee -a "$LOG"
    if ! "$@" 2>&1 | tee -a "$LOG"; then
        echo "" | tee -a "$LOG"
        echo "!!! $name FAILED — aborting" | tee -a "$LOG"
        exit 1
    fi
}

# ------------------------------------------------------------------
# Phase 1 — Gemini 3 Flash preview
# ------------------------------------------------------------------
run_step "Phase 1 (Gemini 3 Flash)" \
    chem2textqa qa-generate \
        --input "$SHARED_EVIDENCE" \
        --output "$P1_OUT" \
        --errors "$P1_ERR" \
        --model google/gemini-3-flash-preview \
        --workers "$WORKERS"

# ------------------------------------------------------------------
# Phase 2 — Kimi K2.5 (reasoning disabled by default in our code)
# ------------------------------------------------------------------
run_step "Phase 2 (Kimi K2.5)" \
    chem2textqa qa-independent \
        --input "$P1_OUT" \
        --output "$P2_OUT" \
        --errors "$P2_ERR" \
        --model moonshotai/kimi-k2.5 \
        --workers "$WORKERS"

# ------------------------------------------------------------------
# Phase 3 — Gemma 4 31B
# ------------------------------------------------------------------
run_step "Phase 3 (Gemma 4 31B)" \
    chem2textqa qa-judge \
        --input "$P2_OUT" \
        --output "$P3_OUT" \
        --model google/gemma-4-31b-it \
        --workers "$WORKERS"

# ------------------------------------------------------------------
# Assembly
# ------------------------------------------------------------------
run_step "Assembly" \
    chem2textqa qa-assemble \
        --phase0 "$SHARED_EVIDENCE" \
        --phase1 "$P1_OUT" \
        --phase2 "$P2_OUT" \
        --phase3 "$P3_OUT" \
        --output-jsonl "$FINAL_JSONL" \
        --output-json  "$FINAL_JSON" \
        --summary      "$SUMMARY"

# ------------------------------------------------------------------
# Gold-only variant (agree verdicts only)
# ------------------------------------------------------------------
run_step "Assembly (gold-only)" \
    chem2textqa qa-assemble \
        --phase0 "$SHARED_EVIDENCE" \
        --phase1 "$P1_OUT" \
        --phase2 "$P2_OUT" \
        --phase3 "$P3_OUT" \
        --output-jsonl "$OUT_DIR/dataset_gold.jsonl" \
        --output-json  "$OUT_DIR/dataset_gold.json" \
        --summary      "$OUT_DIR/dataset_gold_summary.json" \
        --agree-only

# ------------------------------------------------------------------
# Canary pass (contamination-sensitive held-out set, ~120 compounds)
# Skipped if the canary evidence file doesn't exist.
# ------------------------------------------------------------------
if [[ -f "$CANARY_EVIDENCE" ]]; then
    run_step "Canary Phase 1 (Gemini 3 Flash)" \
        chem2textqa qa-generate \
            --input "$CANARY_EVIDENCE" \
            --output "$CANARY_P1_OUT" \
            --errors "$CANARY_P1_ERR" \
            --model google/gemini-3-flash-preview \
            --workers "$WORKERS"

    run_step "Canary Phase 2 (Kimi K2.5)" \
        chem2textqa qa-independent \
            --input "$CANARY_P1_OUT" \
            --output "$CANARY_P2_OUT" \
            --errors "$CANARY_P2_ERR" \
            --model moonshotai/kimi-k2.5 \
            --workers "$WORKERS"

    run_step "Canary Phase 3 (Gemma 4 31B)" \
        chem2textqa qa-judge \
            --input "$CANARY_P2_OUT" \
            --output "$CANARY_P3_OUT" \
            --model google/gemma-4-31b-it \
            --workers "$WORKERS"

    run_step "Canary Assembly" \
        chem2textqa qa-assemble \
            --phase0 "$CANARY_EVIDENCE" \
            --phase1 "$CANARY_P1_OUT" \
            --phase2 "$CANARY_P2_OUT" \
            --phase3 "$CANARY_P3_OUT" \
            --output-jsonl "$CANARY_FINAL_JSONL" \
            --summary      "$CANARY_DIR/dataset_summary.json"
fi

echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "FULL PREMIUM RUN COMPLETE" | tee -a "$LOG"
echo "  Finished: $(date)" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Outputs:" | tee -a "$LOG"
echo "  Main all QA:    $FINAL_JSONL" | tee -a "$LOG"
echo "  Main gold:      $OUT_DIR/dataset_gold.jsonl" | tee -a "$LOG"
echo "  Main summary:   $SUMMARY" | tee -a "$LOG"
if [[ -f "$CANARY_EVIDENCE" ]]; then
    echo "  Canary all QA:  $CANARY_FINAL_JSONL" | tee -a "$LOG"
    echo "  Canary summary: $CANARY_DIR/dataset_summary.json" | tee -a "$LOG"
fi
echo "  Full log:       $LOG" | tee -a "$LOG"
