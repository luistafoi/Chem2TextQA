#!/usr/bin/env bash
# ============================================================
# 100-compound pilot — improved prompts + heuristic Phase 3
#
# Tests three changes vs the 500b baseline (83.4% agree):
#   1. Heuristic Phase 3 pre-filter (auto-agree on high-jaccard pairs)
#   2. Judge prompt with 3 worked examples
#   3. Phase 1 prompt with worked example + evidence-proportional
#      question counts + stricter forbidden-phrase list + anti-redundancy
#
# Evidence is sliced from the cached full-premium Phase 0 (seed=11) —
# no re-extraction cost.
#
# Models unchanged from baseline: Gemini 3 Flash + Kimi K2.5 + Gemma 4 31B.
#
# Usage:
#   tmux new -s pilot100
#   bash run_100_pilot_improved.sh
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

EXP_DIR="data/qa_pipeline/experiments/pilot100_improved"
SHARED_DIR="data/qa_pipeline/experiments/pilot100_improved_shared"
SHARED_EVIDENCE="$SHARED_DIR/phase_0_evidence/evidence_per_cid.jsonl"

if [[ ! -f "$SHARED_EVIDENCE" ]]; then
    echo "ERROR: $SHARED_EVIDENCE missing — sample it with pilot_100 prep script"
    exit 1
fi

mkdir -p "$EXP_DIR/phase_1_qa" "$EXP_DIR/phase_2_independent" "$EXP_DIR/phase_3_validate"

LOG="$EXP_DIR/pilot100.log"
echo "100-compound improved-prompt pilot started at $(date)" | tee "$LOG"
echo "Evidence: $SHARED_EVIDENCE ($(wc -l < "$SHARED_EVIDENCE") compounds)" | tee -a "$LOG"

# ------------------------------------------------------------------
# Phase 1 — Gemini 3 Flash with improved prompt
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "PHASE 1 — Gemini 3 Flash preview (improved prompt)" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-generate \
    --input  "$SHARED_EVIDENCE" \
    --output "$EXP_DIR/phase_1_qa/qa_pairs.jsonl" \
    --errors "$EXP_DIR/phase_1_qa/errors.jsonl" \
    --model  google/gemini-3-flash-preview \
    --workers 50 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------------
# Phase 2 — Kimi K2.5 (reasoning disabled)
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "PHASE 2 — Kimi K2.5" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-independent \
    --input  "$EXP_DIR/phase_1_qa/qa_pairs.jsonl" \
    --output "$EXP_DIR/phase_2_independent/qa_independent.jsonl" \
    --errors "$EXP_DIR/phase_2_independent/errors.jsonl" \
    --model  moonshotai/kimi-k2.5 \
    --workers 50 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------------
# Phase 3 — Gemma 4 31B with heuristic pre-filter + new prompt
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "PHASE 3 — Gemma 4 31B (heuristic + worked-example prompt)" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-judge \
    --input  "$EXP_DIR/phase_2_independent/qa_independent.jsonl" \
    --output "$EXP_DIR/phase_3_validate/validated.jsonl" \
    --model  google/gemma-4-31b-it \
    --workers 50 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------------
# Assembly
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "ASSEMBLY" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-assemble \
    --phase0 "$SHARED_EVIDENCE" \
    --phase1 "$EXP_DIR/phase_1_qa/qa_pairs.jsonl" \
    --phase2 "$EXP_DIR/phase_2_independent/qa_independent.jsonl" \
    --phase3 "$EXP_DIR/phase_3_validate/validated.jsonl" \
    --output-jsonl "$EXP_DIR/dataset_final.jsonl" \
    --output-json  "$EXP_DIR/dataset_final.json" \
    --summary      "$EXP_DIR/dataset_summary.json" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Pilot finished at $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Baseline to beat: 83.4% agree (pilot500b)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Outputs:" | tee -a "$LOG"
echo "  Final:   $EXP_DIR/dataset_final.jsonl" | tee -a "$LOG"
echo "  Summary: $EXP_DIR/dataset_summary.json" | tee -a "$LOG"
echo "  Log:     $LOG" | tee -a "$LOG"
