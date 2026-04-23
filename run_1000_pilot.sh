#!/usr/bin/env bash
# ============================================================
# 1000-compound pilot — current soft-rule pipeline
#
# Pipeline:
#   Phase 0 — 500-sentence cap, RANDOM sampling across all articles
#             (`iter_compound_evidence`, seeded per-CID for reproducibility)
#   Phase 1 — Gemini 3 Flash preview + soft-rule prompt:
#               structural claims from SMILES,
#               functional claims from evidence silently,
#               freeform topic tags (taxonomy as inspiration only),
#               evidence-proportional counts (5–7 / 10–15 / 15–25 / 25–35 / 35–50)
#   Phase 2 — Kimi K2.5 (reasoning disabled), mirrored soft rule
#   Phase 3 — Gemma 4 31B + heuristic pre-filter + worked-example judge
#
# CIDs are sampled (seed=19) from the 15,667 compounds that pass Phase 0
# extraction, so every sampled CID is guaranteed to have usable evidence.
#
# Expected: ~13–15K raw Q&A, ~11–13K gold. Cost ~$35–45. Time ~20–30 min
# at 50 workers (Phase 2 dominates — every question re-sends evidence).
#
# Usage:
#   tmux new -s pilot1000
#   bash run_1000_pilot.sh
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

CID_FILE="data/qa_pipeline/pilot_1000_cids.txt"
if [[ ! -f "$CID_FILE" ]]; then
    echo "ERROR: $CID_FILE missing"
    exit 1
fi

EXP_DIR="data/qa_pipeline/experiments/pilot1000_random200"
P0_OUT="$EXP_DIR/phase_0_evidence/evidence_per_cid.jsonl"
P1_OUT="$EXP_DIR/phase_1_qa/qa_pairs.jsonl"
P2_OUT="$EXP_DIR/phase_2_independent/qa_independent.jsonl"
P3_OUT="$EXP_DIR/phase_3_validate/validated.jsonl"
FINAL_JSONL="$EXP_DIR/dataset_final.jsonl"
FINAL_JSON="$EXP_DIR/dataset_final.json"
SUMMARY="$EXP_DIR/dataset_summary.json"
LOG="$EXP_DIR/pilot1000.log"

mkdir -p "$EXP_DIR/phase_0_evidence" "$EXP_DIR/phase_1_qa" \
         "$EXP_DIR/phase_2_independent" "$EXP_DIR/phase_3_validate"

echo "1000-compound pilot started at $(date)" | tee "$LOG"
echo "  Seed: 19 | CIDs: $(wc -l < "$CID_FILE")" | tee -a "$LOG"
echo "  Cap:  200 sentences, random sampled across articles" | tee -a "$LOG"

# ------------------------------------------------------------------
# Phase 0 — re-extract at cap=200, random
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "PHASE 0 — random-sampled evidence (cap=200)" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-extract-evidence \
    --input data/filtered/drug_articles_v2_premium.jsonl \
    --output "$P0_OUT" \
    --target-cids "$CID_FILE" 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------------
# Phase 1 — Gemini 3 Flash + improved prompt
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "PHASE 1 — Gemini 3 Flash preview" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-generate \
    --input "$P0_OUT" \
    --output "$P1_OUT" \
    --errors "$EXP_DIR/phase_1_qa/errors.jsonl" \
    --model google/gemini-3-flash-preview \
    --workers 50 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------------
# Phase 2 — Kimi K2.5
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "PHASE 2 — Kimi K2.5" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-independent \
    --input "$P1_OUT" \
    --output "$P2_OUT" \
    --errors "$EXP_DIR/phase_2_independent/errors.jsonl" \
    --model moonshotai/kimi-k2.5 \
    --workers 50 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------------
# Phase 3 — Gemma 4 31B + heuristic pre-filter + worked-example judge
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "PHASE 3 — Gemma 4 31B" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-judge \
    --input "$P2_OUT" \
    --output "$P3_OUT" \
    --model google/gemma-4-31b-it \
    --workers 50 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------------
# Assembly — both full and gold-only
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "ASSEMBLY" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-assemble \
    --phase0 "$P0_OUT" \
    --phase1 "$P1_OUT" \
    --phase2 "$P2_OUT" \
    --phase3 "$P3_OUT" \
    --output-jsonl "$FINAL_JSONL" \
    --output-json  "$FINAL_JSON" \
    --summary      "$SUMMARY" 2>&1 | tee -a "$LOG"

chem2textqa qa-assemble \
    --phase0 "$P0_OUT" \
    --phase1 "$P1_OUT" \
    --phase2 "$P2_OUT" \
    --phase3 "$P3_OUT" \
    --output-jsonl "$EXP_DIR/dataset_gold.jsonl" \
    --output-json  "$EXP_DIR/dataset_gold.json" \
    --summary      "$EXP_DIR/dataset_gold_summary.json" \
    --agree-only 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Pilot finished at $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Outputs:" | tee -a "$LOG"
echo "  All QA:    $FINAL_JSONL" | tee -a "$LOG"
echo "  Gold only: $EXP_DIR/dataset_gold.jsonl" | tee -a "$LOG"
echo "  Summary:   $SUMMARY" | tee -a "$LOG"
echo "  Log:       $LOG" | tee -a "$LOG"
