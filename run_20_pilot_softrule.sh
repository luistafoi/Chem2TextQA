#!/usr/bin/env bash
# ============================================================
# 20-compound pilot — soft-rule design sanity check
#
# What's being tested:
#   • Phase 0 — 500-sentence cap, random sampling across all articles
#   • Phase 1 — soft rule (structural from SMILES, functional from
#               evidence silently), freeform topic tags, engineering
#               questions, scaled count (35–50 for compounds with 300+
#               evidence sentences)
#   • Phase 2 — mirrored soft rule for blind re-answer
#   • Phase 3 — existing judge (agree/disagree/unclear)
#
# No baseline A/B — just produce outputs so you can eyeball quality
# before committing to the 1000-compound run.
#
# Cost: tiny (~$0.50). Time: ~5 minutes.
#
# Usage:
#   bash run_20_pilot_softrule.sh
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

CID_FILE="data/qa_pipeline/pilot_20_softrule_cids.txt"
if [[ ! -f "$CID_FILE" ]]; then
    echo "ERROR: $CID_FILE missing"
    exit 1
fi

EXP_DIR="data/qa_pipeline/experiments/pilot20_softrule"
P0_OUT="$EXP_DIR/phase_0_evidence/evidence_per_cid.jsonl"
P1_OUT="$EXP_DIR/phase_1_qa/qa_pairs.jsonl"
P2_OUT="$EXP_DIR/phase_2_independent/qa_independent.jsonl"
P3_OUT="$EXP_DIR/phase_3_validate/validated.jsonl"
FINAL_JSONL="$EXP_DIR/dataset_final.jsonl"
FINAL_JSON="$EXP_DIR/dataset_final.json"
SUMMARY="$EXP_DIR/dataset_summary.json"
LOG="$EXP_DIR/pilot20.log"

mkdir -p "$EXP_DIR/phase_0_evidence" "$EXP_DIR/phase_1_qa" \
         "$EXP_DIR/phase_2_independent" "$EXP_DIR/phase_3_validate"

echo "20-compound soft-rule pilot started at $(date)" | tee "$LOG"
echo "  Seed: 23 | CIDs: $(wc -l < "$CID_FILE")" | tee -a "$LOG"
echo "  Cap:  500 sentences, random sampled across articles" | tee -a "$LOG"
echo "  Design: soft rule (structural from SMILES, functional from evidence silently)" | tee -a "$LOG"

# ------------------------------------------------------------------
# Phase 0 — random-sampled evidence at cap=500
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "PHASE 0 — random evidence extraction (cap=500)" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-extract-evidence \
    --input data/filtered/drug_articles_v2_premium.jsonl \
    --output "$P0_OUT" \
    --target-cids "$CID_FILE" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Evidence counts per compound:" | tee -a "$LOG"
python3 -c "
import json
with open('$P0_OUT') as f:
    for line in f:
        rec = json.loads(line)
        print(f'  CID {rec[\"cid\"]:>12}: {len(rec.get(\"evidence_sentences\", [])):>4} sentences, {rec.get(\"num_pmids\", 0):>4} articles')
" | tee -a "$LOG"

# ------------------------------------------------------------------
# Phase 1 — Gemini 3 Flash + soft-rule prompt
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
    --workers 20 2>&1 | tee -a "$LOG"

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
    --workers 20 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------------
# Phase 3 — Gemma 4 31B
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "PHASE 3 — Gemma 4 31B" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
chem2textqa qa-judge \
    --input "$P2_OUT" \
    --output "$P3_OUT" \
    --model google/gemma-4-31b-it \
    --workers 20 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------------
# Assembly
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

# ------------------------------------------------------------------
# Quick visual inspection — first compound's Q&A
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "SAMPLE OUTPUT (first compound's Q&A)" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
python3 -c "
import json
with open('$FINAL_JSONL') as f:
    rec = json.loads(next(f))
print(f'CID {rec[\"cid\"]} | SMILES: {rec.get(\"smiles\",\"\")}')
print(f'Evidence sentences: {rec.get(\"num_evidence_sentences\",0)}')
print(f'Q&A pairs: {len(rec.get(\"qa_pairs\", []))}')
print()
for qa in rec.get('qa_pairs', []):
    print(f'  [{qa.get(\"topic\",\"?\")}] ({qa.get(\"verdict\",\"?\")}) {qa[\"question\"]}')
    print(f'    P1: {qa[\"phase1_answer\"][:200]}')
    print(f'    P2: {qa[\"phase2_answer\"][:200]}')
    print()
" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Pilot finished at $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Outputs:" | tee -a "$LOG"
echo "  Final:   $FINAL_JSONL" | tee -a "$LOG"
echo "  Summary: $SUMMARY" | tee -a "$LOG"
echo "  Log:     $LOG" | tee -a "$LOG"
