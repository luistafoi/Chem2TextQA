#!/usr/bin/env bash
# ============================================================
# Soft-rule probes — evaluate SMILES vs context-sentence reliance
# under the new Phase 1/2 design.
#
# Runs three probes on a shared 30-compound sample (seed=29):
#
#   (1) Ablation        — evidence_A + SMILES_A  vs  random evidence + SMILES_A
#                         Expect structural to stay same, functional to change
#                         if hints actually drive functional claims.
#
#   (2) SMILES swap     — evidence_A + SMILES_B (hybrid) vs real baselines
#                         Expect structural to track SMILES donor B,
#                         functional to track evidence owner A.
#
#   (3) Empty evidence  — evidence_A + SMILES_A  vs  []    + SMILES_A
#                         Expect empty run to produce mostly structural Q&A;
#                         functional Q&A in empty run = hallucination.
#
# All three share a single 30-compound sample from the cached full-premium
# Phase 0 pool so results cross-reference cleanly.
#
# Cost: ~$2 total (4 × 30 Phase 1 calls on Gemini 3 Flash).
# Time: ~3 minutes.
#
# Usage:
#   tmux new -s probes
#   bash run_softrule_probes.sh
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

BASE="data/qa_pipeline/experiments/softrule_probes30"

# Shared sample files
SHARED_REAL_EV="$BASE/shared_real/evidence.jsonl"
SHARED_REAL_QA="$BASE/shared_real/qa_pairs.jsonl"

# (1) Ablation
ABL_SCR_EV="$BASE/ablation/scrambled_evidence.jsonl"
ABL_SCR_QA="$BASE/ablation/scrambled_qa_pairs.jsonl"
ABL_REPORT="$BASE/ablation/report.json"

# (2) SMILES swap
SWAP_HYBRID_EV="$BASE/swap/hybrid_evidence.jsonl"
SWAP_MAP="$BASE/swap/hybrid_map.json"
SWAP_HYBRID_QA="$BASE/swap/hybrid_qa_pairs.jsonl"
SWAP_REPORT="$BASE/swap/report.json"

# (3) Empty evidence
EMPTY_EV="$BASE/empty/empty_evidence.jsonl"
EMPTY_QA="$BASE/empty/empty_qa_pairs.jsonl"
EMPTY_REPORT="$BASE/empty/report.json"

LOG="$BASE/probes.log"
mkdir -p "$BASE/shared_real" "$BASE/ablation" "$BASE/swap" "$BASE/empty"

echo "Soft-rule probes started at $(date)" | tee "$LOG"

# Clean any prior Phase 1 outputs so we re-run fresh under the current prompts.
rm -f "$SHARED_REAL_QA" "$ABL_SCR_QA" "$SWAP_HYBRID_QA" "$EMPTY_QA"

# ------------------------------------------------------------------
# Step 1 — sample 30 compounds (real baseline) + scrambled + empty variants
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "STEP 1 — sample + prepare variants" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"

python3 scripts/ablation_probe.py prepare \
    --input "$SRC_EVIDENCE" \
    -n 30 --seed 29 \
    --output-real "$SHARED_REAL_EV" \
    --output-scrambled "$ABL_SCR_EV" 2>&1 | tee -a "$LOG"

python3 scripts/smiles_swap_probe.py prepare \
    --real-evidence "$SHARED_REAL_EV" \
    --seed 29 \
    --output "$SWAP_HYBRID_EV" \
    --mapping "$SWAP_MAP" 2>&1 | tee -a "$LOG"

# Empty probe: single neutral placeholder sentence that satisfies the
# Phase 1 evidence guard but carries no topic signal. This lets us observe
# the model's behaviour under deprivation rather than just hitting the
# pipeline's empty-evidence refusal.
python3 -c "
import json
from pathlib import Path
src = Path('$SHARED_REAL_EV')
dst = Path('$EMPTY_EV')
dst.parent.mkdir(parents=True, exist_ok=True)
placeholder = {
    'id': 1, 'pmid': 'PLACEHOLDER', 'source': 'placeholder',
    'text': '[COMPOUND] is a chemical entity studied in the literature.',
}
n = 0
with src.open() as f, dst.open('w') as out:
    for line in f:
        rec = json.loads(line)
        rec['evidence_sentences'] = [placeholder]
        rec['num_pmids'] = 0
        rec['pmids'] = []
        out.write(json.dumps(rec, ensure_ascii=False) + '\n')
        n += 1
print(f'  Wrote empty (with neutral placeholder): {dst} ({n} records)')
" 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------------
# Step 2 — Phase 1 on all four variants (real, scrambled, hybrid, empty)
# ------------------------------------------------------------------
run_p1() {
    local name="$1" in="$2" out="$3"
    echo "" | tee -a "$LOG"
    echo "-------------------------------------------" | tee -a "$LOG"
    echo "Phase 1 on $name" | tee -a "$LOG"
    echo "-------------------------------------------" | tee -a "$LOG"
    chem2textqa qa-generate \
        --input "$in" --output "$out" \
        --errors "${out%.jsonl}.errors.jsonl" \
        --model google/gemini-3-flash-preview \
        --workers 20 2>&1 | tee -a "$LOG"
}

echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "STEP 2 — Phase 1 on each variant" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
run_p1 "REAL (baseline)"    "$SHARED_REAL_EV"  "$SHARED_REAL_QA"
run_p1 "SCRAMBLED evidence" "$ABL_SCR_EV"      "$ABL_SCR_QA"
run_p1 "SWAPPED SMILES"     "$SWAP_HYBRID_EV"  "$SWAP_HYBRID_QA"
run_p1 "EMPTY evidence"     "$EMPTY_EV"        "$EMPTY_QA"

# ------------------------------------------------------------------
# Step 3 — compare each probe
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "STEP 3 — compare" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo ">>> Probe 1: ABLATION (real vs scrambled)" | tee -a "$LOG"
python3 scripts/ablation_probe.py compare \
    --real "$SHARED_REAL_QA" \
    --scrambled "$ABL_SCR_QA" \
    --output "$ABL_REPORT" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo ">>> Probe 2: SMILES SWAP (hybrid vs real baselines)" | tee -a "$LOG"
python3 scripts/smiles_swap_probe.py compare \
    --real "$SHARED_REAL_QA" \
    --hybrid "$SWAP_HYBRID_QA" \
    --mapping "$SWAP_MAP" \
    --output "$SWAP_REPORT" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo ">>> Probe 3: EMPTY EVIDENCE (real vs empty)" | tee -a "$LOG"
python3 scripts/empty_evidence_probe.py compare \
    --real "$SHARED_REAL_QA" \
    --empty "$EMPTY_QA" \
    --output "$EMPTY_REPORT" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Probes finished at $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Reports:" | tee -a "$LOG"
echo "  Ablation:     $ABL_REPORT" | tee -a "$LOG"
echo "  SMILES swap:  $SWAP_REPORT" | tee -a "$LOG"
echo "  Empty:        $EMPTY_REPORT" | tee -a "$LOG"
echo "  Log:          $LOG" | tee -a "$LOG"
