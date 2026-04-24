#!/usr/bin/env bash
# ============================================================
# SMOKE TEST — verifies the full premium pipeline end-to-end on a
# tiny slice of the v3 evidence, to catch plumbing failures before
# the $750 commitment.
#
# What it proves:
#   1. Phase 1 with the soft-rule prompt populates `evidence_ids`
#   2. Phase 2 and Phase 3 flow through without schema errors
#   3. Assembly emits `split` and `evidence_ids` fields in
#      dataset_final.jsonl
#   4. The main + canary dual-pass structure in the master script
#      actually runs both branches successfully
#
# Cost: ~$2. Time: ~3 minutes at 20 workers.
# Output: data/qa_pipeline/experiments/smoke15/
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

EVIDENCE_DIR="data/qa_pipeline/phase0_full_premium_v3"
MAIN_EVIDENCE="$EVIDENCE_DIR/evidence_main.jsonl"
CANARY_EVIDENCE="$EVIDENCE_DIR/evidence_canary.jsonl"
if [[ ! -f "$MAIN_EVIDENCE" ]] || [[ ! -f "$CANARY_EVIDENCE" ]]; then
    echo "ERROR: expected $MAIN_EVIDENCE and $CANARY_EVIDENCE"
    exit 1
fi

OUT="data/qa_pipeline/experiments/smoke15"
MAIN_DIR="$OUT/main"
CANARY_DIR="$OUT/canary"
SMOKE_MAIN_EV="$OUT/smoke_main.jsonl"
SMOKE_CANARY_EV="$OUT/smoke_canary.jsonl"
LOG="$OUT/smoke.log"

mkdir -p "$MAIN_DIR/phase_1_qa" "$MAIN_DIR/phase_2_independent" "$MAIN_DIR/phase_3_validate"
mkdir -p "$CANARY_DIR/phase_1_qa" "$CANARY_DIR/phase_2_independent" "$CANARY_DIR/phase_3_validate"

echo "Smoke test started at $(date)" | tee "$LOG"

# ------------------------------------------------------------------
# Step 1 — slice tiny subsets (10 main + 5 canary) with a fixed seed
# ------------------------------------------------------------------
python3 -c "
import json, random
random.seed(37)

main = []
with open('$MAIN_EVIDENCE') as f:
    for line in f:
        main.append(line)
canary = []
with open('$CANARY_EVIDENCE') as f:
    for line in f:
        canary.append(line)

# Prefer compounds with >= 10 evidence sentences so evidence_ids has something to pick
def with_n_sents(line, n):
    return len((json.loads(line).get('evidence_sentences') or [])) >= n

main_good = [l for l in main if with_n_sents(l, 10)]
canary_good = [l for l in canary if with_n_sents(l, 5)]
main_sel = random.sample(main_good, min(10, len(main_good)))
canary_sel = random.sample(canary_good, min(5, len(canary_good)))

with open('$SMOKE_MAIN_EV', 'w') as f:
    f.writelines(main_sel)
with open('$SMOKE_CANARY_EV', 'w') as f:
    f.writelines(canary_sel)
print(f'  Main smoke subset:   {len(main_sel)}  → $SMOKE_MAIN_EV')
print(f'  Canary smoke subset: {len(canary_sel)} → $SMOKE_CANARY_EV')
" | tee -a "$LOG"

run_step() {
    local name="$1"; shift
    echo "" | tee -a "$LOG"
    echo "--- $name ---" | tee -a "$LOG"
    if ! "$@" 2>&1 | tee -a "$LOG"; then
        echo "!!! $name FAILED" | tee -a "$LOG"
        exit 1
    fi
}

# ------------------------------------------------------------------
# Step 2 — Main branch: P1 → P2 → P3 → assemble
# ------------------------------------------------------------------
MAIN_P1="$MAIN_DIR/phase_1_qa/qa_pairs.jsonl"
MAIN_P2="$MAIN_DIR/phase_2_independent/qa_independent.jsonl"
MAIN_P3="$MAIN_DIR/phase_3_validate/validated.jsonl"
MAIN_FINAL="$MAIN_DIR/dataset_final.jsonl"

run_step "Main Phase 1 (Gemini 3 Flash)" \
    chem2textqa qa-generate \
        --input "$SMOKE_MAIN_EV" \
        --output "$MAIN_P1" \
        --errors "$MAIN_DIR/phase_1_qa/errors.jsonl" \
        --model google/gemini-3-flash-preview \
        --workers 20

run_step "Main Phase 2 (Kimi K2.5)" \
    chem2textqa qa-independent \
        --input "$MAIN_P1" \
        --output "$MAIN_P2" \
        --errors "$MAIN_DIR/phase_2_independent/errors.jsonl" \
        --model moonshotai/kimi-k2.5 \
        --workers 20

run_step "Main Phase 3 (Gemma 4 31B)" \
    chem2textqa qa-judge \
        --input "$MAIN_P2" \
        --output "$MAIN_P3" \
        --model google/gemma-4-31b-it \
        --workers 20

run_step "Main Assembly" \
    chem2textqa qa-assemble \
        --phase0 "$SMOKE_MAIN_EV" \
        --phase1 "$MAIN_P1" \
        --phase2 "$MAIN_P2" \
        --phase3 "$MAIN_P3" \
        --output-jsonl "$MAIN_FINAL" \
        --output-json  "$MAIN_DIR/dataset_final.json" \
        --summary      "$MAIN_DIR/dataset_summary.json"

# ------------------------------------------------------------------
# Step 3 — Canary branch (same pipeline, separate outputs)
# ------------------------------------------------------------------
CAN_P1="$CANARY_DIR/phase_1_qa/qa_pairs.jsonl"
CAN_P2="$CANARY_DIR/phase_2_independent/qa_independent.jsonl"
CAN_P3="$CANARY_DIR/phase_3_validate/validated.jsonl"
CAN_FINAL="$CANARY_DIR/dataset_final.jsonl"

run_step "Canary Phase 1" \
    chem2textqa qa-generate \
        --input "$SMOKE_CANARY_EV" \
        --output "$CAN_P1" \
        --errors "$CANARY_DIR/phase_1_qa/errors.jsonl" \
        --model google/gemini-3-flash-preview \
        --workers 20

run_step "Canary Phase 2" \
    chem2textqa qa-independent \
        --input "$CAN_P1" \
        --output "$CAN_P2" \
        --errors "$CANARY_DIR/phase_2_independent/errors.jsonl" \
        --model moonshotai/kimi-k2.5 \
        --workers 20

run_step "Canary Phase 3" \
    chem2textqa qa-judge \
        --input "$CAN_P2" \
        --output "$CAN_P3" \
        --model google/gemma-4-31b-it \
        --workers 20

run_step "Canary Assembly" \
    chem2textqa qa-assemble \
        --phase0 "$SMOKE_CANARY_EV" \
        --phase1 "$CAN_P1" \
        --phase2 "$CAN_P2" \
        --phase3 "$CAN_P3" \
        --output-jsonl "$CAN_FINAL" \
        --summary      "$CANARY_DIR/dataset_summary.json"

# ------------------------------------------------------------------
# Step 4 — Verifications: schema + evidence_ids + split + canary
# ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "SMOKE-TEST VERIFICATIONS" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"

python3 -c "
import json, sys
from collections import Counter

def load(p):
    with open(p) as f:
        return [json.loads(line) for line in f]

checks = []
def check(name, ok, detail=''):
    status = 'PASS' if ok else 'FAIL'
    msg = f'  [{status}] {name}'
    if detail:
        msg += f'  — {detail}'
    print(msg)
    checks.append(ok)

main = load('$MAIN_FINAL')
canary = load('$CAN_FINAL')

check('Main branch produced records', len(main) > 0, f'{len(main)} compounds')
check('Canary branch produced records', len(canary) > 0, f'{len(canary)} compounds')

# Schema check: split + evidence_ids present
main_has_split = all('split' in r for r in main)
check('Every main compound has split tag', main_has_split)
canary_all_canary_tag = all(r.get('split') == 'canary' for r in canary)
check('Every canary compound has split=\"canary\"', canary_all_canary_tag,
      f'observed: {Counter(r.get(\"split\") for r in canary).most_common()}')

main_split_dist = Counter(r.get('split') for r in main)
check('Main split distribution excludes canary', 'canary' not in main_split_dist,
      f'observed: {main_split_dist.most_common()}')

# evidence_ids field: present in every QA?
total_qa = 0; qa_with_field = 0; qa_with_nonempty = 0
empty_but_structural = 0; empty_but_functional = 0
import sys
sys.path.insert(0, 'scripts')
from topic_bucket import bucket_topic
for r in main + canary:
    for qa in r.get('qa_pairs', []):
        total_qa += 1
        if 'evidence_ids' in qa:
            qa_with_field += 1
            if qa['evidence_ids']:
                qa_with_nonempty += 1
            else:
                if bucket_topic(qa.get('topic')) == 'structural':
                    empty_but_structural += 1
                elif bucket_topic(qa.get('topic')) == 'functional':
                    empty_but_functional += 1

check('evidence_ids field present on every Q&A', qa_with_field == total_qa,
      f'{qa_with_field}/{total_qa}')
check('>=50% of functional Q&A have non-empty evidence_ids',
      (qa_with_nonempty / total_qa > 0.2 if total_qa else False),
      f'non-empty rate: {100*qa_with_nonempty/total_qa:.1f}% of {total_qa} Q&A')
print(f'    empty evidence_ids on structural Q&A: {empty_but_structural}  (acceptable)')
print(f'    empty evidence_ids on functional Q&A: {empty_but_functional}  (concerning if >>0)')

# Validate evidence_ids reference real IDs
bad_ids = 0
for r in main + canary:
    valid_ids = {s.get('id') for s in (r.get('evidence_sentences') or [])}
    for qa in r.get('qa_pairs', []):
        for eid in (qa.get('evidence_ids') or []):
            if eid not in valid_ids:
                bad_ids += 1
check('evidence_ids reference real sentence IDs (no fabrication)', bad_ids == 0,
      f'{bad_ids} dangling IDs')

# Show a sample
print()
print('  Sample (1 compound):')
sample = main[0] if main else canary[0]
print(f'    CID {sample[\"cid\"]} | split={sample.get(\"split\")} | qa_pairs={len(sample.get(\"qa_pairs\", []))}')
for qa in sample.get('qa_pairs', [])[:2]:
    print(f'      [{qa.get(\"topic\")}] verdict={qa.get(\"verdict\")} evidence_ids={qa.get(\"evidence_ids\")}')
    print(f'        Q: {qa[\"question\"][:100]}')
    print(f'        P1 ans: {qa[\"phase1_answer\"][:160]}')

if all(checks):
    print()
    print('  ALL CHECKS PASS — safe to run the full premium pipeline.')
    sys.exit(0)
else:
    print()
    print('  SOME CHECKS FAILED — investigate before the full run.')
    sys.exit(1)
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Smoke test finished at $(date)" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
