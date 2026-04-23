#!/usr/bin/env bash
# ============================================================
# Run an alternate Phase 1 model experiment.
#
# Reuses the existing Phase 0 evidence (shared across experiments —
# evidence extraction is model-independent). Writes Phase 1/2/3 and
# assembly outputs to data/qa_pipeline/experiments/<name>/.
#
# Usage:
#   bash run_qa_experiment.sh \
#     --name deepseek_chat \
#     --phase1-model deepseek/deepseek-chat
#
#   bash run_qa_experiment.sh \
#     --name gpt5_mini \
#     --phase1-model openai/gpt-5-mini \
#     --phase2-model moonshotai/kimi-k2.5
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

# Load .env if present
if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

NAME=""
PHASE1_MODEL=""
PHASE2_MODEL=""
PHASE3_MODEL=""
WORKERS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "ERROR: --name requires a value"; exit 1
            fi
            NAME="$2"; shift 2 ;;
        --phase1-model)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "ERROR: --phase1-model requires a value"; exit 1
            fi
            PHASE1_MODEL="$2"; shift 2 ;;
        --phase2-model)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "ERROR: --phase2-model requires a value"; exit 1
            fi
            PHASE2_MODEL="$2"; shift 2 ;;
        --phase3-model)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "ERROR: --phase3-model requires a value"; exit 1
            fi
            PHASE3_MODEL="$2"; shift 2 ;;
        --workers)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "ERROR: --workers requires a value"; exit 1
            fi
            WORKERS="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ -z "$NAME" ]]; then
    echo "ERROR: --name is required (e.g. --name deepseek_chat)"
    exit 1
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set (add to .env)"
    exit 1
fi

# Shared Phase 0 evidence
EVIDENCE="data/qa_pipeline/phase_0_evidence/evidence_per_cid.jsonl"
if [[ ! -f "$EVIDENCE" ]]; then
    echo "ERROR: Phase 0 evidence not found at $EVIDENCE"
    echo "Run run_qa_phase0.sh first."
    exit 1
fi

EXP_DIR="data/qa_pipeline/experiments/${NAME}"
P1_OUT="$EXP_DIR/phase_1_qa/qa_pairs.jsonl"
P1_ERR="$EXP_DIR/phase_1_qa/errors.jsonl"
P2_OUT="$EXP_DIR/phase_2_independent/qa_independent.jsonl"
P2_ERR="$EXP_DIR/phase_2_independent/errors.jsonl"
P3_OUT="$EXP_DIR/phase_3_validate/validated.jsonl"
LOG="$EXP_DIR/experiment.log"

mkdir -p "$EXP_DIR/phase_1_qa" "$EXP_DIR/phase_2_independent" "$EXP_DIR/phase_3_validate"

{
    echo "QA experiment: $NAME"
    echo "  Started:      $(date)"
    echo "  Evidence:     $EVIDENCE"
    echo "  Output dir:   $EXP_DIR"
    [[ -n "$PHASE1_MODEL" ]] && echo "  Phase 1 model: $PHASE1_MODEL"
    [[ -n "$PHASE2_MODEL" ]] && echo "  Phase 2 model: $PHASE2_MODEL"
    [[ -n "$PHASE3_MODEL" ]] && echo "  Phase 3 model: $PHASE3_MODEL"
    echo "==========================================="
} | tee "$LOG"

run_step() {
    local name="$1"
    shift
    echo "" | tee -a "$LOG"
    echo ">>> $name ($(date))" | tee -a "$LOG"
    if ! "$@" 2>&1 | tee -a "$LOG"; then
        echo "!!! $name FAILED — aborting" | tee -a "$LOG"
        exit 1
    fi
}

# Phase 1
P1_ARGS=(--input "$EVIDENCE" --output "$P1_OUT" --errors "$P1_ERR")
[[ -n "$PHASE1_MODEL" ]] && P1_ARGS+=(--model "$PHASE1_MODEL")
[[ -n "$WORKERS" ]] && P1_ARGS+=(--workers "$WORKERS")
run_step "Phase 1 ($NAME)" chem2textqa qa-generate "${P1_ARGS[@]}"

# Phase 2
P2_ARGS=(--input "$P1_OUT" --output "$P2_OUT" --errors "$P2_ERR")
[[ -n "$PHASE2_MODEL" ]] && P2_ARGS+=(--model "$PHASE2_MODEL")
[[ -n "$WORKERS" ]] && P2_ARGS+=(--workers "$WORKERS")
run_step "Phase 2 ($NAME)" chem2textqa qa-independent "${P2_ARGS[@]}"

# Phase 3
P3_ARGS=(--input "$P2_OUT" --output "$P3_OUT")
[[ -n "$PHASE3_MODEL" ]] && P3_ARGS+=(--model "$PHASE3_MODEL")
[[ -n "$WORKERS" ]] && P3_ARGS+=(--workers "$WORKERS")
run_step "Phase 3 ($NAME)" chem2textqa qa-judge "${P3_ARGS[@]}"

# Assembly
run_step "Assembly ($NAME)" chem2textqa qa-assemble \
    --phase0 "$EVIDENCE" \
    --phase1 "$P1_OUT" \
    --phase2 "$P2_OUT" \
    --phase3 "$P3_OUT" \
    --output-jsonl "$EXP_DIR/dataset_final.jsonl" \
    --output-json "$EXP_DIR/dataset_final.json" \
    --summary "$EXP_DIR/dataset_summary.json"

echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "Experiment $NAME finished at $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Outputs:" | tee -a "$LOG"
echo "  $EXP_DIR/dataset_final.jsonl" | tee -a "$LOG"
echo "  $EXP_DIR/dataset_final.json" | tee -a "$LOG"
echo "  $EXP_DIR/dataset_summary.json" | tee -a "$LOG"
