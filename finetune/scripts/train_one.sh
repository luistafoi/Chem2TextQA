#!/usr/bin/env bash
# Run a single LoRA SFT job with LlaSMol hyperparameters on all 8 GPUs.
# Usage: ./train_one.sh <hf_model_id> <run_tag> [per_device_bs] [grad_accum]
set -euo pipefail

REPO=/mnt/data_lab/ChemQA
export HF_HOME=$REPO/hf_cache
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
source $REPO/.venv/bin/activate

MODEL_NAME=${1:?model_id required}
RUN_TAG=${2:?run_tag required}
PDB=${3:-16}
GAS=${4:-4}
GRAD_CKPT=${5:-1}
DDP_FIND_UNUSED=${6:-0}
NGPU=${NGPU:-8}

OUT=$REPO/checkpoints/$RUN_TAG
LOG=$REPO/logs/${RUN_TAG}.log
mkdir -p $OUT

echo ">>> $RUN_TAG: model=$MODEL_NAME pdb=$PDB gas=$GAS global=$((PDB*GAS*NGPU))"
echo ">>> log -> $LOG"

torchrun --standalone --nproc_per_node=$NGPU \
  $REPO/scripts/finetune.py \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUT" \
  --run_name "$RUN_TAG" \
  --per_device_batch_size $PDB \
  --grad_accum_steps $GAS \
  --max_seq_len 512 \
  --num_epochs 3 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --lora_r 16 --lora_alpha 16 --lora_dropout 0.05 \
  --grad_ckpt $GRAD_CKPT \
  --ddp_find_unused $DDP_FIND_UNUSED \
  2>&1 | tee "$LOG"
