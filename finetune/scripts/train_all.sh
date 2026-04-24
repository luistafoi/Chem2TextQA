#!/usr/bin/env bash
# Sequentially fine-tune all three models on 8 GPUs via torchrun.
# LlaSMol hyperparameters; global batch = per_device * grad_accum * nGPU = 16 * 4 * 8 = 512.
set -euo pipefail

REPO=/mnt/data_lab/ChemQA
export HF_HOME=$REPO/hf_cache
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
source $REPO/.venv/bin/activate

NGPU=${NGPU:-8}

run_one () {
  local MODEL_NAME=$1
  local RUN_TAG=$2
  local PDB=${3:-16}
  local GAS=${4:-4}
  local OUT=$REPO/checkpoints/$RUN_TAG
  mkdir -p $OUT
  local LOG=$REPO/logs/${RUN_TAG}.log
  echo ">>> Training $MODEL_NAME as $RUN_TAG (pdb=$PDB gas=$GAS global=$((PDB*GAS*NGPU)))"
  echo ">>> Log: $LOG"
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
    2>&1 | tee $LOG
}

# Per-device batch sizes tuned for 8xB200 (183GB each) with bf16 + LoRA + grad-ckpt.
# Global batch is held at 512 in all cases to match the paper.
run_one Qwen/Qwen2.5-14B-Instruct       qwen2_5_14b_chemqa   16 4
run_one meta-llama/Llama-3.1-8B-Instruct llama3_1_8b_chemqa  16 4
run_one unsloth/gemma-3-12b-it          gemma3_12b_chemqa    16 4

echo "All three trainings complete."
