#!/usr/bin/env bash
# Evaluate all 3 models (base and finetuned) on the ChemQA holdout test set
# and on the LlaSMol SMolInstruct molecule_captioning test subset.
# Runs up to 8 evals concurrently across GPUs, one eval per GPU.
set -euo pipefail

REPO=/mnt/data_lab/ChemQA
export HF_HOME=$REPO/hf_cache
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=0
source $REPO/.venv/bin/activate

CHEMQA_TEST=$REPO/data/processed/test.jsonl
SMOL_TEST=$REPO/eval/data/smolinstruct_mcap_test.jsonl
N=${N:-500}                          # ChemQA samples per eval
N_SMOL=${N_SMOL:-200}                # SMolInstruct samples per eval
MAX_NEW=${MAX_NEW:-256}
BS=${BS:-8}

[ -f "$SMOL_TEST" ] || python3 $REPO/eval/scripts/prep_smolinstruct.py

declare -a JOBS=()   # "GPU|tag|base_model|adapter|input|out|N"

# ChemQA: base + finetuned for each of 3 models (6 evals)
JOBS+=("0|qwen2_5_14b/base_chemqa|Qwen/Qwen2.5-14B-Instruct||$CHEMQA_TEST|$REPO/eval/results/qwen2_5_14b/base_chemqa.jsonl|$N")
JOBS+=("1|qwen2_5_14b/ft_chemqa|Qwen/Qwen2.5-14B-Instruct|$REPO/checkpoints/qwen2_5_14b_chemqa/final|$CHEMQA_TEST|$REPO/eval/results/qwen2_5_14b/ft_chemqa.jsonl|$N")
JOBS+=("2|llama3_1_8b/base_chemqa|meta-llama/Llama-3.1-8B-Instruct||$CHEMQA_TEST|$REPO/eval/results/llama3_1_8b/base_chemqa.jsonl|$N")
JOBS+=("3|llama3_1_8b/ft_chemqa|meta-llama/Llama-3.1-8B-Instruct|$REPO/checkpoints/llama3_1_8b_chemqa/final|$CHEMQA_TEST|$REPO/eval/results/llama3_1_8b/ft_chemqa.jsonl|$N")
JOBS+=("4|gemma3_12b/base_chemqa|unsloth/gemma-3-12b-it||$CHEMQA_TEST|$REPO/eval/results/gemma3_12b/base_chemqa.jsonl|$N")
JOBS+=("5|gemma3_12b/ft_chemqa|unsloth/gemma-3-12b-it|$REPO/checkpoints/gemma3_12b_chemqa/final|$CHEMQA_TEST|$REPO/eval/results/gemma3_12b/ft_chemqa.jsonl|$N")

# SMolInstruct molecule_captioning: base + finetuned (6 more evals)
JOBS+=("6|qwen2_5_14b/base_smol_mcap|Qwen/Qwen2.5-14B-Instruct||$SMOL_TEST|$REPO/eval/results/qwen2_5_14b/base_smol_mcap.jsonl|$N_SMOL")
JOBS+=("7|qwen2_5_14b/ft_smol_mcap|Qwen/Qwen2.5-14B-Instruct|$REPO/checkpoints/qwen2_5_14b_chemqa/final|$SMOL_TEST|$REPO/eval/results/qwen2_5_14b/ft_smol_mcap.jsonl|$N_SMOL")

launch () {
  local spec=$1
  IFS='|' read -r gpu tag base adapter inp out n <<< "$spec"
  mkdir -p "$(dirname "$out")"
  local log=$REPO/eval/results/$tag.log
  mkdir -p "$(dirname "$log")"
  echo ">>> [gpu=$gpu] $tag  n=$n  base=$base  adapter=${adapter:-<none>}"
  local ADAPTER_ARG=""
  [ -n "$adapter" ] && ADAPTER_ARG="--adapter_dir $adapter"
  CUDA_VISIBLE_DEVICES=$gpu python3 $REPO/eval/scripts/generate.py \
    --base_model "$base" \
    $ADAPTER_ARG \
    --input_file "$inp" \
    --output_file "$out" \
    --num_samples $n \
    --max_new_tokens $MAX_NEW \
    --batch_size $BS \
    --gpu 0 \
    > "$log" 2>&1
}

# Run first 8 in parallel (one per GPU), wait, then any remaining serially.
pids=()
for i in "${!JOBS[@]}"; do
  if [ $i -lt 8 ]; then
    launch "${JOBS[$i]}" &
    pids+=($!)
  fi
done
echo ">>> launched ${#pids[@]} parallel eval jobs"
for pid in "${pids[@]}"; do wait $pid || true; done
echo ">>> parallel batch done"

# Any remaining (>=8th index) run one at a time on GPU 0
for i in "${!JOBS[@]}"; do
  if [ $i -ge 8 ]; then
    launch "${JOBS[$i]}"
  fi
done

# Score everything
echo ">>> scoring"
for i in "${!JOBS[@]}"; do
  IFS='|' read -r gpu tag base adapter inp out n <<< "${JOBS[$i]}"
  [ -f "$out" ] || { echo "MISSING $out"; continue; }
  python3 $REPO/eval/scripts/score.py --pred_file "$out"
done

python3 $REPO/eval/scripts/summarize.py > $REPO/eval/results/summary.md
echo ">>> summary -> $REPO/eval/results/summary.md"
cat $REPO/eval/results/summary.md
