#!/usr/bin/env bash
# Run the remaining SMolInstruct molecule_captioning evals (Llama & Gemma,
# base + finetuned) in parallel across 4 GPUs.
set -euo pipefail
REPO=/mnt/data_lab/ChemQA
export HF_HOME=$REPO/hf_cache
export TOKENIZERS_PARALLELISM=false
source $REPO/.venv/bin/activate

SMOL=$REPO/eval/data/smolinstruct_mcap_test.jsonl
N=${N:-200}
MAX_NEW=${MAX_NEW:-256}
BS=${BS:-8}

run () {
  local gpu=$1 tag=$2 base=$3 adapter=$4
  local out=$REPO/eval/results/$tag.jsonl
  local log=$REPO/eval/results/$tag.log
  mkdir -p "$(dirname "$out")"
  local ADAPTER_ARG=""
  [ -n "$adapter" ] && ADAPTER_ARG="--adapter_dir $adapter"
  echo ">>> [gpu=$gpu] $tag"
  CUDA_VISIBLE_DEVICES=$gpu python3 $REPO/eval/scripts/generate.py \
    --base_model "$base" $ADAPTER_ARG \
    --input_file "$SMOL" --output_file "$out" \
    --num_samples $N --max_new_tokens $MAX_NEW --batch_size $BS --gpu 0 > "$log" 2>&1
}

run 0 llama3_1_8b/base_smol_mcap  meta-llama/Llama-3.1-8B-Instruct "" &
run 1 llama3_1_8b/ft_smol_mcap    meta-llama/Llama-3.1-8B-Instruct $REPO/checkpoints/llama3_1_8b_chemqa/final &
run 2 gemma3_12b/base_smol_mcap   unsloth/gemma-3-12b-it "" &
run 3 gemma3_12b/ft_smol_mcap     unsloth/gemma-3-12b-it $REPO/checkpoints/gemma3_12b_chemqa/final &
wait
echo ">>> extras done"

# Score all
for p in $REPO/eval/results/*/*.jsonl; do
  [ -f "${p%.jsonl}.metrics.json" ] && continue
  python3 $REPO/eval/scripts/score.py --pred_file "$p"
done

python3 $REPO/eval/scripts/summarize.py > $REPO/eval/results/summary.md
echo ">>> summary -> $REPO/eval/results/summary.md"
cat $REPO/eval/results/summary.md
