#!/usr/bin/env bash
# Full ChemQA holdout eval via vLLM. One eval per GPU, 6 GPUs in parallel,
# gpu_memory_utilization=0.95 to pack KV cache.
set -euo pipefail

REPO=/mnt/data_lab/ChemQA
export HF_HOME=$REPO/hf_cache
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
source $REPO/.venv/bin/activate

TEST=$REPO/data/processed/test.jsonl
N=${N:-1000000}
MAX_NEW=${MAX_NEW:-256}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}
GPU_UTIL=${GPU_UTIL:-0.95}
OUT_ROOT=$REPO/eval/results_full
mkdir -p $OUT_ROOT

run () {
  local gpu=$1 tag=$2 base=$3 adapter=$4
  local out=$OUT_ROOT/$tag.jsonl
  local log=$OUT_ROOT/$tag.log
  mkdir -p "$(dirname "$out")"
  local ADAPTER_ARG=""
  [ -n "$adapter" ] && ADAPTER_ARG="--adapter_dir $adapter"
  echo ">>> [gpu=$gpu] $tag"
  CUDA_VISIBLE_DEVICES=$gpu python3 $REPO/eval/scripts/generate_vllm.py \
    --base_model "$base" $ADAPTER_ARG \
    --input_file "$TEST" --output_file "$out" \
    --num_samples $N --max_new_tokens $MAX_NEW \
    --max_model_len $MAX_MODEL_LEN \
    --gpu_memory_utilization $GPU_UTIL \
    > "$log" 2>&1
}

run 0 qwen2_5_14b/base_chemqa   Qwen/Qwen2.5-14B-Instruct       "" &
run 1 qwen2_5_14b/ft_chemqa     Qwen/Qwen2.5-14B-Instruct       $REPO/checkpoints/qwen2_5_14b_chemqa/final &
run 2 llama3_1_8b/base_chemqa   meta-llama/Llama-3.1-8B-Instruct "" &
run 3 llama3_1_8b/ft_chemqa     meta-llama/Llama-3.1-8B-Instruct $REPO/checkpoints/llama3_1_8b_chemqa/final &
run 4 gemma3_12b/base_chemqa    unsloth/gemma-3-12b-it          "" &
run 5 gemma3_12b/ft_chemqa      unsloth/gemma-3-12b-it          $REPO/checkpoints/gemma3_12b_chemqa/final &
wait
echo ">>> all 6 vLLM evals done"

for p in $OUT_ROOT/*/*.jsonl; do
  [ -f "${p%.jsonl}.metrics.json" ] && continue
  python3 $REPO/eval/scripts/score.py --pred_file "$p"
done

RESULTS_DIR=$OUT_ROOT python3 -c "
import os, json
from pathlib import Path
ROOT = Path(os.environ['RESULTS_DIR'])
rows = []
for p in sorted(ROOT.rglob('*.metrics.json')):
    m = json.loads(p.read_text())
    tag = p.relative_to(ROOT).with_suffix('').as_posix().replace('.metrics','')
    model, cond = tag.split('/', 1)
    variant = 'base' if cond.startswith('base') else 'finetuned'
    rows.append((model, variant, m.get('n'), m.get('CIDEr')))
print('# ChemQA Full Holdout Evaluation Results\n')
print('CIDEr computed with pycocoevalcap over the complete test split.\n')
print('## ChemQA holdout test set (full)\n')
print('| model | variant | n | CIDEr |')
print('|---|---|---|---|')
for model, variant, n, cider in sorted(rows):
    print(f'| {model} | {variant} | {n} | {cider:.4f} |')
" > $OUT_ROOT/summary.md
echo ">>> summary -> $OUT_ROOT/summary.md"
cat $OUT_ROOT/summary.md
