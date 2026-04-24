#!/usr/bin/env bash
# Retry the 4 failed full evals (Qwen base/ft + Llama base/ft) with CUDA_HOME
# set so vLLM can invoke nvcc when building CUDA graphs. Gemma already done.
set -euo pipefail
REPO=/mnt/data_lab/ChemQA
export HF_HOME=$REPO/hf_cache
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/mnt/data_lab/ChemQA/cuda_home
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:${CPATH:-}
export LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
# Clear stale flashinfer build cache so it retries compile with new headers/libs
rm -rf /home/ubuntu/.cache/flashinfer 2>/dev/null || true
source $REPO/.venv/bin/activate

TEST=$REPO/data/processed/test.jsonl
N=${N:-1000000}
MAX_NEW=${MAX_NEW:-256}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}
GPU_UTIL=${GPU_UTIL:-0.95}
OUT_ROOT=$REPO/eval/results_full

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
wait
echo ">>> 4 retry evals done"

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
print('CIDEr computed with pycocoevalcap over the complete test split (n=26,205).\n')
print('## ChemQA holdout test set (full)\n')
print('| model | variant | n | CIDEr |')
print('|---|---|---|---|')
for model, variant, n, cider in sorted(rows):
    print(f'| {model} | {variant} | {n} | {cider:.4f} |')
" > $OUT_ROOT/summary.md
cat $OUT_ROOT/summary.md
