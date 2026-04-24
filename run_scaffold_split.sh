#!/usr/bin/env bash
# ============================================================
# Apply scaffold-based MoleculeNet-standard 70/15/15 split to the
# full-premium dataset, rewriting the `split` field on every compound
# record in place. Canary compounds stay tagged "canary".
#
# Requires: rdkit (installs it if missing)
#
# Outputs:
#   data/qa_pipeline/full_premium_kimi/cid_to_split.json
#   data/qa_pipeline/full_premium_kimi/scaffold_split_report.json
#   (dataset_final.jsonl, dataset_gold.jsonl, canary/dataset_final.jsonl
#    are updated in place with the new `split` field)
#
# Usage:
#   bash run_scaffold_split.sh
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="data/qa_pipeline/full_premium_kimi"
MAIN="$OUT_DIR/dataset_final.jsonl"
GOLD="$OUT_DIR/dataset_gold.jsonl"
CANARY="$OUT_DIR/canary/dataset_final.jsonl"
MAPPING="$OUT_DIR/cid_to_split.json"
REPORT="$OUT_DIR/scaffold_split_report.json"

for f in "$MAIN" "$GOLD" "$CANARY"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: $f missing. Run the full pipeline first."
        exit 1
    fi
done

# ------------------------------------------------------------------
# Ensure RDKit is available
# ------------------------------------------------------------------
if ! python3 -c "import rdkit" 2>/dev/null; then
    echo ">>> Installing RDKit (one-time)..."
    pip install rdkit
fi

# ------------------------------------------------------------------
# Step 1 — compute scaffold-based CID → split mapping
# ------------------------------------------------------------------
echo ""
echo "==========================================="
echo ">>> Computing MoleculeNet-standard scaffold split (70/15/15, strict)"
echo "==========================================="
python3 scripts/compute_scaffold_splits.py compute \
    --dataset "$MAIN" \
    --canary  "$CANARY" \
    --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 \
    --mode strict --seed 42 \
    --output "$MAPPING" \
    --report "$REPORT"

# ------------------------------------------------------------------
# Step 2 — apply the mapping to every dataset file in place
# ------------------------------------------------------------------
for target in "$MAIN" "$GOLD" "$CANARY"; do
    echo ""
    echo "==========================================="
    echo ">>> Applying split to $target"
    echo "==========================================="
    python3 scripts/compute_scaffold_splits.py apply \
        --src "$target" \
        --mapping "$MAPPING"
done

echo ""
echo "==========================================="
echo "Scaffold split complete."
echo "==========================================="
echo "Mapping: $MAPPING"
echo "Report:  $REPORT"
echo "Updated in place:"
echo "  $MAIN"
echo "  $GOLD"
echo "  $CANARY"
