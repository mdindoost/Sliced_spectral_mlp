#!/usr/bin/env bash
# Bias ablation experiment for SlicedSpectralMLP with per-slice row norm.
# 4 variants × 4 datasets = 16 runs.
#
# Usage:
#   bash experiments/run_rownorm_nobias.sh 2>&1 | tee outputs/rownorm_nobias/run.log

set -euo pipefail

DATASETS="cora citeseer pubmed cornell"
PY="./venv/bin/python"
CFG="experiments/configs/rownorm_nobias"

VARIANTS=(
    "sliced_orig"
    "sliced_rownorm_bias"
    "sliced_rownorm_nobias_head"
    "sliced_rownorm_addhiddenbias"
)

for ds in $DATASETS; do
    echo ""
    echo "========================================================"
    echo "Dataset: $ds"
    echo "========================================================"
    for variant in "${VARIANTS[@]}"; do
        echo ""
        echo "--- $ds / $variant ---"
        $PY experiments/run_experiment.py \
            --config $CFG/${ds}_${variant}.yaml
    done
done

echo ""
echo "All 16 runs complete."
