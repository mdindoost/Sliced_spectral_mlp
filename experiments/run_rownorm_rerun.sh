#!/usr/bin/env bash
# Row normalisation RERUN — fixed per-slice normalization.
#
# This script reruns all row norm experiments after the fix that moves
# F.normalize inside the slice loop (applied to x[:, :d_j], not full x).
#
# Old (wrong) results:  outputs/rownorm/          — DO NOT DELETE
# New (correct) results: outputs/rownorm_fixed/    — written here
#
# Review each section before running. Execute one section at a time:
#   bash experiments/run_rownorm_rerun.sh 2>&1 | tee outputs/rownorm_fixed/run.log
#
# Or run a single dataset:
#   python scripts/run_rownorm.py --dataset cora --out-dir outputs/rownorm_fixed
#
# Hyperparameters (same as all prior experiments):
#   k=64, n_layers=2, lr=0.01, wd=5e-4, epochs=200, seed=42

set -euo pipefail

DATASETS="cora citeseer pubmed cornell"
CUTOFFS="0 3 7 11 15 20 31"
PY="./venv/bin/python"
OUT="outputs/rownorm_fixed"

mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# EXPERIMENT B + C + A — all-in-one per dataset via run_rownorm.py
# Covers:
#   A) Shuffle diagnostic: original vs row-norm (per-slice)
#   B) Per-slice curve: Sliced-orig vs Sliced-rownorm (does j* change?)
#   C) Full 7-model comparison table (StandardMLP, RowNormMLP, Sliced variants)
# ---------------------------------------------------------------------------
echo "# EXPERIMENTS A + B + C — all datasets via run_rownorm.py"
echo ""

for ds in $DATASETS; do
    echo "# $ds"
    echo "$PY scripts/run_rownorm.py --dataset $ds --epochs 200 --out-dir $OUT"
    # $PY scripts/run_rownorm.py --dataset $ds --epochs 200 --out-dir $OUT
done

# ---------------------------------------------------------------------------
# EXPERIMENT D — Cutoff sensitivity scan
# Run truncated loss training for j*=0,3,7,11,15,20,31
# both with and without row-norm input (per-slice, fixed).
# ---------------------------------------------------------------------------
echo ""
echo "# EXPERIMENT D — Cutoff sensitivity scan"

for ds in $DATASETS; do
    echo ""
    echo "# $ds — original input (no row-norm)"
    for j in $CUTOFFS; do
        echo "$PY experiments/run_experiment.py \\"
        echo "    --config experiments/configs/rownorm/${ds}_sliced_orig.yaml \\"
        echo "    --loss_cutoff $j"
        # $PY experiments/run_experiment.py \
        #     --config experiments/configs/rownorm/${ds}_sliced_orig.yaml \
        #     --loss_cutoff $j
    done

    echo ""
    echo "# $ds — per-slice row-norm input (fixed)"
    for j in $CUTOFFS; do
        echo "$PY experiments/run_experiment.py \\"
        echo "    --config experiments/configs/rownorm/${ds}_sliced_rownorm.yaml \\"
        echo "    --loss_cutoff $j"
        # $PY experiments/run_experiment.py \
        #     --config experiments/configs/rownorm/${ds}_sliced_rownorm.yaml \
        #     --loss_cutoff $j
    done
done

echo ""
echo "# ----------------------------------------------------------------"
echo "# All commands listed above. Uncomment the actual $PY lines to run."
echo "# Results will be written to $OUT/{dataset}/"
echo "# Compare against outputs/rownorm/ (old wrong results) for diff."
echo "# ----------------------------------------------------------------"
