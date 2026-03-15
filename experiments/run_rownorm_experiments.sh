#!/usr/bin/env bash
# Row normalisation experiment runner.
# Lists all commands in execution order. Commands are COMMENTED OUT —
# uncomment and run individually or pipe to bash.
#
# Usage:
#   Review this file first, then run one section at a time:
#     bash experiments/run_rownorm_experiments.sh 2>&1 | tee outputs/rownorm/run.log
#
# Or to run a single experiment:
#   python experiments/run_experiment.py --config experiments/configs/rownorm/cora_sliced_rownorm.yaml

set -euo pipefail

DATASETS="cora citeseer pubmed cornell"
CUTOFFS="0 3 7 11 15 20 31"
PY="./venv/bin/python"

# ---------------------------------------------------------------------------
# EXPERIMENT A — Shuffle diagnostics with row-norm input
# Run shuffle diagnostic for each dataset using row-norm U_tilde.
# Handled by scripts/run_rownorm.py (exp_a).
# ---------------------------------------------------------------------------
echo "# EXPERIMENT A — Shuffle diagnostics (row-norm input)"

for ds in $DATASETS; do
  echo "$PY scripts/run_rownorm.py --dataset $ds --epochs 200"
  # $PY scripts/run_rownorm.py --dataset $ds --epochs 200
done

# ---------------------------------------------------------------------------
# EXPERIMENT B — Per-slice curves: original vs row-norm input
# Produces per_slice_comparison.png for each dataset.
# ---------------------------------------------------------------------------
echo ""
echo "# EXPERIMENT B — Per-slice curves (handled by run_rownorm.py above)"
echo "# outputs/rownorm/{dataset}/per_slice_comparison.png"

# ---------------------------------------------------------------------------
# EXPERIMENT C — Full comparison table (5 model variants)
# ---------------------------------------------------------------------------
echo ""
echo "# EXPERIMENT C — Full comparison table"

for ds in $DATASETS; do
  echo ""
  echo "# $ds"

  echo "$PY experiments/run_experiment.py --config experiments/configs/rownorm/${ds}_sliced_orig.yaml"
  # $PY experiments/run_experiment.py --config experiments/configs/rownorm/${ds}_sliced_orig.yaml

  echo "$PY experiments/run_experiment.py --config experiments/configs/rownorm/${ds}_sliced_rownorm.yaml"
  # $PY experiments/run_experiment.py --config experiments/configs/rownorm/${ds}_sliced_rownorm.yaml

  echo "$PY experiments/run_experiment.py --config experiments/configs/rownorm/${ds}_sliced_rownorm_nosph.yaml"
  # $PY experiments/run_experiment.py --config experiments/configs/rownorm/${ds}_sliced_rownorm_nosph.yaml

  # RowNormMLP variants — use scripts/run_rownorm.py
  echo "$PY scripts/run_rownorm.py --dataset $ds --epochs 200"
  # $PY scripts/run_rownorm.py --dataset $ds --epochs 200
done

# ---------------------------------------------------------------------------
# EXPERIMENT D — Cutoff sensitivity scan
# Run truncated loss training for j*=0,3,7,11,15,20,31
# both with and without row-norm input.
# ---------------------------------------------------------------------------
echo ""
echo "# EXPERIMENT D — Cutoff sensitivity (orig and row-norm)"

for ds in $DATASETS; do
  echo ""
  echo "# $ds — original input"
  for j in $CUTOFFS; do
    echo "$PY experiments/run_experiment.py --config experiments/configs/rownorm/${ds}_sliced_orig.yaml --loss_cutoff $j"
    # $PY experiments/run_experiment.py --config experiments/configs/rownorm/${ds}_sliced_orig.yaml --loss_cutoff $j
  done

  echo "# $ds — row-norm input"
  for j in $CUTOFFS; do
    echo "$PY experiments/run_experiment.py --config experiments/configs/rownorm/${ds}_sliced_rownorm.yaml --loss_cutoff $j"
    # $PY experiments/run_experiment.py --config experiments/configs/rownorm/${ds}_sliced_rownorm.yaml --loss_cutoff $j
  done
done

echo ""
echo "# All experiments listed. Uncomment lines above and run to execute."
echo "# Or use: python scripts/run_rownorm.py [--dataset DATASET] for all-in-one."
