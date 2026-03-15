"""
Multi-seed experiment — fixed splits, outputs to outputs/multiseed/.

This is a thin wrapper around run_fixed_split_multiseed.py that redirects
all output to outputs/multiseed/ instead of outputs/fixed_split_multiseed/.

Split protocol (identical to run_fixed_split_multiseed.py):
  cora, citeseer, pubmed   — PyG Planetoid fixed splits
  amazon_photo, actor,     — fixed stratified 60/20/20 (seed=100),
  cornell                    saved to outputs/fixed_splits/ and reused

Seeds [0,1,2,3,4] affect weight initialisation only.

Usage
-----
    python scripts/run_multiseed.py
    python scripts/run_multiseed.py --dataset cora citeseer
    python scripts/run_multiseed.py --seeds 0 1 --epochs 200 --skip-pytest

For the primary evaluation output, use run_fixed_split_multiseed.py directly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.run_fixed_split_multiseed as _exp

# Redirect all output paths to outputs/multiseed/
_exp.OUT_ROOT        = "outputs/multiseed"
_exp.FIXED_SPLIT_DIR = "outputs/fixed_splits"   # shared — same saved masks
_exp.PER_SEED_DIR    = "outputs/multiseed/per_seed"
_exp.AGG_DIR         = "outputs/multiseed/aggregated"
_exp.TAB_DIR         = "outputs/multiseed/tables"
_exp.FIG_DIR         = "outputs/multiseed/figures"

if __name__ == "__main__":
    _exp.main()
