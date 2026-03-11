"""
Main experiment runner for SlicedSpectralMLP.

Usage
-----
  python experiments/run_experiment.py --config experiments/configs/cora.yaml
  python experiments/run_experiment.py --config experiments/configs/cora.yaml --epochs 5

The script loads dataset-specific config from a YAML file, trains
SlicedSpectralMLP with all three weight strategies plus StandardMLP
baselines, and saves results to outputs/<dataset>/.

Config format: see experiments/configs/default.yaml for all fields.
Command-line flags override the YAML config values.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# Allow running from project root without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml is required. Install with: pip install pyyaml")
    sys.exit(1)

from src.data.loaders import load_dataset
from src.models.sliced_mlp import SlicedSpectralMLP
from src.models.baselines import StandardMLP, train_baseline
from src.evaluation.metrics import accuracy, per_slice_accuracy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    """Load YAML config, falling back to empty dict if file missing."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _train_sliced(
    model: SlicedSpectralMLP,
    U: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    epochs: int,
    lr: float,
    wd: float,
    loss_cutoff=None,
) -> tuple:
    """Train one SlicedSpectralMLP run. Returns (best_val, val_coarse, val_full)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val_acc = 0.0
    best_state = None
    val_accs_coarse, val_accs_full = [], []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        all_logits = model(U)
        model.compute_loss(all_logits, labels, train_mask,
                           loss_cutoff=loss_cutoff).backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            all_logits = model(U)
        coarse_val = accuracy(all_logits[0],  labels, val_mask)
        full_val   = accuracy(all_logits[-1], labels, val_mask)
        val_accs_coarse.append(coarse_val)
        val_accs_full.append(full_val)

        if full_val > best_val_acc:
            best_val_acc = full_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return best_val_acc, val_accs_coarse, val_accs_full


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: dict) -> None:
    dataset  = cfg["dataset"]
    k        = cfg.get("k", 64)
    n_layers = cfg.get("n_layers", 2)
    lr       = cfg.get("lr", 0.01)
    wd       = cfg.get("weight_decay", 5e-4)
    epochs   = cfg.get("epochs", 200)
    seed     = cfg.get("seed", 42)
    n_splits = cfg.get("n_splits", 1)
    strategy = cfg.get("loss_strategy", "uniform")
    cutoff   = cfg.get("loss_cutoff", None)

    out_dir = os.path.join("outputs", dataset)
    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Data
    # ----------------------------------------------------------------
    print(f"Loading dataset '{dataset}' (k={k})…")
    U, labels, train_mask, val_mask, test_mask, eigenvalues = load_dataset(
        dataset, k=k, split_idx=0
    )
    n_classes = int(labels.max().item()) + 1
    print(
        f"  {U.shape[0]} nodes | {n_classes} classes | "
        f"{train_mask.sum().item()} train | "
        f"{val_mask.sum().item()} val | "
        f"{test_mask.sum().item()} test"
    )

    # ----------------------------------------------------------------
    # SlicedSpectralMLP — configured strategy
    # ----------------------------------------------------------------
    torch.manual_seed(seed)
    model = SlicedSpectralMLP(
        k=k,
        n_classes=n_classes,
        n_layers=n_layers,
        loss_weights=strategy,
        eigenvalues=eigenvalues if strategy == "eigenvalue" else None,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\nSlicedSpectralMLP | k={k} | slices={model.n_slices} | "
        f"layers={n_layers} | loss={strategy} | params={n_params:,}"
    )
    print(f"Training for {epochs} epochs…")

    best_val, val_coarse, val_full = _train_sliced(
        model, U, labels, train_mask, val_mask,
        epochs=epochs, lr=lr, wd=wd, loss_cutoff=cutoff,
    )

    model.eval()
    with torch.no_grad():
        all_logits = model(U)
    slice_test_accs = per_slice_accuracy(all_logits, labels, test_mask)

    print(f"\nSliced({strategy}): best_val={best_val:.4f}")
    print(f"  test (coarse j=0):  {slice_test_accs[0]:.4f}")
    print(f"  test (full  j={k//2}): {slice_test_accs[-1]:.4f}")
    print(f"  test (best slice):  {max(slice_test_accs):.4f}")

    # ----------------------------------------------------------------
    # Baselines
    # ----------------------------------------------------------------
    print("\n[Baselines]")
    bl_results = {}
    for bl_name, n_feat in [("MLP-full", k), ("MLP-half", k // 2)]:
        torch.manual_seed(seed)
        X = U[:, :n_feat]
        bl = StandardMLP(n_features=n_feat, n_classes=n_classes,
                         hidden_dim=k, n_layers=n_layers)
        bv, ta = train_baseline(bl, X, labels, train_mask, val_mask, test_mask,
                                 lr=lr, wd=wd, epochs=epochs)
        bl_results[bl_name] = {"best_val": bv, "test": ta}
        print(f"  {bl_name:<12} val={bv:.4f}  test={ta:.4f}")

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    div = "=" * 65
    lines = [
        "",
        div,
        f"EXPERIMENT RESULTS — {dataset.upper()}  (k={k}, seed={seed}, {epochs} epochs)",
        div,
        f"{'Method':<25} {'Best Val':>9} {'Test(full)':>11} {'Test(coarse)':>13} {'Test(best)':>11}",
        "-" * 65,
        f"{'Sliced('+strategy+')':<25} {best_val:>9.4f} {slice_test_accs[-1]:>11.4f} "
        f"{slice_test_accs[0]:>13.4f} {max(slice_test_accs):>11.4f}",
    ]
    for bl_name, r in bl_results.items():
        lines.append(
            f"{bl_name:<25} {r['best_val']:>9.4f} {r['test']:>11.4f} "
            f"{'—':>13} {'—':>11}"
        )
    lines.append(div)

    table_str = "\n".join(lines)
    print(table_str)

    table_path = os.path.join(out_dir, "comparison_table.txt")
    with open(table_path, "w") as f:
        f.write(table_str + "\n")
    print(f"\nSaved {table_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SlicedSpectralMLP experiment")
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config file (e.g. experiments/configs/cora.yaml)")
    p.add_argument("--dataset",  type=str, default=None,
                   help="Override dataset name from config")
    p.add_argument("--epochs",   type=int, default=None,
                   help="Override number of training epochs")
    p.add_argument("--k",        type=int, default=None,
                   help="Override number of eigenvectors")
    p.add_argument("--lr",       type=float, default=None,
                   help="Override learning rate")
    p.add_argument("--seed",     type=int, default=None,
                   help="Override random seed")
    p.add_argument("--loss_strategy", type=str, default=None,
                   choices=["uniform", "coarse", "eigenvalue"],
                   help="Override loss weighting strategy")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Load YAML config
    cfg = _load_config(args.config)

    # Apply CLI overrides
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.k is not None:
        cfg["k"] = args.k
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.loss_strategy is not None:
        cfg["loss_strategy"] = args.loss_strategy

    run(cfg)
