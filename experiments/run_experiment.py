"""
Main experiment runner for SlicedSpectralMLP.

Usage
-----
  python experiments/run_experiment.py --config experiments/configs/cora.yaml
  python experiments/run_experiment.py --config experiments/configs/cora.yaml --epochs 5
  python experiments/run_experiment.py --config experiments/configs/cora.yaml --dry-run

The script loads dataset-specific config from a YAML file, trains
SlicedSpectralMLP with all three weight strategies plus StandardMLP
baselines, and saves results to outputs/<dataset>/.

Config format: see experiments/configs/default.yaml for all fields.
Command-line flags override the YAML config values.

--dry-run prints the fully resolved config (defaults merged, CLI overrides
applied) and exits without training. Use this to verify before every full run.
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
from src.utils.io import validate_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Canonical defaults — single source of truth for all keys.
_DEFAULTS = {
    "k": 64,
    "n_layers": 2,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "epochs": 200,
    "n_classes": None,
    "loss_strategy": "uniform",
    "loss_cutoff": None,
    "seed": 42,
    "n_splits": 1,
    "use_row_norm_input": False,
    "use_sphere_norm": True,
    "hidden_bias": False,
    "head_bias": True,
}


def _load_config(path: str) -> dict:
    """Load YAML config, falling back to empty dict if file missing."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _resolve_config(yaml_path: str, cli_overrides: dict) -> dict:
    """
    Build the fully resolved config:
      1. Start with _DEFAULTS.
      2. Overlay with values from the YAML file.
      3. Overlay with any non-None CLI overrides.
    """
    cfg = dict(_DEFAULTS)
    cfg.update(_load_config(yaml_path))
    for key, val in cli_overrides.items():
        if val is not None:
            cfg[key] = val
    return cfg


def _print_dry_run(cfg: dict) -> None:
    """Print the resolved config in a readable format and exit."""
    dataset = cfg.get("dataset", "<not set>")
    print(f"\nResolved config for {dataset}:")
    keys_ordered = [
        "dataset", "n_classes", "k", "n_layers", "lr", "weight_decay",
        "epochs", "seed", "loss_strategy", "loss_cutoff", "n_splits",
    ]
    # Print known keys in order, then any extras
    printed = set()
    for key in keys_ordered:
        if key in cfg:
            print(f"  {key:<15} {cfg[key]}")
            printed.add(key)
    for key, val in cfg.items():
        if key not in printed:
            print(f"  {key:<15} {val}")
    print()


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
    validate_config(cfg, check_data=True)

    dataset  = cfg["dataset"]
    k        = cfg["k"]
    n_layers = cfg["n_layers"]
    lr       = cfg["lr"]
    wd       = cfg["weight_decay"]
    epochs   = cfg["epochs"]
    seed     = cfg["seed"]
    n_splits      = cfg.get("n_splits", 1)
    strategy      = cfg.get("loss_strategy", "uniform")
    cutoff        = cfg.get("loss_cutoff", None)
    use_row_norm  = cfg.get("use_row_norm_input", False)
    use_sph_norm  = cfg.get("use_sphere_norm", True)
    hidden_bias   = cfg.get("hidden_bias", False)
    head_bias     = cfg.get("head_bias", True)

    out_root = cfg.get("output_dir", "outputs")
    out_dir  = os.path.join(out_root, dataset)
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

    if use_row_norm:
        print("  row_norm_input=True  (per-slice normalization inside model)")

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
        use_row_norm_input=use_row_norm,
        use_sphere_norm=use_sph_norm,
        hidden_bias=hidden_bias,
        head_bias=head_bias,
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

    run_name   = cfg.get("run_name", None)
    fname      = f"{run_name}_results.txt" if run_name else "comparison_table.txt"
    table_path = os.path.join(out_dir, fname)
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
    p.add_argument("--dry-run", action="store_true",
                   help="Print the fully resolved config and exit without training")
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
    p.add_argument("--row_norm_input", action="store_true", default=None,
                   help="Row-normalise U before feeding into model (Option B)")
    p.add_argument("--no_sphere_norm", action="store_true", default=None,
                   help="Disable internal sphere normalisation after each layer")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    cli_overrides = {
        "dataset":            args.dataset,
        "epochs":             args.epochs,
        "k":                  args.k,
        "lr":                 args.lr,
        "seed":               args.seed,
        "loss_strategy":      args.loss_strategy,
        "use_row_norm_input": True if args.row_norm_input else None,
        "use_sphere_norm":    False if args.no_sphere_norm else None,
    }
    cfg = _resolve_config(args.config, cli_overrides)

    if args.dry_run:
        _print_dry_run(cfg)
        validate_config(cfg, check_data=False)
        print("Config OK (structural checks passed; use without --dry-run to also check dataset).")
        sys.exit(0)

    run(cfg)
