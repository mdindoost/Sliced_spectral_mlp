"""
Evaluate SlicedSpectralMLP (3 weight strategies) against StandardMLP baselines.

Usage
-----
  python experiments/run_baselines.py --dataset cora
  python experiments/run_baselines.py --dataset cornell

Output (printed + saved to outputs/<dataset>/eval/)
---------------------------------------------------
  - Per-slice test accuracy table for all methods
  - outputs/<dataset>/eval/spectral_resolution_curve.png
  - outputs/<dataset>/eval/training_curves_coarse_vs_full.png
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.loaders import load_dataset
from src.models.sliced_mlp import SlicedSpectralMLP
from src.models.baselines import StandardMLP, train_baseline
from src.evaluation.metrics import accuracy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _train_sliced(
    model, U, labels, train_mask, val_mask, epochs, lr, wd
):
    """Train model and return (best_val_acc, val_accs_coarse, val_accs_full)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val_acc = 0.0
    best_state = None
    val_accs_coarse, val_accs_full = [], []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        all_logits = model(U)
        model.compute_loss(all_logits, labels, train_mask).backward()
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
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    out_dir = os.path.join("outputs", args.dataset, "eval")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading '{args.dataset}' (k={args.k})…")
    U, labels, train_mask, val_mask, test_mask, eigenvalues = load_dataset(
        args.dataset, k=args.k
    )
    n_classes = int(labels.max().item()) + 1
    k = args.k

    results = {}

    # ----------------------------------------------------------------
    # SlicedSpectralMLP — three weight strategies
    # ----------------------------------------------------------------
    STRATEGIES = ["uniform", "coarse", "eigenvalue"]
    best_strategy_acc = 0.0
    best_strategy_name = "uniform"

    for strategy in STRATEGIES:
        label = f"Sliced({strategy})"
        print(f"\n[{label}] training for {args.epochs} epochs…")

        model = SlicedSpectralMLP(
            k=k, n_classes=n_classes, n_layers=args.n_layers,
            loss_weights=strategy,
            eigenvalues=eigenvalues if strategy == "eigenvalue" else None,
        )

        best_val, val_coarse, val_full = _train_sliced(
            model, U, labels, train_mask, val_mask,
            epochs=args.epochs, lr=args.lr, wd=args.wd,
        )

        model.eval()
        with torch.no_grad():
            all_logits = model(U)
        slice_test_accs = [accuracy(logits, labels, test_mask) for logits in all_logits]

        results[label] = {
            "slice_test_accs": slice_test_accs,
            "val_coarse": val_coarse,
            "val_full":   val_full,
            "best_val":   best_val,
        }

        if best_val > best_strategy_acc:
            best_strategy_acc  = best_val
            best_strategy_name = label

        print(f"  best val acc = {best_val:.4f}")

    # ----------------------------------------------------------------
    # Baselines
    # ----------------------------------------------------------------
    BASELINES = [("StandardMLP-full", k), ("StandardMLP-half", k // 2)]
    print("\n[Baselines]")
    for bl_name, n_feat in BASELINES:
        X = U[:, :n_feat]
        bl_model = StandardMLP(n_features=n_feat, n_classes=n_classes,
                                hidden_dim=k, n_layers=args.n_layers)
        best_val, test_acc = train_baseline(
            bl_model, X, labels, train_mask, val_mask, test_mask,
            lr=args.lr, wd=args.wd, epochs=args.epochs,
        )
        results[bl_name] = {"slice_test_accs": [test_acc], "best_val": best_val}
        print(f"  {bl_name:<22} val={best_val:.4f}  test={test_acc:.4f}")

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    divider = "=" * 62
    print(f"\n{divider}")
    print(f"COMPARISON TABLE — {args.dataset.upper()}  (k={k})")
    print(divider)
    print(f"{'Method':<25} {'Best Val':>9} {'Test (full)':>12} {'Test (coarse)':>14}")
    print("-" * 62)
    for name, r in results.items():
        full_test   = r["slice_test_accs"][-1]
        coarse_test = r["slice_test_accs"][0]
        print(f"{name:<25} {r['best_val']:>9.4f} {full_test:>12.4f} {coarse_test:>14.4f}")
    print(divider)

    # ----------------------------------------------------------------
    # Plot 1: spectral resolution curve
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10.colors
    slice_dims = [k // 2 + j for j in range(model.n_slices)]

    for i, strategy in enumerate(STRATEGIES):
        label = f"Sliced({strategy})"
        accs  = results[label]["slice_test_accs"]
        ax.plot(slice_dims, accs, marker="o", ms=3, color=colors[i], label=label)

    ax.axhline(results["StandardMLP-full"]["slice_test_accs"][0],
               color="black", ls="--", lw=1.5, label=f"StandardMLP-full (d={k})")
    ax.axhline(results["StandardMLP-half"]["slice_test_accs"][0],
               color="gray",  ls=":",  lw=1.5, label=f"StandardMLP-half (d={k//2})")
    ax.set_xlabel("Slice dimension  d_j  (# eigenvectors)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Spectral resolution curve — {args.dataset}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "spectral_resolution_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {path}")

    # ----------------------------------------------------------------
    # Plot 2: coarse vs full training curves (best strategy)
    # ----------------------------------------------------------------
    r = results[best_strategy_name]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(r["val_coarse"], lw=1.5, alpha=0.85, label=f"Coarse slice  j=0, d={k//2}")
    ax.plot(r["val_full"],   lw=1.5, alpha=0.85, label=f"Full slice   j={k//2}, d={k}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation accuracy")
    ax.set_title(f"Coarse vs Full — {args.dataset}, {best_strategy_name}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves_coarse_vs_full.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SlicedSpectralMLP")
    p.add_argument("--dataset",  type=str, default="cora",
                   choices=["cora", "citeseer", "cornell"])
    p.add_argument("--k",        type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--lr",       type=float, default=0.01)
    p.add_argument("--wd",       type=float, default=5e-4)
    p.add_argument("--epochs",   type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(_parse_args())
