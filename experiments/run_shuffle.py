"""
Shuffle diagnostic runner.

Usage
-----
  python experiments/run_shuffle.py --dataset cora
  python experiments/run_shuffle.py --dataset cornell --epochs 100

Trains SlicedSpectralMLP with the original eigenvector ordering and with
a random column permutation. Reports the coarse-slice accuracy drop, which
measures how much the model relies on spectral ordering.

Interpretation:
  Large drop (< -10pp) → model uses spectral ordering → Sliced architecture
                          should help on this dataset.
  Small drop (> -5pp)  → eigenvectors are noise-like for this task → Sliced
                          architecture provides no benefit.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data.loaders import load_dataset
from src.evaluation.shuffle import run_shuffle_diagnostic


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run shuffle diagnostic")
    p.add_argument("--dataset",  type=str, default="cora",
                   choices=["cora", "citeseer", "pubmed", "cornell", "actor", "squirrel"])
    p.add_argument("--k",        type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--epochs",   type=int, default=200)
    p.add_argument("--lr",       type=float, default=0.01)
    p.add_argument("--wd",       type=float, default=5e-4)
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print(f"Loading '{args.dataset}' (k={args.k})…")
    U, labels, train_mask, val_mask, test_mask, eigenvalues = load_dataset(
        args.dataset, k=args.k
    )
    n_classes = int(labels.max().item()) + 1
    print(
        f"  N={U.shape[0]}  classes={n_classes}  "
        f"train={train_mask.sum()}  val={val_mask.sum()}  test={test_mask.sum()}"
    )

    print(f"\nRunning shuffle diagnostic ({args.epochs} epochs)…")
    result = run_shuffle_diagnostic(
        U=U, labels=labels,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        k=args.k, n_classes=n_classes,
        n_layers=args.n_layers, seed=args.seed,
        epochs=args.epochs, lr=args.lr, wd=args.wd,
    )

    print("\n" + "=" * 55)
    print(f"SHUFFLE DIAGNOSTIC — {args.dataset.upper()}")
    print("=" * 55)
    print(f"  Unshuffled: coarse={result['unshuffled_coarse']:.4f}  "
          f"full={result['unshuffled_full']:.4f}  val={result['unshuffled_val']:.4f}")
    print(f"  Shuffled:   coarse={result['shuffled_coarse']:.4f}  "
          f"full={result['shuffled_full']:.4f}")
    print(f"\n  COARSE DROP: {result['coarse_drop_pp']:+.1f}pp  "
          f"(reference: Cora=-17.9, Cornell=-8.1, Actor=-0.1)")
    print(f"  FULL DROP:   {result['full_drop_pp']:+.1f}pp")

    if result["coarse_drop_pp"] < -10:
        decision = "PROCEED — large spectral signal detected."
    elif result["coarse_drop_pp"] < -5:
        decision = "MARGINAL — moderate spectral signal."
    else:
        decision = "STOP — no clear spectral signal; SlicedMLP unlikely to help."

    print(f"\n  Decision: {decision}")
    print("=" * 55)

    # Save to outputs/
    out_dir = os.path.join("outputs", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    lines = [
        f"SHUFFLE DIAGNOSTIC — {args.dataset.upper()}",
        "=" * 55,
        f"N={U.shape[0]}  k={args.k}  epochs={args.epochs}  seed={args.seed}",
        f"Unshuffled: coarse={result['unshuffled_coarse']:.4f}  "
        f"full={result['unshuffled_full']:.4f}",
        f"Shuffled:   coarse={result['shuffled_coarse']:.4f}  "
        f"full={result['shuffled_full']:.4f}",
        f"Coarse drop: {result['coarse_drop_pp']:+.1f}pp",
        f"Full drop:   {result['full_drop_pp']:+.1f}pp",
        f"Decision: {decision}",
    ]
    path = os.path.join(out_dir, "diagnostic.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved {path}")
