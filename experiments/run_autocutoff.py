"""
Automatic loss cutoff selection.

Usage
-----
  python experiments/run_autocutoff.py --dataset cora --strategy c
  python experiments/run_autocutoff.py --dataset cora --strategy a
  python experiments/run_autocutoff.py --dataset cora --strategy b --warmup 10

Strategies:
  a — eigenvalue gap (pre-training, no model training required)
  b — warmup-based (trains for W epochs with all slices, picks peak)
  c — median eigenvalue threshold (pre-training, the winner on Cora)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data.loaders import load_dataset
from src.models.sliced_mlp import SlicedSpectralMLP
from src.evaluation.metrics import accuracy
from src.cutoff import strategy_a, strategy_b, strategy_c


def _train_with_weights(
    U, labels, train_mask, val_mask, test_mask,
    k, n_classes, n_layers, weight_vec, epochs, lr, wd, seed
):
    torch.manual_seed(seed)
    model = SlicedSpectralMLP(k=k, n_classes=n_classes, n_layers=n_layers,
                               loss_weights="uniform")
    w = torch.tensor(weight_vec, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state = 0.0, None
    active_js = [j for j, wj in enumerate(weight_vec) if wj > 0]
    top_j = active_js[-1] if active_js else len(weight_vec) - 1

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        import torch.nn.functional as F
        lg = model(U)
        loss = torch.zeros(1)[0]
        for j, logits in enumerate(lg):
            if weight_vec[j] > 0:
                loss = loss + weight_vec[j] * F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            lg = model(U)
        v = accuracy(lg[top_j], labels, val_mask)
        if v > best_val:
            best_val = v
            best_state = {k2: v2.clone() for k2, v2 in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        lg = model(U)
    slice_test = [accuracy(l, labels, test_mask) for l in lg]
    return best_val, slice_test


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Automatic cutoff selection")
    p.add_argument("--dataset",  type=str, default="cora",
                   choices=["cora", "citeseer", "pubmed", "cornell", "actor", "squirrel"])
    p.add_argument("--strategy", type=str, default="c", choices=["a", "b", "c"],
                   help="Cutoff strategy: a=eig-gap, b=warmup, c=eig-threshold (default: c)")
    p.add_argument("--warmup",   type=int, default=10,
                   help="Warmup epochs for strategy b (default: 10)")
    p.add_argument("--k",        type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--epochs",   type=int, default=200)
    p.add_argument("--lr",       type=float, default=0.01)
    p.add_argument("--wd",       type=float, default=5e-4)
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    k = args.k

    print(f"Loading '{args.dataset}' (k={k})…")
    U, labels, train_mask, val_mask, test_mask, eigenvalues = load_dataset(
        args.dataset, k=k
    )
    n_classes = int(labels.max().item()) + 1
    print(f"  N={U.shape[0]}  classes={n_classes}")

    # ----------------------------------------------------------------
    # Select cutoff
    # ----------------------------------------------------------------
    if args.strategy == "a":
        cutoff_j = strategy_a.select_cutoff(eigenvalues, k)
        weight_vec = strategy_a.uniform_weights(cutoff_j, k)
        strategy_name = f"Strategy A (eig-gap, cutoff j={cutoff_j})"

    elif args.strategy == "b":
        print(f"\nRunning {args.warmup}-epoch warmup…")
        warmup_curves = strategy_b.run_warmup(
            U, labels, train_mask, val_mask,
            k=k, n_classes=n_classes, n_layers=args.n_layers,
            warmup_epochs=args.warmup, lr=args.lr, wd=args.wd, seed=args.seed,
        )
        cutoff_j = strategy_b.select_cutoff(warmup_curves, args.warmup)
        weight_vec = strategy_b.uniform_weights(cutoff_j, k)
        strategy_name = f"Strategy B (warmup W={args.warmup}, cutoff j={cutoff_j})"

    else:  # c
        weight_vec = strategy_c.compute_weights(eigenvalues, k)
        cutoff_j = strategy_c.select_cutoff(eigenvalues, k)
        threshold = strategy_c.get_threshold(eigenvalues, k)
        strategy_name = f"Strategy C (eig-thresh λ≤{threshold:.4f}, cutoff j={cutoff_j})"

    print(f"\n{strategy_name}")
    print(f"Active slices: j=0..{cutoff_j}  ({cutoff_j+1} slices)")

    # ----------------------------------------------------------------
    # Train with selected weights
    # ----------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs…")
    best_val, slice_test = _train_with_weights(
        U, labels, train_mask, val_mask, test_mask,
        k=k, n_classes=n_classes, n_layers=args.n_layers,
        weight_vec=weight_vec, epochs=args.epochs,
        lr=args.lr, wd=args.wd, seed=args.seed,
    )

    print(f"\nResults:")
    print(f"  best_val:       {best_val:.4f}")
    print(f"  test@cutoff j={cutoff_j}: {slice_test[cutoff_j]:.4f}")
    print(f"  test (coarse):  {slice_test[0]:.4f}")
    print(f"  test (best):    {max(slice_test):.4f}")

    out_dir = os.path.join("outputs", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    lines = [
        f"AUTOCUTOFF RESULTS — {args.dataset.upper()}",
        "=" * 60,
        f"{strategy_name}",
        f"k={k}  epochs={args.epochs}  seed={args.seed}",
        "",
        f"best_val:           {best_val:.4f}",
        f"test@cutoff j={cutoff_j}:   {slice_test[cutoff_j]:.4f}",
        f"test (coarse j=0):  {slice_test[0]:.4f}",
        f"test (best slice):  {max(slice_test):.4f}",
    ]
    path = os.path.join(out_dir, f"autocutoff_strategy_{args.strategy}.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved {path}")
