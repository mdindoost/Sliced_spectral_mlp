"""
Training script for SlicedSpectralMLP.

Usage
-----
  python train.py                          # Cora, uniform weights
  python train.py --dataset cornell        # Cornell (heterophilous)
  python train.py --loss_weights coarse
  python train.py --loss_weights eigenvalue

Output
------
  outputs/checkpoints/best_model.pt   — best checkpoint (by full-slice val acc)
  outputs/grad_heatmaps/epoch_XXXX.png — |W[0].grad| heatmap every 10 epochs
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import load_dataset
from model import SlicedSpectralMLP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_grad_heatmap(grad: torch.Tensor, epoch: int, out_dir: str) -> None:
    """Save |W[0].grad| as a (k×k) heatmap PNG."""
    g = grad.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(g, cmap="hot", aspect="auto",
                   vmin=0, vmax=g.max() if g.max() > 0 else 1.0)
    ax.set_title(f"|W[0].grad|  epoch {epoch}")
    ax.set_xlabel("column j")
    ax.set_ylabel("row i")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(out_dir, f"epoch_{epoch:04d}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor,
               mask: torch.Tensor) -> float:
    return (logits.argmax(-1)[mask] == labels[mask]).float().mean().item()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    os.makedirs("outputs/grad_heatmaps", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)

    # ----------------------------------------------------------------
    # Data
    # ----------------------------------------------------------------
    print(f"Loading dataset '{args.dataset}' (k={args.k})…")
    U, labels, train_mask, val_mask, test_mask, eigenvalues = load_dataset(
        args.dataset, k=args.k
    )
    n_classes = int(labels.max().item()) + 1

    if args.shuffle_eigenvectors:
        perm = torch.randperm(U.shape[1])
        U = U[:, perm]
        eigenvalues = eigenvalues[perm.numpy()]
        print(f"  Eigenvectors shuffled (random column permutation)")

    print(
        f"  {U.shape[0]} nodes | {n_classes} classes | "
        f"{train_mask.sum().item()} train | "
        f"{val_mask.sum().item()} val | "
        f"{test_mask.sum().item()} test"
    )

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    model = SlicedSpectralMLP(
        k=args.k,
        n_classes=n_classes,
        n_layers=args.n_layers,
        loss_weights=args.loss_weights,
        eigenvalues=eigenvalues if args.loss_weights == "eigenvalue" else None,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    k = args.k
    print(
        f"  SlicedSpectralMLP | k={k} | slices={model.n_slices} | "
        f"layers={args.n_layers} | loss={args.loss_weights} | params={n_params:,}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )

    best_val_acc = 0.0
    best_epoch = 0

    # Column header — sample 5 slices for display (coarse, 3 mid, full)
    sample_js = sorted(
        {0, k // 8, k // 4, 3 * k // 8, k // 2}
    )
    slice_header = " | ".join(f"j{j:02d}(d{k//2+j:3d})" for j in sample_js)
    print(f"\n{'Epoch':>5} | {'loss':>8} | {slice_header} | {'best':>6}")
    print("-" * (38 + 15 * len(sample_js)))

    # ----------------------------------------------------------------
    # Epoch loop
    # ----------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):

        # --- Train step ---
        model.train()
        optimizer.zero_grad()
        all_logits = model(U)
        loss = model.compute_loss(all_logits, labels, train_mask,
                                   loss_cutoff=args.loss_cutoff)
        loss.backward()

        # Save gradient heatmap BEFORE optimizer.step() (while grad is fresh)
        if epoch % 10 == 0 and model.W[0].grad is not None:
            _save_grad_heatmap(
                torch.abs(model.W[0].grad),
                epoch,
                "outputs/grad_heatmaps",
            )

        optimizer.step()

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            all_logits = model(U)

        # Per-slice val accuracy
        slice_val_accs = [
            _accuracy(logits, labels, val_mask)
            for logits in all_logits
        ]
        full_val_acc = slice_val_accs[-1]

        if full_val_acc > best_val_acc:
            best_val_acc = full_val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "outputs/checkpoints/best_model.pt")

        # --- Logging (sampled slices) ---
        sampled = " | ".join(f"{slice_val_accs[j]:.4f}       " for j in sample_js)
        print(
            f"{epoch:5d} | {loss.item():8.4f} | {sampled} | {best_val_acc:.4f}",
            flush=True,
        )

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    print(f"\nDone. Best full-slice val acc = {best_val_acc:.4f} @ epoch {best_epoch}")
    print(
        "Checkpoints saved to  outputs/checkpoints/best_model.pt\n"
        "Gradient heatmaps in  outputs/grad_heatmaps/"
    )

    # Print all-slice val accuracy at best epoch using best model
    model.load_state_dict(torch.load("outputs/checkpoints/best_model.pt"))
    model.eval()
    with torch.no_grad():
        all_logits = model(U)
    print(f"\nPer-slice val accuracy at best epoch ({best_epoch}):")
    for j, logits in enumerate(all_logits):
        acc = _accuracy(logits, labels, val_mask)
        d_j = k // 2 + j
        bar = "█" * int(acc * 30)
        print(f"  j={j:3d}  d={d_j:3d}  {acc:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SlicedSpectralMLP")
    p.add_argument("--dataset",      type=str,   default="cora",
                   choices=["cora", "citeseer", "cornell"],
                   help="Dataset name (default: cora)")
    p.add_argument("--k",            type=int,   default=64,
                   help="Number of eigenvectors (default: 64)")
    p.add_argument("--n_layers",     type=int,   default=2,
                   help="Number of shared hidden layers (default: 2)")
    p.add_argument("--lr",           type=float, default=0.01,
                   help="Adam learning rate (default: 0.01)")
    p.add_argument("--wd",           type=float, default=5e-4,
                   help="Weight decay (default: 5e-4)")
    p.add_argument("--epochs",       type=int,   default=200,
                   help="Training epochs (default: 200)")
    p.add_argument("--loss_weights", type=str,   default="uniform",
                   choices=["uniform", "coarse", "eigenvalue"],
                   help="Slice loss weighting strategy (default: uniform)")
    p.add_argument("--shuffle_eigenvectors", action="store_true",
                   help="Randomly permute eigenvector columns before slicing")
    p.add_argument("--loss_cutoff", type=int, default=None,
                   help="Zero loss for slices j > loss_cutoff (default: all slices active)")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
