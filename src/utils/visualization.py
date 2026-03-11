"""
Visualization utilities.

Functions
---------
  save_grad_heatmap(grad, epoch, out_dir)     — save |W[0].grad| heatmap PNG
  plot_per_slice_curves(slice_dims, results, out_path) — spectral resolution plot
  plot_training_curves(val_coarse, val_full, title, out_path) — coarse vs full
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def save_grad_heatmap(grad: torch.Tensor, epoch: int, out_dir: str) -> None:
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


def plot_per_slice_curves(
    slice_dims: List[int],
    results: Dict[str, List[float]],
    baselines: Optional[Dict[str, float]] = None,
    title: str = "Spectral resolution curve",
    out_path: str = "per_slice_accuracy.png",
) -> None:
    """
    Plot per-slice test accuracy curves for multiple methods.

    Args:
        slice_dims:  List of slice dimension values (x-axis).
        results:     {method_name: [accuracy_per_slice]}.
        baselines:   {method_name: scalar_accuracy} for horizontal lines.
        title:       Plot title.
        out_path:    Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.tab10.colors

    for i, (name, accs) in enumerate(results.items()):
        ax.plot(slice_dims, accs, marker="o", ms=3,
                color=colors[i % 10], label=name)

    if baselines:
        for i, (name, val) in enumerate(baselines.items()):
            ax.axhline(val, color="black" if i == 0 else "gray",
                       ls="--" if i == 0 else ":", lw=1.5, label=name)

    ax.set_xlabel("Slice dimension  d_j  (# eigenvectors)", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(
    val_coarse: List[float],
    val_full: List[float],
    title: str = "Coarse vs Full training curves",
    out_path: str = "training_curves.png",
    k: int = 64,
) -> None:
    """
    Plot coarse vs full slice validation accuracy over epochs.

    Args:
        val_coarse: Per-epoch validation accuracy for coarse slice (j=0).
        val_full:   Per-epoch validation accuracy for full slice (j=k//2).
        title:      Plot title.
        out_path:   Output PNG path.
        k:          Number of eigenvectors.
    """
    half = k // 2
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(val_coarse, lw=1.5, alpha=0.85,
            label=f"Coarse slice  j=0, d={half}")
    ax.plot(val_full,   lw=1.5, alpha=0.85,
            label=f"Full slice   j={half}, d={k}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
