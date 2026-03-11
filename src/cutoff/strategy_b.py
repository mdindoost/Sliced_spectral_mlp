"""
Strategy B: Warmup-based method for automatic loss cutoff selection.

Trains the model for W warmup epochs with all slices active, then selects
the cutoff j as the slice with the highest validation accuracy at epoch W.

This is the most reliable automatic method because it directly measures
task-relevant discriminability per slice — unlike eigenvalue gap (which
measures spectral structure, not class alignment).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from src.models.sliced_mlp import SlicedSpectralMLP
from src.evaluation.metrics import accuracy


def run_warmup(
    U: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    k: int,
    n_classes: int,
    n_layers: int = 2,
    warmup_epochs: int = 10,
    lr: float = 0.01,
    wd: float = 5e-4,
    seed: int = 42,
) -> Dict[int, List[float]]:
    """
    Run a warmup phase and record per-slice validation accuracy at each epoch.

    Args:
        U:             (N, k) eigenvector matrix.
        labels:        (N,) class labels.
        train_mask:    (N,) boolean training mask.
        val_mask:      (N,) boolean validation mask.
        k, n_classes, n_layers, warmup_epochs, lr, wd, seed: hyperparameters.

    Returns:
        warmup_curves: dict {epoch: [val_acc_j for j in range(n_slices)]}
    """
    torch.manual_seed(seed)
    model = SlicedSpectralMLP(k=k, n_classes=n_classes, n_layers=n_layers,
                               loss_weights="uniform")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n_slices = k // 2 + 1
    warmup_curves: Dict[int, List[float]] = {}

    for epoch in range(1, warmup_epochs + 1):
        model.train()
        optimizer.zero_grad()
        lg = model(U)
        model.compute_loss(lg, labels, train_mask).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            lg = model(U)
        warmup_curves[epoch] = [accuracy(lg[j], labels, val_mask)
                                 for j in range(n_slices)]

    return warmup_curves


def select_cutoff(warmup_curves: Dict[int, List[float]], warmup_epochs: int) -> int:
    """
    Select cutoff j as the slice with peak validation accuracy at warmup_epochs.

    Args:
        warmup_curves:  Output of run_warmup().
        warmup_epochs:  Which epoch to use for cutoff selection.

    Returns:
        cutoff_j: Slice index j with highest val acc at epoch warmup_epochs.
    """
    val_at_W = warmup_curves[warmup_epochs]
    return int(np.argmax(val_at_W))


def uniform_weights(cutoff_j: int, k: int) -> np.ndarray:
    """
    Uniform weights over j=0..cutoff_j, zero elsewhere.
    """
    half = k // 2
    n_slices = half + 1
    w = np.zeros(n_slices)
    w[:cutoff_j + 1] = 1.0 / (cutoff_j + 1)
    return w
