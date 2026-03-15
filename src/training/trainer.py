"""
Training utilities for SlicedSpectralMLP.

Functions
---------
  train_sliced(model, U, labels, train_mask, val_mask, epochs, lr, wd,
               loss_cutoff, heatmap_epochs, heatmap_dir)
               — train loop with early stopping + optional gradient heatmaps
  train_epoch(model, U, labels, train_mask, optimizer, loss_cutoff)
               — single training epoch
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set

import torch
import torch.optim

from src.models.sliced_mlp import SlicedSpectralMLP
from src.evaluation.metrics import accuracy
from src.utils.visualization import save_grad_heatmap


def train_sliced(
    model: SlicedSpectralMLP,
    U: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.01,
    wd: float = 5e-4,
    loss_cutoff: Optional[int] = None,
    heatmap_epochs: Optional[Set[int]] = None,
    heatmap_dir: Optional[str] = None,
) -> tuple:
    """
    Train SlicedSpectralMLP and return best checkpoint state.

    Args:
        model:          SlicedSpectralMLP instance.
        U:              (N, k) eigenvector matrix.
        labels:         (N,) class labels.
        train_mask:     (N,) boolean training mask.
        val_mask:       (N,) boolean validation mask.
        epochs:         Number of training epochs.
        lr:             Adam learning rate.
        wd:             Adam weight decay.
        loss_cutoff:    If set, zero loss for slices j > loss_cutoff.
        heatmap_epochs: Set of epoch indices at which to save grad heatmaps.
        heatmap_dir:    Directory for grad heatmap PNGs.

    Returns:
        (best_val_acc, val_accs_coarse, val_accs_full)
        Model is restored to its best-val-acc checkpoint before returning.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val_acc = 0.0
    best_state: Optional[Dict] = None
    val_accs_coarse: List[float] = []
    val_accs_full: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        all_logits = model(U)
        loss = model.compute_loss(all_logits, labels, train_mask,
                                   loss_cutoff=loss_cutoff)
        loss.backward()

        if heatmap_epochs and epoch in heatmap_epochs and model.W[0].grad is not None:
            if heatmap_dir:
                os.makedirs(heatmap_dir, exist_ok=True)
                save_grad_heatmap(torch.abs(model.W[0].grad), epoch, heatmap_dir)

        optimizer.step()

        model.eval()
        with torch.no_grad():
            all_logits = model(U)

        coarse_val = accuracy(all_logits[0],  labels, val_mask)
        full_val   = accuracy(all_logits[-1], labels, val_mask)
        val_accs_coarse.append(coarse_val)
        val_accs_full.append(full_val)

        # IMPORTANT — Bug 3 fix (2026-03)
        # Model selection must use the LAST ACTIVE slice (j=loss_cutoff),
        # NOT the full slice (j=n_slices-1).
        #
        # Reason: when loss_cutoff < n_slices, the full slice receives no
        # loss gradient and its val accuracy is essentially random noise.
        # Using it for checkpoint selection caused wrong checkpoints to be
        # saved and corrupted all cutoff sensitivity results.
        #
        # The regression test in tests/test_bug3_model_selection.py
        # verifies this behavior. Do not change this without updating
        # that test first.
        track_j = loss_cutoff if loss_cutoff is not None else len(all_logits) - 1
        track_val = accuracy(all_logits[track_j], labels, val_mask)

        if track_val > best_val_acc:
            best_val_acc = track_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_acc, val_accs_coarse, val_accs_full


def train_epoch(
    model: SlicedSpectralMLP,
    U: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_cutoff: Optional[int] = None,
) -> float:
    """
    Single training epoch. Returns scalar loss value.
    """
    model.train()
    optimizer.zero_grad()
    all_logits = model(U)
    loss = model.compute_loss(all_logits, labels, train_mask,
                               loss_cutoff=loss_cutoff)
    loss.backward()
    optimizer.step()
    return loss.item()
