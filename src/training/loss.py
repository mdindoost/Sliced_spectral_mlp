"""
Loss functions and weight strategies for SlicedSpectralMLP.

Functions
---------
  compute_loss_weights(strategy, k, eigenvalues) — compute normalized slice weights
  sliced_loss(all_logits, labels, mask, weights, loss_cutoff) — weighted CE loss
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def compute_loss_weights(
    strategy: str,
    k: int,
    eigenvalues: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute normalized loss weights for each slice.

    Args:
        strategy:    One of {'uniform', 'coarse', 'eigenvalue'}.
        k:           Number of eigenvectors (full input width).
        eigenvalues: Array of k eigenvalues; required for 'eigenvalue' strategy.

    Returns:
        weights: (n_slices,) float32 numpy array that sums to 1.0,
                 where n_slices = k//2 + 1.
    """
    half = k // 2
    n = half + 1

    if strategy == "uniform":
        w = np.ones(n, dtype=np.float32)

    elif strategy == "coarse":
        w = np.arange(float(n), 0.0, -1.0, dtype=np.float32)

    elif strategy == "eigenvalue":
        if eigenvalues is None:
            raise ValueError(
                "eigenvalues must be supplied when strategy='eigenvalue'"
            )
        eig = np.asarray(eigenvalues, dtype=np.float32)
        indices = [min(half + j, len(eig) - 1) for j in range(n)]
        lam = np.clip(eig[indices], a_min=1e-6, a_max=None)
        w = 1.0 / lam

    else:
        raise ValueError(
            f"Unknown strategy='{strategy}'. "
            "Choose from: 'uniform', 'coarse', 'eigenvalue'."
        )

    return w / w.sum()


def sliced_loss(
    all_logits: List[torch.Tensor],
    labels: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    loss_cutoff: Optional[int] = None,
) -> torch.Tensor:
    """
    Weighted sum of per-slice cross-entropy losses.

    Args:
        all_logits:  List of (N, n_classes) tensors from model.forward().
        labels:      (N,) integer class labels.
        mask:        (N,) boolean mask selecting nodes for this split.
        weights:     (n_slices,) tensor of normalized weights.
        loss_cutoff: If set, slices j > loss_cutoff contribute zero loss;
                     active slices are re-weighted uniformly to 1/n_active.

    Returns:
        Scalar loss tensor.
    """
    loss = torch.zeros(1, device=weights.device, dtype=torch.float32)[0]
    if loss_cutoff is not None:
        n_active = loss_cutoff + 1
        for j, logits in enumerate(all_logits):
            if j > loss_cutoff:
                continue
            ce = F.cross_entropy(logits[mask], labels[mask])
            loss = loss + ce / n_active
    else:
        for j, logits in enumerate(all_logits):
            ce = F.cross_entropy(logits[mask], labels[mask])
            loss = loss + weights[j] * ce
    return loss
