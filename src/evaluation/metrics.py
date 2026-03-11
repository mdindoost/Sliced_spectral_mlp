"""
Evaluation metrics for SlicedSpectralMLP.

Functions
---------
  accuracy(logits, labels, mask)            — classification accuracy
  per_slice_accuracy(all_logits, labels, mask) — accuracy for each slice
  best_slice_accuracy(all_logits, labels, mask) — max accuracy across slices
"""

from __future__ import annotations

from typing import List

import torch


def accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Fraction of correctly classified nodes within mask."""
    return (logits.argmax(-1)[mask] == labels[mask]).float().mean().item()


def per_slice_accuracy(
    all_logits: List[torch.Tensor],
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> List[float]:
    """
    Compute accuracy for each slice.

    Args:
        all_logits: output of model.forward() — list of (N, n_classes) tensors.
        labels:     (N,) class labels.
        mask:       (N,) boolean mask.

    Returns:
        List of float accuracies, one per slice.
    """
    return [accuracy(logits, labels, mask) for logits in all_logits]


def best_slice_accuracy(
    all_logits: List[torch.Tensor],
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Return the maximum accuracy across all slices."""
    return max(per_slice_accuracy(all_logits, labels, mask))
