"""
Strategy C: Median eigenvalue threshold (the winner).

Selects slices whose marginal eigenvalue is at or below the median of
the slice-range eigenvalues (lambda_{k//2} to lambda_{k-1}).
Weights are inversely proportional to eigenvalue (smoother eigenvectors
get higher weight).

This is a pre-training method (no model training required) and is the
best-performing automatic strategy on homophilous graphs like Cora.
"""

from __future__ import annotations

import numpy as np


def compute_weights(eigenvalues: np.ndarray, k: int) -> np.ndarray:
    """
    Compute Strategy C weights using the median eigenvalue threshold.

    Args:
        eigenvalues: (k,) sorted eigenvalues (ascending).
        k:           Number of eigenvectors.

    Returns:
        weights: (n_slices,) normalized numpy array.
                 Active slices: those with lambda_{k//2+j} <= median.
                 Inactive slices: weight = 0.
    """
    half = k // 2
    n_slices = half + 1
    eig_slices = np.array([eigenvalues[min(half + j, k - 1)] for j in range(n_slices)])
    lambda_threshold = float(np.median(eig_slices))

    raw_w = np.zeros(n_slices)
    for j in range(n_slices):
        if eig_slices[j] <= lambda_threshold:
            raw_w[j] = 1.0 / max(eig_slices[j], 1e-6)

    if raw_w.sum() > 0:
        return raw_w / raw_w.sum()
    return raw_w


def select_cutoff(eigenvalues: np.ndarray, k: int) -> int:
    """
    Return the highest active slice index j under Strategy C.

    Args:
        eigenvalues: (k,) sorted eigenvalues (ascending).
        k:           Number of eigenvectors.

    Returns:
        cutoff_j: Last j with positive weight.
    """
    weights = compute_weights(eigenvalues, k)
    active = np.where(weights > 0)[0]
    if len(active) == 0:
        return k // 2  # fallback: all active
    return int(active[-1])


def get_threshold(eigenvalues: np.ndarray, k: int) -> float:
    """Return the median eigenvalue threshold used by Strategy C."""
    half = k // 2
    n_slices = half + 1
    eig_slices = np.array([eigenvalues[min(half + j, k - 1)] for j in range(n_slices)])
    return float(np.median(eig_slices))
