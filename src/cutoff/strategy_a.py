"""
Strategy A: Eigenvalue gap method for automatic loss cutoff selection.

Selects the cutoff j as the index with the largest gap between
consecutive eigenvalues in the slice range (lambda_{k//2+j} to lambda_{k//2+j+1}).

This is a pre-training method (no model training required).
"""

from __future__ import annotations

import numpy as np


def select_cutoff(eigenvalues: np.ndarray, k: int) -> int:
    """
    Select loss cutoff using the eigenvalue gap method.

    Args:
        eigenvalues: (k,) array of sorted eigenvalues (ascending).
        k:           Number of eigenvectors (full input width).

    Returns:
        cutoff_j: The slice index j ∈ {0, ..., k//2-1} where the largest
                  eigenvalue gap occurs.
    """
    half = k // 2
    n_slices = half + 1
    # eig_slices[j] = eigenvalue of the marginal eigenvector at slice j
    eig_slices = np.array([eigenvalues[min(half + j, k - 1)] for j in range(n_slices)])
    gaps = np.diff(eig_slices)  # shape (n_slices - 1,)
    cutoff_j = int(np.argmax(gaps))
    return cutoff_j


def uniform_weights(cutoff_j: int, k: int) -> np.ndarray:
    """
    Uniform weights over j=0..cutoff_j, zero elsewhere.

    Args:
        cutoff_j: Active slice range is j=0..cutoff_j (inclusive).
        k:        Number of eigenvectors.

    Returns:
        weights: (n_slices,) normalized numpy array.
    """
    half = k // 2
    n_slices = half + 1
    w = np.zeros(n_slices)
    w[:cutoff_j + 1] = 1.0 / (cutoff_j + 1)
    return w
