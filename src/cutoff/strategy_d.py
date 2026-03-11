"""
Strategy D: Label energy cutoff (pre-training, zero gradient cost).

For each eigenvector u_i, the label energy is:

    s_i = || u_i^T Y_train ||^2_F  /  || Y_train ||^2_F

where Y_train is the (N, C) one-hot label matrix restricted to training nodes
(non-training rows are zero).  s_i measures the fraction of label variance in
the training set that is explained by eigenvector i.

Cutoff rule
-----------
j* = largest j in {0, …, k//2 - 1} such that

    s_{k//2 + j}  >=  threshold * s_0

where s_0 = energy at eigenvector index 0 (the trivial DC component, which
serves as the reference "coarsest signal") and threshold = 0.10.

Intuition: keep slices while their marginal eigenvector still carries at
least 10 % of the label signal present in the DC component.  On homophilous
graphs the low-frequency slice eigenvectors pass easily; the cutoff fires when
entering the high-frequency band where class structure has faded.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_label_energy(
    U: np.ndarray,
    Y_onehot: np.ndarray,
) -> np.ndarray:
    """
    Compute per-eigenvector label energy s_i = ||u_i^T Y||^2_F / ||Y||^2_F.

    Args:
        U:        (N, k) float32/64 array — eigenvector matrix (all nodes).
        Y_onehot: (N, C) float32 array   — one-hot labels, zero for
                  non-training nodes (so only training labels contribute).

    Returns:
        energies: (k,) float64 array — s_i for i = 0 … k-1, sorted ascending
                  by eigenvalue (same order as columns of U).
    """
    U_np = U.numpy() if isinstance(U, torch.Tensor) else np.asarray(U, dtype=np.float64)
    Y_np = Y_onehot.numpy() if isinstance(Y_onehot, torch.Tensor) \
           else np.asarray(Y_onehot, dtype=np.float64)

    proj = U_np.T @ Y_np          # (k, C)
    num  = (proj ** 2).sum(axis=1)  # (k,)
    denom = (Y_np ** 2).sum()       # scalar = n_train (for one-hot labels)
    if denom == 0:
        return np.zeros(U_np.shape[1])
    return num / denom


def select_cutoff_v2(
    U: np.ndarray,
    Y_onehot: np.ndarray,
    k: int,
    threshold: float = 0.10,
) -> int:
    """
    Strategy D v2: use s[k//2] (label energy at the slice boundary) as
    the reference rather than s[0] (DC component).

    j* = LARGEST j in {0, …, k//2 - 1} such that

        s[k//2 + j]  >=  threshold * s[k//2]

    This anchors the threshold to the energy at the start of the slice
    range, making it scale-invariant across datasets.  Returns 0 if
    s[k//2] == 0 or if no j passes (energy collapses at j=0).
    """
    energies = compute_label_energy(U, Y_onehot)
    s_ref = float(energies[k // 2])
    bar   = threshold * s_ref if s_ref > 0 else 0.0

    half   = k // 2
    j_star = 0
    for j in range(half):          # j = 0 … k//2 − 1  →  indices 32 … 63
        idx = half + j
        if float(energies[idx]) >= bar:
            j_star = j

    return j_star


def select_cutoff(
    U: np.ndarray,
    Y_onehot: np.ndarray,
    k: int,
    threshold: float = 0.10,
) -> int:
    """
    Select the Strategy D loss cutoff j*.

    j* is the LARGEST j in {0, …, k//2 - 1} such that

        s_{k//2 + j}  >=  threshold * s_0

    Returns 0 if no eigenvector in the slice range clears the bar
    (energy has already collapsed at j=0), which means "use coarsest slice only".
    Returns k//2 - 1 if all eigenvectors pass (use full model).

    Args:
        U:         (N, k) eigenvector matrix.
        Y_onehot:  (N, C) one-hot training labels (zeros for non-train nodes).
        k:         Number of eigenvectors.
        threshold: Fraction of s_0 used as the cutoff bar (default 0.10).

    Returns:
        j_star: int in {0, …, k//2 - 1}.
    """
    energies = compute_label_energy(U, Y_onehot)
    s0 = float(energies[0])
    bar = threshold * s0 if s0 > 0 else 0.0

    half = k // 2
    j_star = 0
    for j in range(half):          # j = 0 … k//2 - 1  (indices 32 … 63)
        idx = half + j
        if float(energies[idx]) >= bar:
            j_star = j             # keep updating: last update = largest j

    return j_star


def uniform_weights(cutoff_j: int, k: int) -> np.ndarray:
    """
    Uniform weights over j=0..cutoff_j, zero for j > cutoff_j.

    Args:
        cutoff_j: Active slice range is j=0..cutoff_j (inclusive).
        k:        Number of eigenvectors.

    Returns:
        weights: (n_slices,) normalized numpy array.
    """
    n_slices = k // 2 + 1
    w = np.zeros(n_slices)
    w[:cutoff_j + 1] = 1.0 / (cutoff_j + 1)
    return w


def make_Y_onehot(labels: torch.Tensor, train_mask: torch.Tensor, n_classes: int) -> np.ndarray:
    """
    Build the (N, C) one-hot matrix with zeros for non-training nodes.

    Args:
        labels:     (N,) long tensor of class indices.
        train_mask: (N,) bool tensor.
        n_classes:  Number of classes C.

    Returns:
        Y_onehot: (N, C) float32 numpy array.
    """
    N = labels.shape[0]
    Y = np.zeros((N, n_classes), dtype=np.float32)
    train_idx = torch.where(train_mask)[0].numpy()
    for i in train_idx:
        Y[i, int(labels[i])] = 1.0
    return Y
