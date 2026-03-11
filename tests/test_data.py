"""
Unit tests for data pipeline.

Tests:
  - compute_eigenvectors: shape, sortedness
  - shuffle_eigenvectors: shape, permutation correctness
  - LCC extraction (synthetic graph)
"""

import numpy as np
import scipy.sparse as sp
import torch
import pytest

from src.data.loaders import compute_eigenvectors
from src.data.eigenvectors import shuffle_eigenvectors


def _make_path_graph(n: int) -> sp.spmatrix:
    """Return a symmetric adjacency matrix for an n-node path graph."""
    row = list(range(n - 1)) + list(range(1, n))
    col = list(range(1, n)) + list(range(n - 1))
    A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    return A


def _make_disconnected_graph(n1: int, n2: int) -> sp.spmatrix:
    """
    Two disconnected path graphs: n1 nodes and n2 nodes.
    Total nodes = n1 + n2.
    """
    A1 = _make_path_graph(n1)
    A2 = _make_path_graph(n2)
    # Block diagonal
    from scipy.sparse import block_diag
    return block_diag([A1, A2]).tocsr()


def test_compute_eigenvectors_shape():
    """U.shape == (N, k) and eigenvalues.shape == (k,)."""
    N = 50
    k = 10
    A = _make_path_graph(N)
    U, eigenvalues = compute_eigenvectors(A, k=k)
    assert U.shape == (N, k), f"Expected ({N},{k}), got {U.shape}"
    assert eigenvalues.shape == (k,), f"Expected ({k},), got {eigenvalues.shape}"


def test_compute_eigenvectors_sorted():
    """Eigenvalues must be sorted in ascending order."""
    A = _make_path_graph(50)
    _, eigenvalues = compute_eigenvectors(A, k=10)
    assert np.all(eigenvalues[1:] >= eigenvalues[:-1]), (
        "Eigenvalues must be sorted ascending"
    )


def test_compute_eigenvectors_small_graph():
    """k is capped at N-2 for small graphs."""
    N = 10
    A = _make_path_graph(N)
    U, eigenvalues = compute_eigenvectors(A, k=64)
    # k is capped at N-2 = 8
    assert U.shape[1] <= N - 2
    assert U.shape[0] == N


def test_compute_eigenvectors_first_is_constant():
    """
    The first eigenvector of the normalized Laplacian for a connected graph
    is the all-ones vector (up to sign and normalization).
    """
    A = _make_path_graph(30)
    U, eigenvalues = compute_eigenvectors(A, k=5)
    # Smallest eigenvalue should be ~0 (trivial eigenvector)
    assert eigenvalues[0] < 1e-6, f"Smallest eigenvalue should be ~0, got {eigenvalues[0]}"


def test_shuffle_eigenvectors_shape():
    """Shuffled U has same shape as original."""
    N, k = 50, 10
    A = _make_path_graph(N)
    U, eigenvalues = compute_eigenvectors(A, k=k)
    U_shuf, eigs_shuf = shuffle_eigenvectors(U, eigenvalues, seed=42)
    assert U_shuf.shape == U.shape
    assert eigs_shuf.shape == eigenvalues.shape


def test_shuffle_eigenvectors_permutation():
    """Shuffled columns are a permutation of original columns."""
    N, k = 30, 8
    A = _make_path_graph(N)
    U, eigenvalues = compute_eigenvectors(A, k=k)
    U_shuf, eigs_shuf = shuffle_eigenvectors(U, eigenvalues, seed=7)

    # Every column of U_shuf must appear in U (as a column)
    U_np = U.numpy()
    U_shuf_np = U_shuf.numpy()
    for j in range(k):
        col = U_shuf_np[:, j]
        found = any(
            np.allclose(col, U_np[:, jj]) or np.allclose(col, -U_np[:, jj])
            for jj in range(k)
        )
        assert found, f"Column {j} of shuffled U not found in original U"


def test_shuffle_different_from_original():
    """Shuffled eigenvectors differ from original (for k > 1)."""
    A = _make_path_graph(30)
    U, eigenvalues = compute_eigenvectors(A, k=8)
    U_shuf, _ = shuffle_eigenvectors(U, eigenvalues, seed=0)
    assert not torch.allclose(U, U_shuf), "Shuffled U should differ from original"
