"""
Data pipeline: load a PyG graph dataset and compute the k smallest
eigenvectors of the symmetric normalised graph Laplacian.

Supported datasets
------------------
  Homophilous : 'cora', 'citeseer'
  Heterophilous: 'cornell'

Returns
-------
  U           : (N, k) float32 tensor  — eigenvector matrix
  labels      : (N,)   long tensor     — class labels
  train_mask  : (N,)   bool tensor
  val_mask    : (N,)   bool tensor
  test_mask   : (N,)   bool tensor
  eigenvalues : (k,)   float64 ndarray — sorted eigenvalues (ascending)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
from scipy.sparse.linalg import eigsh
import torch
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    name: str,
    root: str = "./data",
    k: int = 64,
    split_idx: int = 0,
) -> tuple:
    """
    Load dataset and compute eigenvectors.

    Args:
        name:      Dataset name (case-insensitive).
        root:      Directory for PyG to cache raw/processed data.
        k:         Number of eigenvectors to compute.
        split_idx: Which split to use for datasets with multiple splits
                   (e.g. WebKB provides 10 splits; Planetoid has only 1).
    Returns:
        (U, labels, train_mask, val_mask, test_mask, eigenvalues)
    """
    name_lower = name.lower()
    data = _load_pyg(name_lower, root)

    N_orig = data.num_nodes

    # ----------------------------------------------------------------
    # Extract Largest Connected Component (LCC)
    # ----------------------------------------------------------------
    row_all = data.edge_index[0].numpy()
    col_all = data.edge_index[1].numpy()
    vals_all = np.ones(len(row_all), dtype=np.float64)
    A_orig = sp.csr_matrix((vals_all, (row_all, col_all)), shape=(N_orig, N_orig))
    A_sym = (A_orig + A_orig.T)
    A_sym.data[:] = 1.0

    n_comp, comp_labels = csg.connected_components(A_sym, directed=False)
    if n_comp > 1:
        comp_sizes = np.bincount(comp_labels)
        lcc_id = np.argmax(comp_sizes)
        lcc_mask_np = (comp_labels == lcc_id)
        lcc_nodes = np.where(lcc_mask_np)[0]
        # remap node indices
        remap = np.full(N_orig, -1, dtype=np.int64)
        remap[lcc_nodes] = np.arange(len(lcc_nodes))
        # keep only edges within LCC
        edge_mask = lcc_mask_np[row_all] & lcc_mask_np[col_all]
        row = remap[row_all[edge_mask]]
        col = remap[col_all[edge_mask]]
        lcc_mask_torch = torch.from_numpy(lcc_mask_np)
        labels_orig = data.y.long()
        labels_lcc  = labels_orig[lcc_mask_torch]
        if data.train_mask.dim() == 1:
            train_mask = data.train_mask[lcc_mask_torch]
            val_mask   = data.val_mask[lcc_mask_torch]
            test_mask  = data.test_mask[lcc_mask_torch]
        else:
            idx = min(split_idx, data.train_mask.shape[1] - 1)
            train_mask = data.train_mask[:, idx][lcc_mask_torch]
            val_mask   = data.val_mask[:, idx][lcc_mask_torch]
            test_mask  = data.test_mask[:, idx][lcc_mask_torch]
        N = len(lcc_nodes)
    else:
        row = row_all
        col = col_all
        N = N_orig
        labels_lcc = data.y.long()
        if data.train_mask.dim() == 1:
            train_mask = data.train_mask
            val_mask   = data.val_mask
            test_mask  = data.test_mask
        else:
            idx = min(split_idx, data.train_mask.shape[1] - 1)
            train_mask = data.train_mask[:, idx]
            val_mask   = data.val_mask[:, idx]
            test_mask  = data.test_mask[:, idx]

    k = min(k, N - 2)  # eigsh requires k < N; leave a small margin

    # ----------------------------------------------------------------
    # Build symmetric normalised Laplacian  L = I - D^{-1/2} A D^{-1/2}
    # ----------------------------------------------------------------
    vals = np.ones(len(row), dtype=np.float64)
    A = sp.csr_matrix((vals, (row, col)), shape=(N, N))
    A = (A + A.T)                              # symmetrise (handles directed edges)
    A.data[:] = 1.0                            # binarise after symmetrising

    deg = np.asarray(A.sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    L = sp.eye(N, format="csr") - D_inv_sqrt @ A @ D_inv_sqrt
    L = L.astype(np.float64)

    # ----------------------------------------------------------------
    # Compute k smallest eigenvectors.
    #
    # Strategy:
    #   • N ≤ 5000 (Cora ~2708, Cornell ~183): convert to dense and use
    #     scipy.linalg.eigh — exact, no convergence issues, runs in < 1s.
    #   • Larger graphs: sparse eigsh with a small diagonal regularisation
    #     (L + ε·I) so that sigma=0 shift-invert never hits a singular matrix.
    # ----------------------------------------------------------------
    if N <= 5_000:
        import scipy.linalg
        L_dense = L.toarray()
        eigenvalues_all, eigenvectors_all = scipy.linalg.eigh(L_dense)
        # eigh returns all eigenvalues in ascending order
        eigenvalues  = eigenvalues_all[:k]
        eigenvectors = eigenvectors_all[:, :k]
    else:
        # Regularise slightly so the zero eigenvalue doesn't cause a
        # singular system in shift-invert mode.
        eps_reg = 1e-8
        L_reg = L + eps_reg * sp.eye(N, format="csr")
        eigenvalues, eigenvectors = eigsh(
            L_reg, k=k, which="LM", sigma=0.0, tol=1e-8, maxiter=20_000
        )
        eigenvalues -= eps_reg   # undo the shift so λ₀ ≈ 0 again
        order = np.argsort(eigenvalues)
        eigenvalues  = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

    U = torch.from_numpy(eigenvectors.astype(np.float32))

    return U, labels_lcc, train_mask, val_mask, test_mask, eigenvalues


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_pyg(name: str, root: str):
    if name in ("cora", "citeseer", "pubmed"):
        dataset = Planetoid(root=root, name=name.capitalize())
    elif name == "cornell":
        dataset = WebKB(root=root, name="Cornell")
    elif name == "actor":
        dataset = Actor(root=f"{root}/actor")
    elif name == "squirrel":
        dataset = WikipediaNetwork(root=root, name="squirrel")
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            "Supported: 'cora', 'citeseer', 'pubmed', 'cornell', 'actor', 'squirrel'."
        )
    return dataset[0]
