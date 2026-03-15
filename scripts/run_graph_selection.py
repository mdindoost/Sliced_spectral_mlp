"""
Graph selection experiment: compare original graph vs feature-kNN graph
eigenvectors across all datasets using the shuffle diagnostic as selector.

For each dataset × graph type:
  1. Build adjacency matrix
  2. Extract LCC
  3. Compute k=64 eigenvectors of the symmetric normalised Laplacian
  4. Run shuffle diagnostic (seed=42, 200 epochs)
  5. Record coarse_drop_pp as selector

Graph types
-----------
  0  original   — PyG adjacency matrix
  1  knn_avg    — kNN with k = avg degree of original graph
  2  knn10      — kNN with k = 10
  3  knn20      — kNN with k = 20
  4  tfidf10    — kNN(10) after TF-IDF transform
  5  hybrid10   — union of original + kNN(10)

Datasets
--------
  All 6 graphs: Cora, CiteSeer, Actor, Cornell
  Graphs 0-2 only: PubMed, Squirrel, Amazon-Photo

Usage
-----
  python scripts/run_graph_selection.py [--dataset cora] [--graphs 0,1,2]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import compute_eigenvectors
from src.evaluation.shuffle import run_shuffle_diagnostic
from src.models.baselines import StandardMLP, train_baseline

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.preprocessing import normalize as sk_normalize
except ImportError:
    print("ERROR: scikit-learn required. Install: pip install scikit-learn")
    sys.exit(1)

try:
    import torch_geometric.datasets as pyg_datasets
    from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork
    from torch_geometric.datasets import Amazon
except ImportError:
    print("ERROR: torch_geometric required.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------

# Graphs to evaluate per dataset (index into GRAPH_NAMES)
_DATASET_GRAPHS = {
    "cora":         [0, 1, 2, 3, 4, 5],
    "citeseer":     [0, 1, 2, 3, 4, 5],
    "actor":        [0, 1, 2, 3, 4, 5],
    "cornell":      [0, 1, 2, 3, 4, 5],
    "pubmed":       [0, 1, 2],
    "squirrel":     [0, 1, 2],
    "amazon-photo": [0, 1, 2],
}

GRAPH_NAMES = {
    0: "original",
    1: "knn_avg",
    2: "knn10",
    3: "knn20",
    4: "tfidf10",
    5: "hybrid10",
}

K_EIGVEC = 64


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_pyg_raw(name: str, root: str = "./data"):
    """Load PyG Data object (raw, no eigenvectors)."""
    name_lower = name.lower()
    if name_lower in ("cora", "citeseer", "pubmed"):
        ds = Planetoid(root=root, name=name_lower.capitalize())
        return ds[0]
    elif name_lower == "cornell":
        ds = WebKB(root=root, name="Cornell")
        return ds[0]
    elif name_lower == "actor":
        ds = Actor(root=f"{root}/actor")
        return ds[0]
    elif name_lower == "squirrel":
        ds = WikipediaNetwork(root=root, name="squirrel")
        return ds[0]
    elif name_lower in ("amazon-photo", "amazon_photo", "photo"):
        ds = Amazon(root=root, name="Photo")
        return ds[0]
    else:
        raise ValueError(f"Unknown dataset: {name}")


def _extract_lcc(data, split_idx: int = 0):
    """
    Extract LCC, return:
      A_sym_lcc (scipy sparse, binary, symmetric, N_lcc x N_lcc)
      features_lcc (N_lcc x F) numpy float32 or None
      labels_lcc (N_lcc,) long tensor
      train_mask, val_mask, test_mask (N_lcc,) bool tensors
      N (int)
    """
    import scipy.sparse.csgraph as csg

    N_orig = data.num_nodes
    row_all = data.edge_index[0].numpy()
    col_all = data.edge_index[1].numpy()
    vals_all = np.ones(len(row_all), dtype=np.float64)

    A_orig = sp.csr_matrix((vals_all, (row_all, col_all)), shape=(N_orig, N_orig))
    A_sym_full = A_orig + A_orig.T
    A_sym_full.data[:] = 1.0

    n_comp, comp_labels = csg.connected_components(A_sym_full, directed=False)

    if n_comp > 1:
        comp_sizes = np.bincount(comp_labels)
        lcc_id = np.argmax(comp_sizes)
        lcc_mask_np = (comp_labels == lcc_id)
        lcc_nodes = np.where(lcc_mask_np)[0]
        remap = np.full(N_orig, -1, dtype=np.int64)
        remap[lcc_nodes] = np.arange(len(lcc_nodes))
        edge_mask = lcc_mask_np[row_all] & lcc_mask_np[col_all]
        row = remap[row_all[edge_mask]]
        col = remap[col_all[edge_mask]]
        N = len(lcc_nodes)
        lcc_torch = torch.from_numpy(lcc_mask_np)
        labels_lcc = data.y.long()[lcc_torch]
        features_lcc = data.x[lcc_torch].numpy().astype(np.float32) \
            if data.x is not None else None
        has_mask = hasattr(data, "train_mask") and data.train_mask is not None
        if has_mask:
            if data.train_mask.dim() == 1:
                train_mask = data.train_mask[lcc_torch]
                val_mask   = data.val_mask[lcc_torch]
                test_mask  = data.test_mask[lcc_torch]
            else:
                idx = min(split_idx, data.train_mask.shape[1] - 1)
                train_mask = data.train_mask[:, idx][lcc_torch]
                val_mask   = data.val_mask[:, idx][lcc_torch]
                test_mask  = data.test_mask[:, idx][lcc_torch]
        else:
            train_mask = val_mask = test_mask = None
    else:
        row = row_all
        col = col_all
        N = N_orig
        labels_lcc = data.y.long()
        features_lcc = data.x.numpy().astype(np.float32) \
            if data.x is not None else None
        has_mask = hasattr(data, "train_mask") and data.train_mask is not None
        if has_mask:
            if data.train_mask.dim() == 1:
                train_mask = data.train_mask
                val_mask   = data.val_mask
                test_mask  = data.test_mask
            else:
                idx = min(split_idx, data.train_mask.shape[1] - 1)
                train_mask = data.train_mask[:, idx]
                val_mask   = data.val_mask[:, idx]
                test_mask  = data.test_mask[:, idx]
        else:
            train_mask = val_mask = test_mask = None

    vals = np.ones(len(row), dtype=np.float64)
    A_lcc = sp.csr_matrix((vals, (row, col)), shape=(N, N))
    A_sym_lcc = A_lcc + A_lcc.T
    A_sym_lcc.data[:] = 1.0

    return A_sym_lcc, features_lcc, labels_lcc, train_mask, val_mask, test_mask, N


def _make_amazon_photo_splits(N: int, seed: int = 42):
    """Random 60/20/20 split for Amazon-Photo (no fixed PyG split)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    n_train = int(0.6 * N)
    n_val   = int(0.2 * N)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True
    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def _avg_degree(A_sym: sp.spmatrix) -> int:
    """Return rounded average node degree."""
    deg = np.asarray(A_sym.sum(axis=1)).ravel()
    return max(1, int(round(deg.mean())))


def _build_knn_adj(features: np.ndarray, k_nn: int, tfidf: bool = False) -> sp.spmatrix:
    """
    Build a symmetric binary kNN adjacency matrix from node features.

    Args:
        features: (N, F) feature matrix (raw counts or already normalised).
        k_nn:     Number of nearest neighbors.
        tfidf:    Apply TF-IDF transformation before computing kNN.

    Returns:
        A_knn: (N, N) symmetric binary scipy csr_matrix.
    """
    N = features.shape[0]
    X = features.copy()

    if tfidf:
        # Treat each node's feature vector as a "document"
        transformer = TfidfTransformer(norm="l2", smooth_idf=True)
        X = transformer.fit_transform(X).toarray().astype(np.float32)
    else:
        # L2-normalise for cosine similarity
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        X = X / norms

    k_nn_clamped = min(k_nn + 1, N - 1)  # +1 because query node is its own neighbor
    nbrs = NearestNeighbors(n_neighbors=k_nn_clamped, metric="cosine", n_jobs=-1)
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Build edge list (exclude self-loop at index 0)
    rows, cols = [], []
    for i in range(N):
        for j_idx in range(1, indices.shape[1]):
            j = indices[i, j_idx]
            rows.append(i)
            cols.append(j)

    vals = np.ones(len(rows), dtype=np.float64)
    A_knn = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    A_knn = A_knn + A_knn.T
    A_knn.data[:] = 1.0
    return A_knn


def _build_hybrid(A_orig: sp.spmatrix, A_knn: sp.spmatrix) -> sp.spmatrix:
    """Union of two binary symmetric adjacency matrices."""
    A_union = A_orig + A_knn
    A_union.data[:] = 1.0
    return A_union


# ---------------------------------------------------------------------------
# Baselines: MLP-half and MLP-full
# ---------------------------------------------------------------------------

def _run_mlp_baselines(U, labels, train_mask, val_mask, test_mask,
                       k, n_classes, n_layers=2, seed=42, epochs=200,
                       lr=0.01, wd=5e-4):
    """Run MLP-full (k features) and MLP-half (k//2 features) baselines."""
    results = {}
    for name, n_feat in [("mlp_full", k), ("mlp_half", k // 2)]:
        torch.manual_seed(seed)
        X = U[:, :n_feat]
        bl = StandardMLP(n_features=n_feat, n_classes=n_classes,
                         hidden_dim=k, n_layers=n_layers)
        bv, ta = train_baseline(bl, X, labels, train_mask, val_mask, test_mask,
                                lr=lr, wd=wd, epochs=epochs)
        results[name] = {"best_val": float(bv), "test": float(ta)}
    return results


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_graph_variant(
    dataset_name: str,
    graph_id: int,
    A_sym: sp.spmatrix,
    features: np.ndarray | None,
    A_orig: sp.spmatrix | None,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    N: int,
    k: int = K_EIGVEC,
    seed: int = 42,
    epochs: int = 200,
    verbose: bool = True,
) -> dict:
    """
    Compute eigenvectors for one graph variant and run shuffle diagnostic.

    Returns dict with all metrics for this (dataset, graph) pair.
    """
    graph_name = GRAPH_NAMES[graph_id]
    n_classes = int(labels.max().item()) + 1
    k_eff = min(k, N - 2)

    if verbose:
        print(f"  [{graph_name}] N={N}, k={k_eff}, n_classes={n_classes}")

    U, eigenvalues = compute_eigenvectors(A_sym, k_eff)

    diag = run_shuffle_diagnostic(
        U, labels, train_mask, val_mask, test_mask,
        k=k_eff, n_classes=n_classes, seed=seed, epochs=epochs,
    )

    bl = _run_mlp_baselines(U, labels, train_mask, val_mask, test_mask,
                             k=k_eff, n_classes=n_classes, seed=seed, epochs=epochs)

    row = {
        "dataset":           dataset_name,
        "graph_id":          graph_id,
        "graph_name":        graph_name,
        "N":                 N,
        "n_classes":         n_classes,
        "k_eff":             k_eff,
        "coarse_drop_pp":    round(float(diag["coarse_drop_pp"]), 3),
        "full_drop_pp":      round(float(diag["full_drop_pp"]), 3),
        "unshuffled_coarse": round(float(diag["unshuffled_coarse"]), 4),
        "unshuffled_full":   round(float(diag["unshuffled_full"]), 4),
        "shuffled_coarse":   round(float(diag["shuffled_coarse"]), 4),
        "shuffled_full":     round(float(diag["shuffled_full"]), 4),
        "mlp_half_test":     round(float(bl["mlp_half"]["test"]), 4),
        "mlp_full_test":     round(float(bl["mlp_full"]["test"]), 4),
        "slice_curve_unshuf": [round(x, 4) for x in diag["slice_test_unshuffled"]],
        "slice_curve_shuf":   [round(x, 4) for x in diag["slice_test_shuffled"]],
    }

    if verbose:
        print(f"    coarse_drop={row['coarse_drop_pp']:+.1f}pp  "
              f"unshuf_coarse={row['unshuffled_coarse']:.4f}  "
              f"mlp_half={row['mlp_half_test']:.4f}")

    return row


def run_dataset(
    dataset_name: str,
    graph_ids: list[int] | None = None,
    out_dir: str = "outputs/graph_selection",
    verbose: bool = True,
    k: int = K_EIGVEC,
    epochs: int = 200,
):
    """
    Run graph selection experiment for one dataset, all requested graph types.
    Saves per-dataset JSON and appends to CSV.
    """
    if graph_ids is None:
        graph_ids = _DATASET_GRAPHS.get(dataset_name, [0, 1, 2])

    print(f"\n{'='*65}")
    print(f"Dataset: {dataset_name.upper()}  |  graphs: {graph_ids}")
    print(f"{'='*65}")

    # ----------------------------------------------------------------
    # Load raw PyG data and extract LCC
    # ----------------------------------------------------------------
    data = _load_pyg_raw(dataset_name)

    # Amazon-Photo has no fixed split — generate one
    has_fixed_split = dataset_name.lower() not in ("amazon-photo", "amazon_photo", "photo")

    A_orig, features, labels, train_mask, val_mask, test_mask, N = \
        _extract_lcc(data, split_idx=0)

    if not has_fixed_split:
        train_mask, val_mask, test_mask = _make_amazon_photo_splits(N, seed=42)

    if verbose:
        print(f"  N={N}, n_classes={int(labels.max())+1}, "
              f"train={train_mask.sum().item()}, "
              f"val={val_mask.sum().item()}, "
              f"test={test_mask.sum().item()}")
        if features is not None:
            print(f"  features: {features.shape[1]}D")

    # Average degree (for knn_avg)
    avg_deg = _avg_degree(A_orig)
    if verbose:
        print(f"  avg_degree={avg_deg}")

    # ----------------------------------------------------------------
    # Build all graph variants
    # ----------------------------------------------------------------
    def get_adj(graph_id: int) -> sp.spmatrix:
        if graph_id == 0:
            return A_orig
        if features is None:
            print(f"  WARNING: no features for {dataset_name}, skipping graph {graph_id}")
            return None
        if graph_id == 1:
            return _build_knn_adj(features, k_nn=avg_deg, tfidf=False)
        elif graph_id == 2:
            return _build_knn_adj(features, k_nn=10, tfidf=False)
        elif graph_id == 3:
            return _build_knn_adj(features, k_nn=20, tfidf=False)
        elif graph_id == 4:
            return _build_knn_adj(features, k_nn=10, tfidf=True)
        elif graph_id == 5:
            A_knn = _build_knn_adj(features, k_nn=10, tfidf=False)
            return _build_hybrid(A_orig, A_knn)
        else:
            raise ValueError(f"Unknown graph_id: {graph_id}")

    # ----------------------------------------------------------------
    # Run each graph variant
    # ----------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for gid in graph_ids:
        print(f"\n  --- graph {gid}: {GRAPH_NAMES[gid]} ---")
        A = get_adj(gid)
        if A is None:
            continue
        row = run_graph_variant(
            dataset_name=dataset_name,
            graph_id=gid,
            A_sym=A,
            features=features,
            A_orig=A_orig,
            labels=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            N=N,
            k=k,
            epochs=epochs,
            verbose=verbose,
        )
        rows.append(row)

    # ----------------------------------------------------------------
    # Save per-dataset JSON (full slice curves)
    # ----------------------------------------------------------------
    json_path = os.path.join(out_dir, f"{dataset_name}_results.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  Saved {json_path}")

    # ----------------------------------------------------------------
    # Save per-dataset summary table
    # ----------------------------------------------------------------
    _print_dataset_summary(dataset_name, rows)

    return rows


def _print_dataset_summary(dataset_name: str, rows: list[dict]):
    """Print a summary table for one dataset."""
    print(f"\n--- Summary: {dataset_name.upper()} ---")
    hdr = f"{'Graph':<12} {'CoarseDrop':>11} {'UnshufCoarse':>13} {'UnshufFull':>11} {'MLP-half':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"  {r['graph_name']:<10} {r['coarse_drop_pp']:>+10.1f}pp "
              f"{r['unshuffled_coarse']:>13.4f} {r['unshuffled_full']:>11.4f} "
              f"{r['mlp_half_test']:>9.4f}")

    # Which graph has the largest (most negative) coarse_drop?
    if rows:
        best = min(rows, key=lambda r: r["coarse_drop_pp"])
        print(f"  => Best graph (largest drop): {best['graph_name']} "
              f"({best['coarse_drop_pp']:+.1f}pp)")


# ---------------------------------------------------------------------------
# CSV append helper
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "dataset", "graph_id", "graph_name", "N", "n_classes", "k_eff",
    "coarse_drop_pp", "full_drop_pp",
    "unshuffled_coarse", "unshuffled_full",
    "shuffled_coarse", "shuffled_full",
    "mlp_half_test", "mlp_full_test",
]


def append_to_csv(rows: list[dict], csv_path: str):
    """Append rows to the summary CSV, writing header if file is new."""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(dataset_name: str, rows: list[dict], out_dir: str):
    """
    Generate 3 figures per dataset:
      1. Per-slice curves (unshuffled) for all graph variants
      2. Shuffle drop bar chart (coarse_drop_pp)
      3. Unshuffled coarse vs MLP-half scatter
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figures")
        return

    fig_dir = os.path.join(out_dir, dataset_name)
    os.makedirs(fig_dir, exist_ok=True)

    colors = plt.cm.tab10.colors
    graph_names = [r["graph_name"] for r in rows]

    # Figure 1: per-slice curves (unshuffled)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, r in enumerate(rows):
        curve = r["slice_curve_unshuf"]
        x = list(range(len(curve)))
        ax.plot(x, curve, color=colors[i % 10], label=r["graph_name"], linewidth=1.5)
    ax.set_xlabel("Slice j", fontsize=11)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_title(f"{dataset_name.upper()} — per-slice accuracy (unshuffled)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "per_slice_curves.png"), dpi=200)
    plt.close(fig)

    # Figure 2: shuffle drop bar chart
    fig, ax = plt.subplots(figsize=(5, 3))
    drops = [r["coarse_drop_pp"] for r in rows]
    bar_colors = ["#d62728" if d >= 0 else ("#2ca02c" if d < -5 else "#ff7f0e")
                  for d in drops]
    bars = ax.barh(graph_names, drops, color=bar_colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coarse-slice drop under shuffling (pp)", fontsize=10)
    ax.set_title(f"{dataset_name.upper()} — shuffle diagnostic", fontsize=11)
    for bar, drop in zip(bars, drops):
        ax.text(drop + (0.3 if drop >= 0 else -0.3),
                bar.get_y() + bar.get_height() / 2,
                f"{drop:+.1f}pp", va="center", ha="left" if drop >= 0 else "right",
                fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "shuffle_drop_bar.png"), dpi=200)
    plt.close(fig)

    # Figure 3: unshuffled_coarse vs mlp_half
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, r in enumerate(rows):
        ax.scatter(r["mlp_half_test"], r["unshuffled_coarse"],
                   color=colors[i % 10], s=60, zorder=3)
        ax.annotate(r["graph_name"],
                    (r["mlp_half_test"], r["unshuffled_coarse"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=8)
    # diagonal reference
    all_vals = [r["mlp_half_test"] for r in rows] + [r["unshuffled_coarse"] for r in rows]
    lo, hi = min(all_vals) - 0.01, max(all_vals) + 0.01
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5, label="y=x")
    ax.set_xlabel("MLP-half test accuracy", fontsize=11)
    ax.set_ylabel("Sliced coarse test accuracy", fontsize=11)
    ax.set_title(f"{dataset_name.upper()} — coarse vs MLP-half", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "coarse_vs_mlp_half.png"), dpi=200)
    plt.close(fig)

    print(f"  Saved figures to {fig_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Graph selection experiment")
    p.add_argument("--dataset", type=str, default=None,
                   help="Run one dataset only (e.g. cora)")
    p.add_argument("--graphs", type=str, default=None,
                   help="Comma-separated graph IDs to run (e.g. 0,1,2)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--k", type=int, default=K_EIGVEC)
    p.add_argument("--out-dir", type=str, default="outputs/graph_selection")
    p.add_argument("--no-figures", action="store_true",
                   help="Skip figure generation")
    return p.parse_args()


def main():
    args = _parse_args()

    # Priority order (full runs first, then restricted)
    priority_order = [
        "cora", "actor", "citeseer", "cornell", "pubmed", "squirrel", "amazon-photo"
    ]

    if args.dataset is not None:
        datasets_to_run = [args.dataset.lower()]
    else:
        datasets_to_run = priority_order

    csv_path = os.path.join(args.out_dir, "summary.csv")
    all_rows = []

    for ds in datasets_to_run:
        if args.graphs is not None:
            graph_ids = [int(x) for x in args.graphs.split(",")]
        else:
            graph_ids = _DATASET_GRAPHS.get(ds, [0, 1, 2])

        rows = run_dataset(
            dataset_name=ds,
            graph_ids=graph_ids,
            out_dir=args.out_dir,
            k=args.k,
            epochs=args.epochs,
        )
        all_rows.extend(rows)
        append_to_csv(rows, csv_path)

        if not args.no_figures:
            make_figures(ds, rows, args.out_dir)

    # Final cross-dataset summary
    print(f"\n{'='*65}")
    print("CROSS-DATASET SUMMARY")
    print(f"{'='*65}")
    print(f"{'Dataset':<14} {'Graph':<12} {'CoarseDrop':>11} {'UnshufCoarse':>13} {'MLP-half':>9}")
    print("-" * 65)

    current_ds = None
    for r in all_rows:
        if r["dataset"] != current_ds:
            current_ds = r["dataset"]
        print(f"  {r['dataset']:<12} {r['graph_name']:<12} "
              f"{r['coarse_drop_pp']:>+10.1f}pp "
              f"{r['unshuffled_coarse']:>13.4f} "
              f"{r['mlp_half_test']:>9.4f}")

    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
