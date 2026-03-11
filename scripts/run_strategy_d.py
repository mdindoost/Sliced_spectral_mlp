"""
Strategy D v2 (label energy cutoff, s[k/2] reference) — full experiment.

Fixes vs v1
-----------
  Bug 1: Shuffle permutation seed was 0; correct value is 42
         (matches src/evaluation/shuffle.py line 86).
  Bug 2: Unshuffled reference was test_coarse_d (from the D-cutoff model,
         which was selected by full-slice val with under-trained top dims).
         Now we run the canonical run_shuffle_diagnostic() for every dataset.
  Bug 3: _train_sliced() always selected by lg[-1]; when loss_cutoff=j*,
         dimensions above d_j* get no direct gradient, making full-slice val
         noisy.  Now selects by the last active slice (lg[loss_cutoff]).

Strategy D v2 change
--------------------
  reference = s[k//2]  (energy at slice boundary, not DC component s[0])
  threshold_bar = 0.10 * reference
  j* = largest j in {0, …, k//2−1} such that s[k//2+j] >= threshold_bar

Datasets: Cora, CiteSeer, PubMed, Cornell, Actor, Amazon-Photo.
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import load_dataset
from src.models.sliced_mlp import SlicedSpectralMLP
from src.models.baselines import StandardMLP, train_baseline
from src.evaluation.metrics import accuracy, per_slice_accuracy
from src.evaluation.shuffle import run_shuffle_diagnostic
from src.cutoff import strategy_c as strat_c
from src.cutoff.strategy_d import (
    compute_label_energy, select_cutoff_v2, make_Y_onehot,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
K        = 64
N_LAYERS = 2
LR       = 0.01
WD       = 5e-4
EPOCHS   = 200
SEED     = 42
FONT     = 10

plt.rcParams.update({
    "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
    "xtick.labelsize": FONT - 1, "ytick.labelsize": FONT - 1,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _train_sliced(model, U, labels, train_mask, val_mask,
                  loss_cutoff=None, epochs=EPOCHS, lr=LR, wd=WD):
    """
    Train SlicedSpectralMLP.

    Model selection: when loss_cutoff is set, select by val accuracy of
    the last active slice (lg[loss_cutoff]).  Without cutoff, select by
    the full slice lg[-1].  This matches run_citeseer.py line 69 and
    avoids selecting on undertrained high-frequency slices.

    Returns (best_val, val_coarse, val_full).
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state = 0.0, None
    val_coarse, val_full = [], []
    select_j = loss_cutoff if loss_cutoff is not None else (K // 2)

    for _ in range(epochs):
        model.train(); opt.zero_grad()
        lg = model(U)
        model.compute_loss(lg, labels, train_mask, loss_cutoff=loss_cutoff).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            lg = model(U)
        vc = accuracy(lg[0],        labels, val_mask)
        vf = accuracy(lg[-1],       labels, val_mask)
        vs = accuracy(lg[select_j], labels, val_mask)   # selection criterion
        val_coarse.append(vc)
        val_full.append(vf)
        if vs > best_val:
            best_val = vs
            best_state = {k2: v2.clone() for k2, v2 in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return best_val, val_coarse, val_full


def _save_label_energy_plot(energies, k, j_star, out_dir, dataset):
    """
    Plot s_i vs eigenvector index with:
      - k/2 boundary (black dashed)
      - threshold bar = 10% of s[k/2] (red dotted horizontal)
      - selected cutoff j* (green dashed)
    """
    half     = k // 2
    s_ref    = float(energies[half])
    bar      = 0.10 * s_ref
    idx      = np.arange(len(energies))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(idx, energies, color="steelblue", linewidth=1.4)
    ax.fill_between(idx, energies, alpha=0.15, color="steelblue")
    ax.axvline(half,          color="black", linestyle="--", linewidth=1.0,
               label=f"k/2={half} boundary  s[{half}]={s_ref:.5f}")
    ax.axvline(half + j_star, color="green", linestyle="--", linewidth=1.2,
               label=f"j*={j_star}  d*={half+j_star}")
    ax.axhline(bar,           color="tomato", linestyle=":",  linewidth=1.0,
               label=f"10% of s[{half}] = {bar:.5f}")
    ax.set_xlabel("Eigenvector index $i$")
    ax.set_ylabel("Label energy $s_i$")
    ax.set_title(
        f"Label energy v2 — {dataset.capitalize()}  "
        f"(s_0={energies[0]:.4f}  s[{half}]={s_ref:.5f})"
    )
    ax.legend(fontsize=FONT - 2)
    fig.tight_layout()
    path = os.path.join(out_dir, "label_energy_curve.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {path}")


def _save_per_slice_plot(slice_accs, k, j_star, mlp_full, mlp_half, out_dir, dataset):
    dims = np.array([k // 2 + j for j in range(len(slice_accs))])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(dims, slice_accs, color="steelblue", marker="o", markersize=3,
            linewidth=1.4, label="Sliced Strategy D v2")
    ax.axhline(mlp_full, color="black", linestyle="--", linewidth=1.1, label="MLP-full")
    ax.axhline(mlp_half, color="gray",  linestyle="--", linewidth=1.1, label="MLP-half")
    ax.axvline(k // 2 + j_star, color="green", linestyle=":", linewidth=1.2,
               label=f"D v2 cutoff j*={j_star}")
    ax.set_xlabel("Slice dimension $d_j$"); ax.set_ylabel("Test accuracy")
    ax.set_title(f"Per-slice accuracy — {dataset.capitalize()}  (Strategy D v2)")
    ax.legend(fontsize=FONT - 1)
    fig.tight_layout()
    path = os.path.join(out_dir, "per_slice_curve.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {path}")


def _save_training_curves(val_coarse, val_full, out_dir, dataset):
    ep = np.arange(1, len(val_coarse) + 1)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(ep, val_coarse, color="steelblue", linewidth=1.3, label="Coarse (j=0)")
    ax.plot(ep, val_full,   color="tomato",    linewidth=1.3, label="Full (j=k/2)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation accuracy")
    ax.set_title(f"Training curves — {dataset.capitalize()}  Strategy D v2")
    ax.legend(fontsize=FONT - 1); ax.set_xlim(1, len(ep))
    fig.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {path}")


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(name: str, split_idx: int = 0) -> dict | None:
    print(f"\n{'='*60}")
    print(f"  DATASET: {name.upper()}")
    print(f"{'='*60}")
    t0 = time.time()

    try:
        U, labels, train_mask, val_mask, test_mask, eigenvalues = load_dataset(
            name, k=K, split_idx=split_idx
        )
    except Exception as e:
        print(f"  SKIP — could not load: {e}")
        return None

    N          = U.shape[0]
    n_classes  = int(labels.max().item()) + 1
    print(f"  N={N}  classes={n_classes}  k={K}  train={int(train_mask.sum())}  "
          f"val={int(val_mask.sum())}  test={int(test_mask.sum())}")

    out_dir = os.path.join("outputs", name, "strategy_d")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. Label energy — Strategy D v2
    # ------------------------------------------------------------------
    Y_onehot = make_Y_onehot(labels, train_mask, n_classes)
    energies = compute_label_energy(U, Y_onehot)

    half  = K // 2
    e0    = float(energies[0])
    e_ref = float(energies[half])          # v2 reference = s[k/2]
    e_43  = float(energies[43]) if 43 < len(energies) else float("nan")
    e_63  = float(energies[K - 1])

    bar_v1 = 0.10 * e0
    bar_v2 = 0.10 * e_ref

    print(f"\n  Label energy:")
    print(f"    s[0]  (DC)             = {e0:.6f}   bar_v1 = {bar_v1:.6f}")
    print(f"    s[32] (slice boundary) = {e_ref:.6f}   bar_v2 = {bar_v2:.6f}")
    print(f"    s[43] (Cora peak ref)  = {e_43:.6f}")
    print(f"    s[63] (final eigvec)   = {e_63:.6f}")
    print(f"    max s_i = {energies.max():.6f}  at i={int(energies.argmax())}")

    j_star_v2 = select_cutoff_v2(U, Y_onehot, K, threshold=0.10)
    e_jstar   = float(energies[half + j_star_v2])
    d_star    = half + j_star_v2

    j_star_c  = strat_c.select_cutoff(eigenvalues, K)

    print(f"\n  Strategy D v2: j*={j_star_v2}  d*={d_star}"
          f"  s[{half+j_star_v2}]={e_jstar:.6f}")
    print(f"  Strategy C:    j*={j_star_c}   d*={half+j_star_c}")

    _save_label_energy_plot(energies, K, j_star_v2, out_dir, name)

    # ------------------------------------------------------------------
    # 1. Canonical shuffle diagnostic  (Bug 1+2 fixed: seed=42, canonical model)
    # ------------------------------------------------------------------
    print(f"\n  Canonical shuffle diagnostic (seed={SEED}, permutation seed={SEED}) …")
    diag = run_shuffle_diagnostic(
        U=U, labels=labels,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        k=K, n_classes=n_classes, n_layers=N_LAYERS,
        seed=SEED, epochs=EPOCHS, lr=LR, wd=WD,
    )
    coarse_unshuf  = diag["unshuffled_coarse"]
    coarse_shuf    = diag["shuffled_coarse"]
    shuffle_drop   = round(diag["coarse_drop_pp"], 1)
    print(f"    Unshuffled coarse = {coarse_unshuf:.4f}")
    print(f"    Shuffled   coarse = {coarse_shuf:.4f}")
    print(f"    Drop = {shuffle_drop:+.1f} pp")

    # ------------------------------------------------------------------
    # 2. Train Strategy D v2 model
    # ------------------------------------------------------------------
    print(f"\n  Training Strategy D v2 (cutoff j={j_star_v2}) …")
    torch.manual_seed(SEED)
    model_d = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                                 loss_weights="uniform")
    best_val_d, vc_d, vf_d = _train_sliced(
        model_d, U, labels, train_mask, val_mask, loss_cutoff=j_star_v2)
    model_d.eval()
    with torch.no_grad():
        lg_d = model_d(U)
    sa_d          = per_slice_accuracy(lg_d, labels, test_mask)
    test_best_d   = max(sa_d)
    test_coarse_d = sa_d[0]
    peak_j        = int(np.argmax(sa_d))
    print(f"    val={best_val_d:.4f}  test_best={test_best_d:.4f}  "
          f"test_coarse={test_coarse_d:.4f}  peak_j={peak_j}")

    # ------------------------------------------------------------------
    # 3. Train Strategy C model
    # ------------------------------------------------------------------
    print(f"  Training Strategy C (cutoff j={j_star_c}) …")
    torch.manual_seed(SEED)
    model_c = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                                 loss_weights="uniform")
    best_val_c, _, _ = _train_sliced(
        model_c, U, labels, train_mask, val_mask, loss_cutoff=j_star_c)
    model_c.eval()
    with torch.no_grad():
        lg_c = model_c(U)
    test_best_c = max(per_slice_accuracy(lg_c, labels, test_mask))
    print(f"    val={best_val_c:.4f}  test_best={test_best_c:.4f}")

    # ------------------------------------------------------------------
    # 4. Baselines
    # ------------------------------------------------------------------
    print(f"  Training MLP-full …")
    torch.manual_seed(SEED)
    mlp_f = StandardMLP(n_features=K, n_classes=n_classes, hidden_dim=K, n_layers=N_LAYERS)
    _, test_full = train_baseline(mlp_f, U, labels, train_mask, val_mask, test_mask,
                                  lr=LR, wd=WD, epochs=EPOCHS)

    print(f"  Training MLP-half …")
    torch.manual_seed(SEED)
    mlp_h = StandardMLP(n_features=half, n_classes=n_classes, hidden_dim=K, n_layers=N_LAYERS)
    _, test_half = train_baseline(mlp_h, U[:, :half], labels, train_mask, val_mask,
                                  test_mask, lr=LR, wd=WD, epochs=EPOCHS)
    print(f"    MLP-full={test_full:.4f}   MLP-half={test_half:.4f}")

    # ------------------------------------------------------------------
    # 5. Alignment check
    # ------------------------------------------------------------------
    known_peaks = {"cora": ("j≈11-15", 11, 15),
                   "citeseer": ("j=31 (no trunc)", 28, 31),
                   "pubmed":   ("j≈0-3", 0, 3)}
    print(f"\n  Per-slice peak: j={peak_j}  d={half+peak_j}  acc={max(sa_d):.4f}")
    if name in known_peaks:
        desc, lo, hi = known_peaks[name]
        aligned = lo <= j_star_v2 <= hi
        print(f"  D v2 j*={j_star_v2}  expected={desc}  "
              f"→ {'ALIGNED' if aligned else 'MISALIGNED'}")

    print(f"\n  SUMMARY:")
    print(f"    Strategy D v2 j*={j_star_v2}: test_best={test_best_d:.4f}  "
          f"test_coarse={test_coarse_d:.4f}")
    print(f"    Strategy C    j*={j_star_c}:  test_best={test_best_c:.4f}")
    print(f"    MLP-half:                      test={test_half:.4f}")
    print(f"    MLP-full:                      test={test_full:.4f}")
    print(f"    D v2 vs C:         {test_best_d - test_best_c:+.4f}")
    print(f"    D v2 vs MLP-half:  {test_best_d - test_half:+.4f}")
    print(f"    Shuffle drop: {shuffle_drop:+.1f} pp")

    _save_per_slice_plot(sa_d, K, j_star_v2, test_full, test_half, out_dir, name)
    _save_training_curves(vc_d, vf_d, out_dir, name)

    elapsed = round(time.time() - t0)
    print(f"  Elapsed: {elapsed}s")

    return {
        "dataset":                name,
        "N":                      N,
        "n_classes":              n_classes,
        "strategy_d_cutoff":      j_star_v2,
        "strategy_d_d_star":      d_star,
        "strategy_d_test_best":   round(test_best_d, 4),
        "strategy_d_test_coarse": round(test_coarse_d, 4),
        "strategy_c_cutoff":      j_star_c,
        "strategy_c_test_best":   round(test_best_c, 4),
        "mlp_half_test":          round(test_half, 4),
        "mlp_full_test":          round(test_full, 4),
        "shuffle_drop":           shuffle_drop,
        "s0":                     round(e0, 6),
        "s_ref_v2":               round(e_ref, 6),
        "s_jstar":                round(e_jstar, 6),
        "s63":                    round(e_63, 6),
    }


# ---------------------------------------------------------------------------
# Amazon-Photo (extra dataset)
# ---------------------------------------------------------------------------

def _load_amazon_photo_tensors():
    """
    Load Amazon-Photo, extract LCC, compute eigenvectors.
    Returns (U, labels, eigenvalues, train_mask, val_mask, test_mask, N, n_classes).
    """
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg
    from torch_geometric.datasets import Amazon
    from src.data.loaders import compute_eigenvectors

    dataset  = Amazon(root="./data", name="photo")
    data     = dataset[0]
    N_orig   = data.num_nodes
    n_classes = int(data.y.max().item()) + 1

    row_all = data.edge_index[0].numpy()
    col_all = data.edge_index[1].numpy()
    A_orig  = sp.csr_matrix(
        (np.ones(len(row_all)), (row_all, col_all)), shape=(N_orig, N_orig)
    )
    A_sym = A_orig + A_orig.T; A_sym.data[:] = 1.0
    n_comp, comp_ids = csg.connected_components(A_sym, directed=False)
    if n_comp > 1:
        lcc_id   = np.argmax(np.bincount(comp_ids))
        lcc_mask = comp_ids == lcc_id
        lcc_nodes = np.where(lcc_mask)[0]
        print(f"  [LCC] {n_comp} components, keeping {len(lcc_nodes)}/{N_orig}")
        remap = np.full(N_orig, -1, dtype=np.int64)
        remap[lcc_nodes] = np.arange(len(lcc_nodes))
        edge_mask = lcc_mask[row_all] & lcc_mask[col_all]
        row = remap[row_all[edge_mask]]
        col = remap[col_all[edge_mask]]
        labels_lcc = data.y.long()[torch.from_numpy(lcc_mask)]
        N = len(lcc_nodes)
    else:
        row, col   = row_all, col_all
        labels_lcc = data.y.long()
        N = N_orig

    A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(N, N))
    A = A + A.T; A.data[:] = 1.0
    U, eigenvalues = compute_eigenvectors(A, K)

    # Reproducible 60/20/20 split (no fixed PyG split for Amazon)
    torch.manual_seed(SEED)
    perm    = torch.randperm(N)
    n_train = int(0.6 * N); n_val = int(0.2 * N)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)
    train_mask[perm[:n_train]]               = True
    val_mask  [perm[n_train:n_train + n_val]] = True
    test_mask [perm[n_train + n_val:]]        = True

    return U, labels_lcc, eigenvalues, train_mask, val_mask, test_mask, N, n_classes


def run_amazon_photo() -> dict | None:
    print(f"\n{'='*60}")
    print(f"  DATASET: AMAZON-PHOTO (extra)")
    print(f"{'='*60}")
    t0 = time.time()

    try:
        U, labels, eigenvalues, train_mask, val_mask, test_mask, N, n_classes = \
            _load_amazon_photo_tensors()
    except Exception as e:
        print(f"  SKIP — Amazon-Photo unavailable: {e}")
        return None

    print(f"  N={N}  classes={n_classes}  k={K}  train={int(train_mask.sum())}  "
          f"val={int(val_mask.sum())}  test={int(test_mask.sum())}")

    out_dir = os.path.join("outputs", "amazon_photo", "strategy_d")
    os.makedirs(out_dir, exist_ok=True)

    # Label energy
    Y_onehot = make_Y_onehot(labels, train_mask, n_classes)
    energies = compute_label_energy(U, Y_onehot)
    half     = K // 2
    e0       = float(energies[0])
    e_ref    = float(energies[half])
    e_43     = float(energies[43]) if 43 < len(energies) else float("nan")
    e_63     = float(energies[K - 1])
    bar_v2   = 0.10 * e_ref

    print(f"\n  Label energy:")
    print(f"    s[0]={e0:.6f}  s[32]={e_ref:.6f}  bar_v2={bar_v2:.6f}")
    print(f"    s[43]={e_43:.6f}  s[63]={e_63:.6f}")

    j_star_v2 = select_cutoff_v2(U, Y_onehot, K, threshold=0.10)
    j_star_c  = strat_c.select_cutoff(eigenvalues, K)
    e_jstar   = float(energies[half + j_star_v2])
    print(f"  Strategy D v2: j*={j_star_v2}  d*={half+j_star_v2}")
    print(f"  Strategy C:    j*={j_star_c}   d*={half+j_star_c}")

    _save_label_energy_plot(energies, K, j_star_v2, out_dir, "amazon_photo")

    # Canonical shuffle diagnostic (Bug 1+2 fixed)
    print(f"  Canonical shuffle diagnostic …")
    diag = run_shuffle_diagnostic(
        U=U, labels=labels,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        k=K, n_classes=n_classes, n_layers=N_LAYERS,
        seed=SEED, epochs=EPOCHS, lr=LR, wd=WD,
    )
    coarse_unshuf = diag["unshuffled_coarse"]
    coarse_shuf   = diag["shuffled_coarse"]
    shuffle_drop  = round(diag["coarse_drop_pp"], 1)
    print(f"    Unshuffled={coarse_unshuf:.4f}  Shuffled={coarse_shuf:.4f}  "
          f"Drop={shuffle_drop:+.1f}pp")

    # Training
    print(f"  Training Strategy D v2 (cutoff j={j_star_v2}) …")
    torch.manual_seed(SEED)
    model_d = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                                 loss_weights="uniform")
    best_val_d, vc_d, vf_d = _train_sliced(
        model_d, U, labels, train_mask, val_mask, loss_cutoff=j_star_v2)
    model_d.eval()
    with torch.no_grad():
        lg_d = model_d(U)
    sa_d          = per_slice_accuracy(lg_d, labels, test_mask)
    test_best_d   = max(sa_d)
    test_coarse_d = sa_d[0]

    print(f"  Training Strategy C (cutoff j={j_star_c}) …")
    torch.manual_seed(SEED)
    model_c = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                                 loss_weights="uniform")
    best_val_c, _, _ = _train_sliced(
        model_c, U, labels, train_mask, val_mask, loss_cutoff=j_star_c)
    model_c.eval()
    with torch.no_grad():
        lg_c = model_c(U)
    test_best_c = max(per_slice_accuracy(lg_c, labels, test_mask))

    print(f"  Training baselines …")
    torch.manual_seed(SEED)
    mlp_f = StandardMLP(n_features=K, n_classes=n_classes, hidden_dim=K, n_layers=N_LAYERS)
    _, test_full = train_baseline(mlp_f, U, labels, train_mask, val_mask, test_mask,
                                  lr=LR, wd=WD, epochs=EPOCHS)
    torch.manual_seed(SEED)
    mlp_h = StandardMLP(n_features=half, n_classes=n_classes, hidden_dim=K, n_layers=N_LAYERS)
    _, test_half = train_baseline(mlp_h, U[:, :half], labels, train_mask, val_mask,
                                  test_mask, lr=LR, wd=WD, epochs=EPOCHS)

    print(f"\n  SUMMARY (Amazon-Photo):")
    print(f"    D v2 j*={j_star_v2}: test_best={test_best_d:.4f}  "
          f"test_coarse={test_coarse_d:.4f}")
    print(f"    Strategy C j*={j_star_c}: test_best={test_best_c:.4f}")
    print(f"    MLP-half={test_half:.4f}  MLP-full={test_full:.4f}")
    print(f"    Shuffle drop: {shuffle_drop:+.1f}pp")

    _save_per_slice_plot(sa_d, K, j_star_v2, test_full, test_half, out_dir, "amazon_photo")
    _save_training_curves(vc_d, vf_d, out_dir, "amazon_photo")

    elapsed = round(time.time() - t0)
    print(f"  Elapsed: {elapsed}s")

    return {
        "dataset":                "amazon_photo",
        "N":                      N,
        "n_classes":              n_classes,
        "strategy_d_cutoff":      j_star_v2,
        "strategy_d_d_star":      half + j_star_v2,
        "strategy_d_test_best":   round(test_best_d, 4),
        "strategy_d_test_coarse": round(test_coarse_d, 4),
        "strategy_c_cutoff":      j_star_c,
        "strategy_c_test_best":   round(test_best_c, 4),
        "mlp_half_test":          round(test_half, 4),
        "mlp_full_test":          round(test_full, 4),
        "shuffle_drop":           shuffle_drop,
        "s0":                     round(e0, 6),
        "s_ref_v2":               round(e_ref, 6),
        "s_jstar":                round(e_jstar, 6),
        "s63":                    round(e_63, 6),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = [
    ("cora",     0),
    ("citeseer", 0),
    ("pubmed",   0),
    ("cornell",  0),
    ("actor",    0),
]

if __name__ == "__main__":
    t_start = time.time()
    results = []

    for ds_name, split in DATASETS:
        r = run_dataset(ds_name, split_idx=split)
        if r is not None:
            results.append(r)

    r_photo = run_amazon_photo()
    if r_photo is not None:
        results.append(r_photo)

    # ------------------------------------------------------------------
    # Summary CSV
    # ------------------------------------------------------------------
    csv_path = "outputs/strategy_d_summary.csv"
    os.makedirs("outputs", exist_ok=True)
    fieldnames = [
        "dataset", "N", "n_classes",
        "strategy_d_cutoff", "strategy_d_d_star",
        "strategy_d_test_best", "strategy_d_test_coarse",
        "strategy_c_cutoff", "strategy_c_test_best",
        "mlp_half_test", "mlp_full_test",
        "shuffle_drop",
        "s0", "s_ref_v2", "s_jstar", "s63",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nSaved {csv_path}")

    # ------------------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------------------
    div = "=" * 108
    print(f"\n{div}")
    print("STRATEGY D v2 SUMMARY — all datasets")
    print(div)
    hdr = (f"{'Dataset':<14} {'N':>6}  {'j*':>3}  {'D v2 best':>10}  "
           f"{'C best':>8}  {'MLP-half':>9}  {'MLP-full':>9}  "
           f"{'Drop':>7}  {'D vs half':>10}")
    print(hdr)
    print("-" * 108)
    for r in results:
        dvh = r["strategy_d_test_best"] - r["mlp_half_test"]
        print(f"{r['dataset']:<14} {r['N']:>6}  {r['strategy_d_cutoff']:>3}  "
              f"{r['strategy_d_test_best']:>10.4f}  "
              f"{r['strategy_c_test_best']:>8.4f}  "
              f"{r['mlp_half_test']:>9.4f}  {r['mlp_full_test']:>9.4f}  "
              f"{r['shuffle_drop']:>+7.1f}pp  {dvh:>+10.4f}")
    print(div)

    print("\nALIGNMENT CHECK (Strategy D v2 j* vs expected per-slice peak):")
    known = {
        "cora":     ("j≈11-15",          11, 15),
        "citeseer": ("j=28-31 (no trunc)", 28, 31),
        "pubmed":   ("j≈0-3",              0,  3),
    }
    for r in results:
        ds = r["dataset"]
        if ds in known:
            desc, lo, hi = known[ds]
            j = r["strategy_d_cutoff"]
            aligned = lo <= j <= hi
            print(f"  {ds:<12}: j*={j:>2}  expected={desc:<22}  "
                  f"s[32]={r['s_ref_v2']:.6f}  bar={0.10*r['s_ref_v2']:.6f}  "
                  f"→ {'ALIGNED ✓' if aligned else 'MISALIGNED ✗'}")

    total = round(time.time() - t_start)
    print(f"\nTotal elapsed: {total}s ({total//60}m {total%60}s)")
