"""
Generate all 6 report figures for SlicedSpectralMLP.

Figures are saved to outputs/report_figures/ at 200 dpi, font 11pt,
white background, tight layout.

Data sources (no retraining):
  - Shuffle tables:    outputs/{dataset}/shuffle_table.txt  (per-slice test acc)
  - Cora per-slice:    outputs/checkpoints/best_model.pt    (inference only)
  - Cora train curves: run 200-epoch forward/backward loop, capture val curves
  - Eigenvalue spectra: load_dataset() for Cora, CiteSeer, PubMed (no training)
  - Grad heatmaps:     outputs/grad_heatmaps/epoch_00{10,50}.png
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
from scipy.stats import linregress

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import load_dataset
from src.models.sliced_mlp import SlicedSpectralMLP
from src.models.baselines import StandardMLP, train_baseline
from src.evaluation.metrics import accuracy, per_slice_accuracy

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
FONT_SIZE = 11
plt.rcParams.update({
    "font.size":         FONT_SIZE,
    "axes.titlesize":    FONT_SIZE,
    "axes.labelsize":    FONT_SIZE,
    "xtick.labelsize":   FONT_SIZE - 1,
    "ytick.labelsize":   FONT_SIZE - 1,
    "legend.fontsize":   FONT_SIZE - 1,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

OUT_DIR = "outputs/report_figures"
os.makedirs(OUT_DIR, exist_ok=True)
DPI = 200

K = 64
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def parse_shuffle_table(path: str):
    """Return (dims, unshuffled_acc, shuffled_acc) as float arrays."""
    dims, unshuf, shuf = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 5 and parts[0].strip().isdigit():
                dims.append(int(parts[1].strip()))
                unshuf.append(float(parts[2].strip()))
                shuf.append(float(parts[3].strip()))
    return np.array(dims), np.array(unshuf), np.array(shuf)


def cora_inference_from_checkpoint(k=K, n_classes=7):
    """Load Cora data + best checkpoint → per-slice test accuracy array."""
    print("  Loading Cora for inference …")
    U, labels, train_mask, val_mask, test_mask, _ = load_dataset("cora", k=k)
    model = SlicedSpectralMLP(k=k, n_classes=n_classes, n_layers=2,
                               loss_weights="uniform", eigenvalues=None)
    state = torch.load("outputs/checkpoints/best_model.pt", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        all_logits = model(U)
    accs = per_slice_accuracy(all_logits, labels, test_mask)
    dims = np.array([k // 2 + j for j in range(len(accs))])
    return dims, np.array(accs)


def cora_training_curves(k=K, n_classes=7, epochs=200, lr=0.01, wd=5e-4, seed=SEED):
    """Run 200-epoch Cora training (no checkpoint save) to capture val curves."""
    print("  Training Cora 200 epochs for curves …")
    U, labels, train_mask, val_mask, test_mask, _ = load_dataset("cora", k=k)
    torch.manual_seed(seed)
    model = SlicedSpectralMLP(k=k, n_classes=n_classes, n_layers=2,
                               loss_weights="uniform", eigenvalues=None)
    # Baselines
    torch.manual_seed(seed)
    mlp_full = StandardMLP(n_features=k,     n_classes=n_classes, hidden_dim=k, n_layers=2)
    torch.manual_seed(seed)
    mlp_half = StandardMLP(n_features=k//2,  n_classes=n_classes, hidden_dim=k, n_layers=2)

    opt_s = torch.optim.Adam(model.parameters(),    lr=lr, weight_decay=wd)
    opt_f = torch.optim.Adam(mlp_full.parameters(), lr=lr, weight_decay=wd)
    opt_h = torch.optim.Adam(mlp_half.parameters(), lr=lr, weight_decay=wd)

    vc_s, vf_s = [], []          # sliced: coarse j=0, full j=32
    vf_full_bl, vf_half_bl = [], []  # baselines: full val acc each epoch

    X_full = U
    X_half = U[:, :k//2]

    for _ in range(epochs):
        # Sliced
        model.train()
        opt_s.zero_grad()
        logits = model(X_full)
        model.compute_loss(logits, labels, train_mask).backward()
        opt_s.step()
        model.eval()
        with torch.no_grad():
            logits = model(X_full)
        vc_s.append(accuracy(logits[0],  labels, val_mask))
        vf_s.append(accuracy(logits[-1], labels, val_mask))

        # MLP-full
        mlp_full.train()
        opt_f.zero_grad()
        import torch.nn.functional as F
        loss_f = F.cross_entropy(mlp_full(X_full)[train_mask], labels[train_mask])
        loss_f.backward()
        opt_f.step()
        mlp_full.eval()
        with torch.no_grad():
            vf_full_bl.append(accuracy(mlp_full(X_full), labels, val_mask))

        # MLP-half
        mlp_half.train()
        opt_h.zero_grad()
        loss_h = F.cross_entropy(mlp_half(X_half)[train_mask], labels[train_mask])
        loss_h.backward()
        opt_h.step()
        mlp_half.eval()
        with torch.no_grad():
            vf_half_bl.append(accuracy(mlp_half(X_half), labels, val_mask))

    return (np.array(vc_s), np.array(vf_s),
            np.array(vf_full_bl), np.array(vf_half_bl))


# ---------------------------------------------------------------------------
# Figure 1 — per_slice_all_datasets.png
# ---------------------------------------------------------------------------

def fig1_per_slice_all_datasets():
    print("\n[Figure 1] per_slice_all_datasets.png")

    # --- load data ---
    cora_dims, cora_acc = cora_inference_from_checkpoint()

    cs_dims, cs_acc, _   = parse_shuffle_table("outputs/citeseer/shuffle_table.txt")
    pm_dims, pm_acc, _   = parse_shuffle_table("outputs/pubmed/shuffle_table.txt")
    ac_dims, ac_acc, _   = parse_shuffle_table("outputs/actor/shuffle_table.txt")

    datasets = [
        dict(name="Cora",     N=2485,  dims=cora_dims, acc=cora_acc,
             mlp_full=0.6448, mlp_half=0.7082),
        dict(name="CiteSeer", N=2120,  dims=cs_dims,   acc=cs_acc,
             mlp_full=0.5505, mlp_half=0.6365),
        dict(name="PubMed",   N=19717, dims=pm_dims,   acc=pm_acc,
             mlp_full=0.7260, mlp_half=0.7560),
        dict(name="Actor",    N=7600,  dims=ac_dims,   acc=ac_acc,
             mlp_full=0.2524, mlp_half=0.2525),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharey=False)
    axes = axes.flatten()

    for ax, d in zip(axes, datasets):
        ax.plot(d["dims"], d["acc"], color="steelblue", marker="o",
                markersize=3, linewidth=1.4, label="Sliced-dense(uniform)")
        ax.axhline(d["mlp_full"], color="black",  linestyle="--",
                   linewidth=1.2, label="MLP-full")
        ax.axhline(d["mlp_half"], color="gray",   linestyle="--",
                   linewidth=1.2, label="MLP-half")
        ax.set_title(f"{d['name']}  (N={d['N']:,})")
        ax.set_xlabel("Slice dimension $d_j$")
        ax.set_ylabel("Test accuracy")
        ax.legend(fontsize=FONT_SIZE - 2, loc="best")
        ax.set_xlim(d["dims"].min() - 0.5, d["dims"].max() + 0.5)

    fig.suptitle("Spectral resolution curves — four dataset regimes",
                 fontsize=FONT_SIZE + 1, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "per_slice_all_datasets.png")


# ---------------------------------------------------------------------------
# Figure 2 — shuffle_drop_barchart.png
# ---------------------------------------------------------------------------

def fig2_shuffle_drop_barchart():
    print("\n[Figure 2] shuffle_drop_barchart.png")

    # Sorted by drop magnitude (most negative at top)
    data = [
        ("Cora",     -17.9, "green"),
        ("CiteSeer", -17.9, "green"),
        ("PubMed",   -10.2, "green"),
        ("Cornell",   -8.1, "orange"),
        ("Actor",     -0.1, "red"),
        ("Squirrel",  +2.7, "red"),
    ]
    labels   = [d[0] for d in data]
    drops    = [d[1] for d in data]
    colors   = [d[2] for d in data]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, drops, color=colors, edgecolor="white",
                   height=0.55, alpha=0.85)
    ax.axvline(0,   color="black", linewidth=1.0, linestyle="--")
    ax.axvline(-10, color="navy",  linewidth=1.0, linestyle="--", alpha=0.7)
    ax.text(-10.2, -0.65, "architecture helps\nthreshold (−10 pp)",
            fontsize=FONT_SIZE - 2, color="navy", ha="right", va="top")

    for bar, val in zip(bars, drops):
        xpos = val - 0.3 if val < 0 else val + 0.3
        ha   = "right" if val < 0 else "left"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f} pp", va="center", ha=ha,
                fontsize=FONT_SIZE - 1, fontweight="bold")

    ax.set_xlabel("Shuffle coarse drop (pp)")
    ax.set_title("Shuffle diagnostic — all datasets",
                 fontsize=FONT_SIZE + 1, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(-22, 7)
    fig.tight_layout()
    _save(fig, "shuffle_drop_barchart.png")


# ---------------------------------------------------------------------------
# Figure 3 — scatter_drop_vs_gain.png
# ---------------------------------------------------------------------------

def fig3_scatter_drop_vs_gain():
    print("\n[Figure 3] scatter_drop_vs_gain.png")

    # (dataset, shuffle_drop, sliced_best - mlp_full) all in pp
    points = [
        ("Cora",     -17.9,  +9.5),
        ("CiteSeer", -17.9, +11.6),
        ("PubMed",   -10.2,  +5.1),
        ("Cornell",   -8.1,  -0.8),
        ("Actor",     -0.1,  -0.1),
    ]
    label_offsets = {
        "Cora":     (-1.0,  +0.8),
        "CiteSeer": (+0.3,  +0.8),
        "PubMed":   (+0.3,  +0.4),
        "Cornell":  (+0.3,  -1.2),
        "Actor":    (+0.3,  +0.4),
    }

    names  = [p[0] for p in points]
    xs     = np.array([p[1] for p in points])
    ys     = np.array([p[2] for p in points])
    colors = ["green" if y > 0 else "red" for y in ys]

    slope, intercept, r, *_ = linregress(xs, ys)
    x_fit = np.linspace(-20, 5, 100)
    y_fit = slope * x_fit + intercept

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(xs, ys, c=colors, s=80, zorder=5, edgecolors="black", linewidths=0.5)
    ax.plot(x_fit, y_fit, color="steelblue", linewidth=1.4,
            linestyle="-", alpha=0.7, label=f"Best fit (r={r:.2f})")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)

    for name, x, y in zip(names, xs, ys):
        dx, dy = label_offsets[name]
        ax.annotate(name, (x, y), xytext=(x + dx, y + dy),
                    fontsize=FONT_SIZE - 1,
                    arrowprops=dict(arrowstyle="-", color="gray",
                                   lw=0.8, shrinkA=4, shrinkB=4))

    ax.set_xlabel("Shuffle coarse drop (pp)")
    ax.set_ylabel("Sliced-best minus MLP-full (pp)")
    ax.set_title("Shuffle drop predicts Sliced advantage over MLP-full",
                 fontsize=FONT_SIZE + 1, fontweight="bold")
    ax.set_xlim(-20, 5)
    ax.set_ylim(-5, 15)
    ax.legend(fontsize=FONT_SIZE - 1)
    fig.tight_layout()
    _save(fig, "scatter_drop_vs_gain.png")


# ---------------------------------------------------------------------------
# Figure 4 — grad_heatmap_comparison.png
# ---------------------------------------------------------------------------

def fig4_grad_heatmap_comparison():
    print("\n[Figure 4] grad_heatmap_comparison.png")

    ep10_path = "outputs/grad_heatmaps/epoch_0010.png"
    ep50_path = "outputs/grad_heatmaps/epoch_0050.png"

    img10 = imread(ep10_path)
    img50 = imread(ep50_path)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, img, subtitle in zip(
        axes,
        [img10, img50],
        ["Epoch 10 (noisy, uniform)", "Epoch 50 (structured, top-left dominant)"],
    ):
        ax.imshow(img)
        ax.set_title(subtitle, fontsize=FONT_SIZE)
        ax.axis("off")

    fig.suptitle(r"Gradient density evolution — Cora  |  $|W^{[0]}.\mathrm{grad}|$",
                 fontsize=FONT_SIZE + 1, fontweight="bold")
    fig.tight_layout()
    _save(fig, "grad_heatmap_comparison.png")


# ---------------------------------------------------------------------------
# Figure 5 — eigenvalue_spectra_comparison.png
# ---------------------------------------------------------------------------

def fig5_eigenvalue_spectra():
    print("\n[Figure 5] eigenvalue_spectra_comparison.png")

    spectra = {}
    for ds in ("cora", "citeseer", "pubmed"):
        print(f"  Loading {ds} eigenvalues …")
        _, _, _, _, _, evals = load_dataset(ds, k=K)
        spectra[ds] = evals  # shape (K,), sorted ascending

    # Strategy C cutoffs (from previous experiments)
    # j=16 for all three (median threshold in slice range)
    strat_c = {"cora": 16, "citeseer": 16, "pubmed": 16}

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    for ax, (ds, title, N) in zip(axes, [
        ("cora",     "Cora\n(N=2,485)",    2485),
        ("citeseer", "CiteSeer\n(N=2,120)", 2120),
        ("pubmed",   "PubMed\n(N=19,717)", 19717),
    ]):
        evals = spectra[ds]
        idx   = np.arange(len(evals))

        ax.plot(idx, evals, color="steelblue", linewidth=1.5)
        ax.fill_between(idx, evals, alpha=0.15, color="steelblue")

        # k/2 boundary
        ax.axvline(K // 2, color="black", linestyle="--",
                   linewidth=1.0, label=f"$k/2={K//2}$ boundary")

        # Strategy C cutoff — the threshold eigenvalue index in the full spectrum
        sc_j   = strat_c[ds]
        sc_idx = K // 2 + sc_j          # index in 0..K-1 spectrum
        ax.axvline(sc_idx, color="green", linestyle="--",
                   linewidth=1.0, label=f"Strategy C cutoff (j={sc_j})")

        # Relative spread in slice range [K//2 .. K-1]
        slice_evals = evals[K // 2:]
        spread_pct  = 100.0 * (slice_evals.max() - slice_evals.min()) / slice_evals.max()
        ax.text(0.97, 0.97, f"slice spread\n{spread_pct:.0f}%",
                transform=ax.transAxes, fontsize=FONT_SIZE - 2,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        ax.set_title(title)
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("$\\lambda$" if ds == "cora" else "")
        ax.legend(fontsize=FONT_SIZE - 2, loc="upper left")
        ax.set_xlim(0, K - 1)

    fig.suptitle("Eigenvalue spectra — Strategy C works when slice spread is large",
                 fontsize=FONT_SIZE + 1, fontweight="bold")
    fig.tight_layout()
    _save(fig, "eigenvalue_spectra_comparison.png")


# ---------------------------------------------------------------------------
# Figure 6 — training_curves_cora.png
# ---------------------------------------------------------------------------

def fig6_training_curves_cora():
    print("\n[Figure 6] training_curves_cora.png")

    vc_s, vf_s, vf_full_bl, vf_half_bl = cora_training_curves()
    epochs_x = np.arange(1, len(vc_s) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: coarse vs full for Sliced-dense-uniform
    ax = axes[0]
    ax.plot(epochs_x, vc_s,    color="steelblue", linewidth=1.4,
            label="Coarse ($j=0$, $d=32$)")
    ax.plot(epochs_x, vf_s,    color="tomato",    linewidth=1.4,
            label="Full ($j=32$, $d=64$)")
    ax.set_title("Coarse vs Full slice — Sliced-dense (Cora)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.legend()
    ax.set_xlim(1, len(vc_s))

    # Right: all three models
    ax = axes[1]
    ax.plot(epochs_x, vf_s,       color="steelblue", linewidth=1.4,
            label="Sliced-dense (full slice)")
    ax.plot(epochs_x, vf_full_bl, color="black",     linewidth=1.2,
            linestyle="--", label="MLP-full")
    ax.plot(epochs_x, vf_half_bl, color="gray",      linewidth=1.2,
            linestyle="--", label="MLP-half")
    ax.set_title("All models — validation accuracy (Cora)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.legend()
    ax.set_xlim(1, len(vc_s))

    fig.suptitle("Training curves — Cora  (k=64, seed=42, 200 epochs)",
                 fontsize=FONT_SIZE + 1, fontweight="bold")
    fig.tight_layout()
    _save(fig, "training_curves_cora.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Saving all figures to {OUT_DIR}/  (DPI={DPI}, font={FONT_SIZE}pt)")
    fig1_per_slice_all_datasets()
    fig2_shuffle_drop_barchart()
    fig3_scatter_drop_vs_gain()
    fig4_grad_heatmap_comparison()
    fig5_eigenvalue_spectra()
    fig6_training_curves_cora()
    print("\nDone. All 6 figures saved.")
