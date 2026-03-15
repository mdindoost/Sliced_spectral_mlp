"""
Generate four summary figures from outputs/fixed_split_multiseed/.

1. main_results_bars.png   — grouped bar chart (5 datasets, excl. Actor)
2. shuffle_drop.png        — horizontal bar chart sorted by magnitude
3. scatter_drop_vs_gain.png — scatter with regression line
4. jcut_stability_heatmap.png — heatmap of j_cut selection counts

Usage
-----
    python scripts/make_multiseed_figures.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = "outputs/fixed_split_multiseed"
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MAIN_CSV   = os.path.join(BASE, "tables", "main_results.csv")
JCUT_CSV   = os.path.join(BASE, "tables", "jcut_stability.csv")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
main = pd.read_csv(MAIN_CSV)
jcut = pd.read_csv(JCUT_CSV)

# Pretty display names
DISPLAY = {
    "cora":         "Cora",
    "citeseer":     "CiteSeer",
    "pubmed":       "PubMed",
    "amazon_photo": "Amazon-Photo",
    "actor":        "Actor",
    "cornell":      "Cornell",
}
main["label"] = main["dataset"].map(DISPLAY)
jcut["label"] = jcut["dataset"].map(DISPLAY)

# ---------------------------------------------------------------------------
# Figure 1 — grouped bar chart (exclude Actor)
# ---------------------------------------------------------------------------
def fig_main_results():
    df = main[main["dataset"] != "actor"].copy()
    datasets = df["label"].tolist()
    n = len(datasets)
    x = np.arange(n)
    width = 0.22

    methods = [
        ("MLP-half",    "MLP-half_mean",    "MLP-half_std",    "#4e79a7"),
        ("Sliced-dense","Sliced-dense_mean", "Sliced-dense_std","#f28e2b"),
        ("Sliced-best", "Sliced-best_mean",  "Sliced-best_std", "#59a14f"),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, (label, mean_col, std_col, color) in enumerate(methods):
        means = df[mean_col].values * 100
        stds  = df[std_col].values  * 100
        bars = ax.bar(x + (i - 1) * width, means, width,
                      label=label, color=color, alpha=0.88,
                      yerr=stds, capsize=3,
                      error_kw={"elinewidth": 1.2, "ecolor": "black", "alpha": 0.7})

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.set_title("Main results: MLP-half vs Sliced (5 seeds, fixed splits)", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Y range: start just below minimum
    ymin = min((df[m].values * 100 - df[s].values * 100).min()
               for _, m, s, _ in methods)
    ax.set_ylim(max(0, ymin - 5), 100)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "main_results_bars.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 2 — horizontal shuffle drop bar chart
# ---------------------------------------------------------------------------
def fig_shuffle_drop():
    df = main.copy()
    df["drop_mean"] = df["shuffle_drop_mean"]
    df["drop_std"]  = df["shuffle_drop_std"]

    # Sort by magnitude (most negative first)
    df = df.sort_values("drop_mean")

    def bar_color(v):
        if v <= -5:
            return "#2ca02c"    # green
        elif v <= -1:
            return "#ff7f0e"    # yellow-orange
        else:
            return "#d62728"    # red

    colors = [bar_color(v) for v in df["drop_mean"]]
    labels = df["label"].tolist()
    means  = df["drop_mean"].values
    stds   = df["drop_std"].values

    fig, ax = plt.subplots(figsize=(6.5, 4))
    y = np.arange(len(labels))

    ax.barh(y, means, xerr=stds, color=colors, alpha=0.88,
            capsize=3, error_kw={"elinewidth": 1.2, "ecolor": "black", "alpha": 0.7})

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Shuffle drop (pp)", fontsize=11)
    ax.set_title("Coarse-slice shuffle drop (mean ± std, 5 seeds)", fontsize=12)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(-5, color="green",  linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(-1, color="orange", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#2ca02c", label="≤ −5 pp (strong signal)"),
        mpatches.Patch(color="#ff7f0e", label="−1 to −5 pp (marginal)"),
        mpatches.Patch(color="#d62728", label="> −1 pp (no signal)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "shuffle_drop.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3 — scatter drop vs gain
# ---------------------------------------------------------------------------
def fig_scatter():
    df = main.copy()
    x  = df["shuffle_drop_mean"].values          # drop (pp)
    xe = df["shuffle_drop_std"].values
    y  = (df["Sliced-best_mean"] - df["MLP-half_mean"]).values * 100   # gain (pp)
    ye = np.sqrt(df["Sliced-best_std"]**2 + df["MLP-half_std"]**2).values * 100
    labels = df["label"].tolist()

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", color="#4e79a7",
                markersize=7, capsize=4,
                elinewidth=1.2, ecolor="gray", alpha=0.9)

    # Label each point
    for xi, yi, lbl in zip(x, y, labels):
        ax.annotate(lbl, (xi, yi),
                    textcoords="offset points", xytext=(6, 2),
                    fontsize=9)

    # Regression line
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min() - 1, x.max() + 1, 100)
    ax.plot(x_line, slope * x_line + intercept,
            color="#d62728", linewidth=1.5, linestyle="--",
            label=f"r = {r:.2f}  (p = {p:.3f})")

    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Mean shuffle drop (pp)", fontsize=11)
    ax.set_ylabel("Sliced-best − MLP-half (pp)", fontsize=11)
    ax.set_title("Shuffle drop predicts Sliced gain", fontsize=12)
    ax.legend(fontsize=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "scatter_drop_vs_gain.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 4 — j_cut stability heatmap
# ---------------------------------------------------------------------------
def fig_jcut_heatmap():
    jcut_cols = ["0", "3", "7", "11", "15", "20", "31"]
    datasets  = jcut["label"].tolist()

    data = jcut[jcut_cols].values.astype(float)

    fig, ax = plt.subplots(figsize=(7, 3.8))

    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=5, aspect="auto")

    # Axis labels
    ax.set_xticks(np.arange(len(jcut_cols)))
    ax.set_xticklabels([f"j={v}" for v in jcut_cols], fontsize=10)
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=11)
    ax.set_xlabel("j_cut value selected by val accuracy", fontsize=11)
    ax.set_title("j_cut selection frequency (count out of 5 seeds)", fontsize=12)

    # Annotate each cell
    for r in range(len(datasets)):
        for c in range(len(jcut_cols)):
            val = int(data[r, c])
            txt_color = "white" if val >= 4 else "black"
            ax.text(c, r, str(val), ha="center", va="center",
                    fontsize=11, color=txt_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Seed count", fontsize=10)
    cbar.set_ticks([0, 1, 2, 3, 4, 5])

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "jcut_stability_heatmap.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fig_main_results()
    fig_shuffle_drop()
    fig_scatter()
    fig_jcut_heatmap()
    print("Done — all figures saved to", FIG_DIR)
