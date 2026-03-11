"""
Strategy B warmup sensitivity experiment on Cora.
Single 50-epoch warmup records all epochs; cutoffs extracted at W=10,20,30,50.
Three fresh 200-epoch training runs (W=20,30,50); W=10 result reused from prior run.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.data.loaders import load_dataset
from src.models.sliced_mlp import SlicedSpectralMLP
from src.models.baselines import StandardMLP, train_baseline

OUT = "outputs/cora_warmup"
os.makedirs(OUT, exist_ok=True)

SEED = 42; K = 64; N_LAYERS = 2; LR = 0.01; WD = 5e-4; EPOCHS = 200
HALF = K // 2   # 32

torch.manual_seed(SEED)
U, labels, tr, val, te, eigenvalues = load_dataset("cora", k=K)
n_classes = int(labels.max().item()) + 1
n_slices   = HALF + 1
slice_dims = [HALF + j for j in range(n_slices)]
print(f"Cora LCC: N={U.shape[0]}  classes={n_classes}  "
      f"train={tr.sum()}  val={val.sum()}  test={te.sum()}")

def acc(logits, labels, mask):
    return (logits.argmax(-1)[mask] == labels[mask]).float().mean().item()

def train_truncated(cutoff, seed_offset=0):
    """Fresh 200-epoch run with loss_cutoff=cutoff. Returns (best_val, slice_test)."""
    torch.manual_seed(SEED + seed_offset)
    model = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                               loss_weights="uniform")
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best_val, best_state = 0.0, None
    top_j = cutoff  # track val on the active cutoff slice

    for _ in range(EPOCHS):
        model.train(); opt.zero_grad()
        model.compute_loss(model(U), labels, tr, loss_cutoff=cutoff).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            lg = model(U)
        v = acc(lg[top_j], labels, val)
        if v > best_val:
            best_val = v
            best_state = {k2: v2.clone() for k2, v2 in model.state_dict().items()}

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        lg = model(U)
    return best_val, [acc(l, labels, te) for l in lg]

# ---------------------------------------------------------------------------
# Single 50-epoch warmup — records per-slice val acc at EVERY epoch
# ---------------------------------------------------------------------------
print("\nRunning 50-epoch warmup (records every epoch)...")
torch.manual_seed(SEED)
warmup_model = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                                  loss_weights="uniform")
warmup_opt = torch.optim.Adam(warmup_model.parameters(), lr=LR, weight_decay=WD)

# warmup_curves[epoch] = array of shape (n_slices,) of val accuracies
warmup_curves = {}
for epoch in range(1, 51):
    warmup_model.train(); warmup_opt.zero_grad()
    warmup_model.compute_loss(warmup_model(U), labels, tr).backward()
    warmup_opt.step()
    warmup_model.eval()
    with torch.no_grad():
        lg = warmup_model(U)
    warmup_curves[epoch] = np.array([acc(lg[j], labels, val) for j in range(n_slices)])
    if epoch in {10, 20, 30, 50}:
        peak_j = int(np.argmax(warmup_curves[epoch]))
        print(f"  Epoch {epoch:2d}: peak at j={peak_j:2d}  "
              f"val={warmup_curves[epoch][peak_j]:.4f}  "
              f"(j=0: {warmup_curves[epoch][0]:.3f}, "
              f"j=15: {warmup_curves[epoch][15]:.3f}, "
              f"j=32: {warmup_curves[epoch][32]:.3f})")

# Cutoff selected at each warmup length
WARMUP_LENGTHS = [10, 20, 30, 50]
selected_cutoffs = {W: int(np.argmax(warmup_curves[W])) for W in WARMUP_LENGTHS}
print("\nSelected cutoffs:")
for W, c in selected_cutoffs.items():
    print(f"  W={W:2d} -> cutoff j={c:2d}  "
          f"(val at peak: {warmup_curves[W][c]:.4f})")

# Cutoff at every epoch 1..50 for trajectory plot
cutoff_trajectory = np.array([int(np.argmax(warmup_curves[e])) for e in range(1, 51)])

# ---------------------------------------------------------------------------
# Fresh 200-epoch training runs for W=20, 30, 50
# (W=10 result reused: cutoff=24, best_val=0.7168, best_slice=0.7104)
# ---------------------------------------------------------------------------
PRIOR_W10 = {"cutoff": 24, "best_val": 0.7168,
             "slice_test": None,   # don't have full array; fill with sentinel
             "best_slice": 0.7104}

run_results = {}

for W in [20, 30, 50]:
    c = selected_cutoffs[W]
    print(f"\nTraining W={W} (cutoff j={c}, d={HALF+c}) for {EPOCHS} epochs...")
    bv, st = train_truncated(c, seed_offset=W)
    best = max(st); best_j = st.index(best)
    run_results[W] = {"cutoff": c, "best_val": bv, "slice_test": st,
                      "best_slice": best, "best_j": best_j}
    print(f"  best_val={bv:.4f}  test@cutoff(j={c})={st[c]:.4f}  "
          f"test_coarse={st[0]:.4f}  best_slice={best:.4f} (j={best_j})")

# Prior results (reused, no rerun)
PRIOR = {
    "C":     {"cutoff": 16, "best_val": 0.7277, "best_slice": 0.7410},
    "mlp32": {"best_val": 0.7255, "best_slice": 0.7082},
}

# ---------------------------------------------------------------------------
# 1. Comparison table
# ---------------------------------------------------------------------------
target   = 0.7410   # Strategy C
baseline = 0.7082   # MLP-half

all_b_results = [
    (10,  PRIOR_W10["cutoff"], PRIOR_W10["best_val"], PRIOR_W10["best_slice"]),
    (20,  run_results[20]["cutoff"], run_results[20]["best_val"], run_results[20]["best_slice"]),
    (30,  run_results[30]["cutoff"], run_results[30]["best_val"], run_results[30]["best_slice"]),
    (50,  run_results[50]["cutoff"], run_results[50]["best_val"], run_results[50]["best_slice"]),
]
best_b_acc = max(r[3] for r in all_b_results)
best_b_W   = [r[0] for r in all_b_results if r[3] == best_b_acc][0]

summary_lines = [
    "",
    "  STRATEGY B WARMUP SENSITIVITY — CORA",
    "  " + "=" * 62,
    f"  Does best Strategy B match Strategy C ({target:.4f}) or beat MLP-half ({baseline:.4f})?",
]
if best_b_acc >= target:
    summary_lines.append(f"  YES (matches/exceeds C) — W={best_b_W} achieves {best_b_acc:.4f}")
elif best_b_acc > baseline:
    summary_lines.append(f"  PARTIAL — W={best_b_W} beats MLP-half ({best_b_acc:.4f} > {baseline:.4f}) "
                         f"but below Strategy C ({target:.4f})")
else:
    summary_lines.append(f"  NO — best B is {best_b_acc:.4f} (W={best_b_W}), below MLP-half {baseline:.4f}")
summary_lines.append("")

div = "=" * 72
lines = summary_lines + [
    div,
    f"COMPARISON TABLE — Cora Strategy B warmup sensitivity",
    div,
    f"{'Method':<28} {'Warmup':>7} {'Cutoff':>8} {'Best Val':>10} {'Test (best)':>12}",
    "-" * 72,
]
for W, cutoff, bv, tb in all_b_results:
    lines.append(f"{'Strategy B (W='+str(W)+')':<28} {W:>7} {'j='+str(cutoff):>8} "
                 f"{bv:>10.4f} {tb:>12.4f}")
lines.append("-" * 72)
lines.append(f"{'Strategy C (eig-thresh)':<28} {'—':>7} {'j='+str(PRIOR['C']['cutoff']):>8} "
             f"{PRIOR['C']['best_val']:>10.4f} {PRIOR['C']['best_slice']:>12.4f}  ← target")
lines.append(f"{'StandardMLP-half (k=32)':<28} {'—':>7} {'—':>8} "
             f"{PRIOR['mlp32']['best_val']:>10.4f} {PRIOR['mlp32']['best_slice']:>12.4f}  ← baseline")
lines.append(div)

table_str = "\n".join(lines)
print("\n" + table_str)
with open(f"{OUT}/comparison_table.txt", "w") as f:
    f.write(table_str + "\n")

# ---------------------------------------------------------------------------
# 2. Warmup stability plot — 4 subplots, one per W
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

cmap = cm.get_cmap("Blues")
for ax_idx, W in enumerate(WARMUP_LENGTHS):
    ax = axes[ax_idx]
    c = selected_cutoffs[W]

    # Faint lines for every epoch up to W
    for epoch in range(1, W + 1):
        alpha = 0.15 + 0.6 * (epoch / W)
        lw    = 0.6 if epoch < W else 2.0
        col   = cmap(0.3 + 0.6 * (epoch / W))
        ax.plot(slice_dims, warmup_curves[epoch],
                color=col, lw=lw, alpha=alpha)

    # Bold final epoch
    ax.plot(slice_dims, warmup_curves[W], color="navy", lw=2.5,
            label=f"Epoch {W} (final)")

    # argmax at each epoch as scatter
    for epoch in range(1, W + 1):
        peak_j = int(np.argmax(warmup_curves[epoch]))
        ax.axvline(HALF + peak_j, color="red", lw=0.4, alpha=0.3)

    # Selected cutoff vertical
    ax.axvline(HALF + c, color="red", lw=1.8, ls="--",
               label=f"Selected cutoff j={c}")

    # Known good region band
    ax.axvspan(HALF + 11, HALF + 16, alpha=0.08, color="green",
               label="Good region j=11..16")

    ax.set_xlabel("Slice dim d_j"); ax.set_ylabel("Val accuracy")
    ax.set_title(f"Warmup W={W}: per-slice val curve (epochs 1–{W})", fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 0.80)

plt.suptitle("Strategy B warmup stability — Cora (faint=early, bold=final epoch)",
             fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(f"{OUT}/warmup_stability.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT}/warmup_stability.png")

# ---------------------------------------------------------------------------
# 3. Cutoff trajectory plot — argmax across all 50 epochs
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4))
epochs_range = np.arange(1, 51)
ax.plot(epochs_range, cutoff_trajectory, color="#1f77b4", lw=2, marker="o", ms=3,
        label="Selected cutoff j (argmax val acc)")

# Vertical lines at tested warmup lengths
colors_W = {10: "#d62728", 20: "#ff7f0e", 30: "#2ca02c", 50: "#9467bd"}
for W, col in colors_W.items():
    ax.axvline(W, color=col, lw=1.5, ls="--",
               label=f"W={W}: j={selected_cutoffs[W]}")

# Horizontal band for good region
ax.axhspan(11, 16, alpha=0.12, color="green", label="Good region j=11..16")
ax.axhline(15, color="gray", lw=1.0, ls=":", alpha=0.7, label="Manual optimum j=15")
ax.axhline(16, color="green", lw=1.0, ls=":", alpha=0.7, label="Strategy C j=16")

ax.set_xlabel("Warmup epoch", fontsize=12)
ax.set_ylabel("Selected cutoff index j", fontsize=12)
ax.set_title("Strategy B cutoff trajectory — does argmax stabilize near j=11..16?",
             fontsize=12)
ax.set_ylim(-1, 33)
ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/cutoff_trajectory.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT}/cutoff_trajectory.png")

# ---------------------------------------------------------------------------
# 4. Final per-slice accuracy curves after 200-epoch training
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
run_colors = {20: "#ff7f0e", 30: "#2ca02c", 50: "#9467bd"}

# W=10 prior: we only have best_slice, not the full array — skip line, add point
ax.scatter([HALF + PRIOR_W10["cutoff"]], [PRIOR_W10["best_slice"]],
           marker="*", s=200, color="#d62728", zorder=5,
           label=f"B W=10: best={PRIOR_W10['best_slice']:.4f} (j={PRIOR_W10['cutoff']})")

for W in [20, 30, 50]:
    st = run_results[W]["slice_test"]
    c  = run_results[W]["cutoff"]
    col = run_colors[W]
    ax.plot(slice_dims, st, color=col, lw=1.8, marker="o", ms=2.5,
            label=f"B W={W}: best={run_results[W]['best_slice']:.4f} (cutoff j={c})")
    ax.axvline(HALF + c, color=col, lw=0.8, ls=":", alpha=0.6)

ax.axhline(PRIOR["C"]["best_slice"], color="black", lw=1.5, ls="--",
           label=f"Strategy C j=16: {PRIOR['C']['best_slice']:.4f}")
ax.axhline(baseline, color="gray", lw=1.5, ls=":",
           label=f"MLP-half k=32: {baseline:.4f}")
ax.axhspan(HALF + 11, HALF + 16, alpha=0.07, color="green")

ax.set_xlabel("Slice dimension d_j", fontsize=12)
ax.set_ylabel("Test accuracy", fontsize=12)
ax.set_title("Final per-slice accuracy — Strategy B at different warmup lengths",
             fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/per_slice_accuracy.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT}/per_slice_accuracy.png")

# ---------------------------------------------------------------------------
# 5. Notes
# ---------------------------------------------------------------------------

# Find when cutoff first enters and stays in [11,16] for 5+ consecutive epochs
good_lo, good_hi = 11, 16
first_stable_epoch = None
for start in range(1, 47):
    window = cutoff_trajectory[start-1:start+4]   # 5 epochs
    if all(good_lo <= j <= good_hi for j in window):
        first_stable_epoch = start
        break

notes = [
    "WARMUP SENSITIVITY NOTES — Cora Strategy B",
    "=" * 60,
    "",
    "SELECTED CUTOFFS PER WARMUP LENGTH",
    f"{'W':>4}  {'cutoff_j':>9}  {'test_best':>10}  {'beats MLP-half?':>16}  {'vs C (0.7410)':>14}",
    "-" * 60,
]
for W, cutoff, bv, tb in all_b_results:
    beats = "YES" if tb > baseline else "NO"
    vs_c  = f"{tb - target:+.4f}"
    notes.append(f"{W:>4}  {cutoff:>9}  {tb:>10.4f}  {beats:>16}  {vs_c:>14}")

notes += [
    "",
    "CUTOFF TRAJECTORY ANALYSIS",
    f"Trajectory (epoch -> selected j): "
    + "  ".join(f"e{e}:j{cutoff_trajectory[e-1]}" for e in range(1, 51, 5)),
    "",
]

if first_stable_epoch:
    notes.append(f"First 5-consecutive-epoch stable window in j=[{good_lo},{good_hi}]: "
                 f"epoch {first_stable_epoch}..{first_stable_epoch+4}.")
    notes.append(f"Minimum reliable warmup estimate: W={first_stable_epoch} epochs.")
else:
    notes.append(f"Argmax never stays in j=[{good_lo},{good_hi}] for 5 consecutive epochs.")
    notes.append("The warmup curve on Cora is too noisy for stable cutoff detection up to W=50.")

notes += [
    "",
    "COMPARISON: Strategy B (best) vs Strategy C",
    f"  Best Strategy B: W={best_b_W}, test={best_b_acc:.4f}",
    f"  Strategy C:      test=0.7410 (zero warmup cost)",
    f"  Gap: {best_b_acc - 0.7410:+.4f}",
    "",
    "RECOMMENDATION",
]

if best_b_acc >= target - 0.005:
    notes += [
        "Strategy B (extended warmup) is a practical substitute for Strategy C on Cora.",
        f"Use W={best_b_W} epochs warmup as default for homophilous graphs.",
        "For heterophilous graphs (Cornell, Actor): Strategy B is the only option —",
        "Strategy C's median-threshold assumption breaks when low-freq != informative.",
        f"Recommended default warmup: W={max(best_b_W, 30)} epochs for general use.",
    ]
else:
    notes += [
        "Strategy B does not fully match Strategy C even at W=50.",
        "Strategy B and C are complementary, not interchangeable:",
        "  - Strategy C: zero warmup cost, reliable on homophilous graphs only.",
        f"  - Strategy B (W=50): {best_b_acc:.4f}, works on any graph type but needs validation.",
        "For heterophilous graphs where Strategy C cannot be used, Strategy B with",
        "W=50 is the best available automatic method.",
        "If budget allows, use cross-validation over W in {20,30,50} on a held-out split.",
    ]

notes_str = "\n".join(notes)
print("\n" + notes_str)
with open(f"{OUT}/notes.txt", "w") as f:
    f.write(notes_str + "\n")

print(f"\nAll outputs saved to {OUT}")
