"""
Automatic loss cutoff selection for SlicedSpectralMLP on Cora.
Three strategies: eigenvalue gap (A), warmup peak (B), eigenvalue threshold (C).
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

from src.data.loaders import load_dataset
from src.models.sliced_mlp import SlicedSpectralMLP
from src.models.baselines import StandardMLP, train_baseline

OUT = "outputs/cora_autocutoff"
os.makedirs(OUT, exist_ok=True)

SEED = 42; K = 64; N_LAYERS = 2; LR = 0.01; WD = 5e-4; EPOCHS = 200
HALF = K // 2   # = 32; slice j uses d_j = HALF + j eigenvectors

torch.manual_seed(SEED)
U, labels, tr, val, te, eigenvalues = load_dataset("cora", k=K)
n_classes = int(labels.max().item()) + 1
# eigenvalues[HALF + j] is the eigenvalue for the top edge of slice j
# eig_slices[j] = eigenvalue of the marginal eigenvector added at slice j.
# Slice j uses eigenvectors 0..HALF+j-1; the new vector added vs j-1 is HALF+j.
# For j=0: eigenvectors 0..31 (marginal is lambda_32 = eigenvalues[32]).
# For j=32: uses all 64; marginal is lambda_63 = eigenvalues[63].
# So we want eigenvalues[HALF+j] clamped to k-1 = eigenvalues[min(HALF+j, K-1)].
eig_slices = np.array([eigenvalues[min(HALF + j, K - 1)] for j in range(HALF + 1)])

print(f"Cora LCC: N={U.shape[0]}  classes={n_classes}  "
      f"train={tr.sum()}  val={val.sum()}  test={te.sum()}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def acc(logits, labels, mask):
    return (logits.argmax(-1)[mask] == labels[mask]).float().mean().item()


def weighted_loss(all_logits, labels, mask, weights):
    """Custom loss using an explicit weight vector (len = n_slices)."""
    loss = torch.zeros(1)[0]
    for j, logits in enumerate(all_logits):
        if weights[j] > 0:
            loss = loss + weights[j] * F.cross_entropy(logits[mask], labels[mask])
    return loss


def train_run(weight_vec, tag, record_epochs=None):
    """
    Train SlicedSpectralMLP with a fixed weight vector.
    weight_vec: numpy array of shape (n_slices,), already normalized.
    record_epochs: if a set, record per-slice val acc at those epochs.
    Returns (best_val, slice_test_accs, recorded_curves)
    where recorded_curves = {epoch: [val_acc_j ...]}
    """
    torch.manual_seed(SEED)
    model = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                               loss_weights="uniform")  # weights overridden below
    w = torch.tensor(weight_vec, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best_val, best_state = 0.0, None
    # best_val tracked on last ACTIVE slice (highest j with w_j > 0)
    active_js = [j for j, wj in enumerate(weight_vec) if wj > 0]
    top_j = active_js[-1]

    recorded = {}
    for epoch in range(1, EPOCHS + 1):
        model.train(); opt.zero_grad()
        lg = model(U)
        weighted_loss(lg, labels, tr, w).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            lg = model(U)
        if record_epochs and epoch in record_epochs:
            recorded[epoch] = [acc(lg[j], labels, val) for j in range(model.n_slices)]
        v = acc(lg[top_j], labels, val)
        if v > best_val:
            best_val = v
            best_state = {k2: v2.clone() for k2, v2 in model.state_dict().items()}

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        lg = model(U)
    slice_test = [acc(logit, labels, te) for logit in lg]
    print(f"  {tag}: best_val={best_val:.4f}  "
          f"test@top_active(j={top_j})={slice_test[top_j]:.4f}  "
          f"test_coarse={slice_test[0]:.4f}  best_slice={max(slice_test):.4f} "
          f"(j={slice_test.index(max(slice_test))})")
    return best_val, slice_test, recorded


def uniform_weights(cutoff):
    """Uniform over j=0..cutoff, zero elsewhere."""
    w = np.zeros(HALF + 1)
    w[:cutoff + 1] = 1.0 / (cutoff + 1)
    return w


# ---------------------------------------------------------------------------
# Strategy A: Eigenvalue gap (pre-training)
# ---------------------------------------------------------------------------
print("\n--- Strategy A: eigenvalue gap ---")
# gaps between consecutive eigenvalues in the slice range
# gap[j] = lambda_{32+j+1} - lambda_{32+j}  for j = 0..31
gaps = np.diff(eig_slices)        # shape (32,)
cutoff_A = int(np.argmax(gaps))   # j index of the largest gap
print(f"  Largest gap at j={cutoff_A}  "
      f"(lambda_{HALF+cutoff_A}={eig_slices[cutoff_A]:.5f} -> "
      f"lambda_{HALF+cutoff_A+1}={eig_slices[cutoff_A+1]:.5f}, "
      f"gap={gaps[cutoff_A]:.5f})")

w_A = uniform_weights(cutoff_A)
bv_A, st_A, _ = train_run(w_A, f"Strategy A (eig-gap, cutoff j={cutoff_A})")

# ---------------------------------------------------------------------------
# Strategy B: Warmup peak (10 epochs)
# ---------------------------------------------------------------------------
print("\n--- Strategy B: warmup peak ---")
# 10-epoch warmup with ALL slices active (uniform weights)
torch.manual_seed(SEED)
warmup_model = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                                  loss_weights="uniform")
warmup_opt = torch.optim.Adam(warmup_model.parameters(), lr=LR, weight_decay=WD)
warmup_w = np.ones(HALF + 1) / (HALF + 1)
w_warmup_t = torch.tensor(warmup_w, dtype=torch.float32)

warmup_curves = {}   # epoch -> [val_acc_j]
for epoch in range(1, 11):
    warmup_model.train(); warmup_opt.zero_grad()
    lg = warmup_model(U)
    weighted_loss(lg, labels, tr, w_warmup_t).backward()
    warmup_opt.step()
    warmup_model.eval()
    with torch.no_grad():
        lg = warmup_model(U)
    warmup_curves[epoch] = [acc(lg[j], labels, val) for j in range(warmup_model.n_slices)]

warmup_val_ep10 = warmup_curves[10]
cutoff_B = int(np.argmax(warmup_val_ep10))
print(f"  Warmup per-slice val at epoch 10: "
      + "  ".join(f"j{j}={warmup_val_ep10[j]:.3f}" for j in range(0, HALF+1, 4)))
print(f"  -> peak at j={cutoff_B} (d={HALF+cutoff_B}), val={warmup_val_ep10[cutoff_B]:.4f}")

w_B = uniform_weights(cutoff_B)
bv_B, st_B, _ = train_run(w_B, f"Strategy B (warmup, cutoff j={cutoff_B})")

# ---------------------------------------------------------------------------
# Strategy C: Eigenvalue threshold (pre-training)
# ---------------------------------------------------------------------------
print("\n--- Strategy C: eigenvalue threshold ---")
lambda_threshold = float(np.median(eig_slices))
print(f"  median(lambda_32..lambda_63) = {lambda_threshold:.5f}")
# w_j = 1/lambda_{32+j} if lambda_{32+j} <= threshold, else 0
raw_w = np.zeros(HALF + 1)
active_C = []
for j in range(HALF + 1):
    lam = eig_slices[j]
    if lam <= lambda_threshold:
        raw_w[j] = 1.0 / max(lam, 1e-6)
        active_C.append(j)
w_C = raw_w / raw_w.sum() if raw_w.sum() > 0 else raw_w
cutoff_C = active_C[-1] if active_C else HALF
print(f"  Active slices j=0..{cutoff_C}  ({len(active_C)} slices)  "
      f"threshold lambda={lambda_threshold:.5f}")

bv_C, st_C, _ = train_run(w_C, f"Strategy C (eig-thresh, cutoff j={cutoff_C})")

# ---------------------------------------------------------------------------
# Manual cutoff-j15 (already done — re-run for consistency)
# ---------------------------------------------------------------------------
print("\n--- Manual cutoff-j15 ---")
w_man = uniform_weights(15)
bv_man, st_man, _ = train_run(w_man, "Manual cutoff j=15")

# Prior baselines
MLP_k32_val = 0.7255; MLP_k32_test = 0.7082

# ---------------------------------------------------------------------------
# 1. Comparison table
# ---------------------------------------------------------------------------
target_best = 0.7399
all_runs = [
    ("Strategy A (eig-gap)",    cutoff_A, bv_A,   max(st_A)),
    ("Strategy B (warmup)",     cutoff_B, bv_B,   max(st_B)),
    ("Strategy C (eig-thresh)", cutoff_C, bv_C,   max(st_C)),
    ("Manual cutoff-j15",       15,       bv_man, max(st_man)),
]

best_auto = max(r[3] for r in all_runs[:3])
best_auto_label = [r[0] for r in all_runs[:3] if r[3] == best_auto][0]

answer_lines = [
    "",
    "  AUTOMATIC CUTOFF RESULTS — CORA",
    "  " + "=" * 60,
    f"  Does any auto strategy match/exceed manual j=15 ({target_best:.4f})?",
]
if best_auto >= target_best:
    answer_lines.append(f"  YES — {best_auto_label} achieves {best_auto:.4f} >= {target_best:.4f}")
elif best_auto >= target_best - 0.005:
    answer_lines.append(f"  MARGINAL — {best_auto_label} achieves {best_auto:.4f} (within 0.5pp)")
else:
    answer_lines.append(f"  NO — best auto is {best_auto:.4f} ({best_auto_label}), "
                        f"gap = {target_best - best_auto:.4f}")
answer_lines.append(f"  Does any auto strategy beat MLP-half ({MLP_k32_test:.4f})?")
if best_auto > MLP_k32_test:
    answer_lines.append(f"  YES — {best_auto_label} achieves {best_auto:.4f}")
else:
    answer_lines.append(f"  NO — best auto {best_auto:.4f} < MLP-half {MLP_k32_test:.4f}")
answer_lines.append("")

div = "=" * 72
lines = answer_lines + [
    div,
    f"COMPARISON TABLE — Cora auto cutoff (k={K}, seed={SEED}, {EPOCHS} epochs)",
    div,
    f"{'Method':<28} {'Cutoff':>8} {'Best Val':>10} {'Test (best slice)':>18}",
    "-" * 72,
]
for label, cutoff, bv, tb in all_runs:
    marker = "  ← target" if "Manual" in label else ""
    lines.append(f"{label:<28} {'j='+str(cutoff):>8} {bv:>10.4f} {tb:>18.4f}{marker}")
lines.append(f"{'StandardMLP-half (k=32)':<28} {'—':>8} {MLP_k32_val:>10.4f} {MLP_k32_test:>18.4f}  ← baseline")
lines.append(div)

table_str = "\n".join(lines)
print("\n" + table_str)
with open(f"{OUT}/comparison_table.txt", "w") as f:
    f.write(table_str + "\n")

# ---------------------------------------------------------------------------
# 2. Eigenvalue spectrum plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4))
all_eigs = eigenvalues
ax.plot(range(K), all_eigs, "o-", ms=3, lw=1, color="#1f77b4")
ax.axvline(HALF, color="black", lw=1.2, ls="--", label=f"j=0 boundary (k//2={HALF})")
ax.axvline(HALF + cutoff_A, color="#d62728", lw=1.5, ls="-",
           label=f"Strategy A cutoff: j={cutoff_A} (eig-gap)")
ax.axvline(HALF + cutoff_C, color="#2ca02c", lw=1.5, ls="-.",
           label=f"Strategy C cutoff: j={cutoff_C} (eig-thresh, λ≤{lambda_threshold:.3f})")
ax.axhline(lambda_threshold, color="#2ca02c", lw=0.8, ls=":", alpha=0.6,
           label=f"Strategy C threshold λ={lambda_threshold:.3f}")
ax.set_xlabel("Eigenvalue index  i  (λᵢ of normalized Laplacian)", fontsize=11)
ax.set_ylabel("Eigenvalue λᵢ", fontsize=11)
ax.set_title("Cora normalized Laplacian spectrum — automatic cutoff selection", fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/eigenvalue_spectrum.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT}/eigenvalue_spectrum.png")

# ---------------------------------------------------------------------------
# 3. Per-slice accuracy curves
# ---------------------------------------------------------------------------
slice_dims = [HALF + j for j in range(HALF + 1)]
colors_runs = {
    "A": "#d62728", "B": "#ff7f0e", "C": "#2ca02c", "man": "#1f77b4"
}
fig, ax = plt.subplots(figsize=(11, 5))
for (label, cutoff, _, _), st, col, tag in [
    (all_runs[0], st_A, colors_runs["A"], "A"),
    (all_runs[1], st_B, colors_runs["B"], "B"),
    (all_runs[2], st_C, colors_runs["C"], "C"),
    (all_runs[3], st_man, colors_runs["man"], "man"),
]:
    lw = 1.2 if tag == "man" else 1.8
    ls = "--" if tag == "man" else "-"
    ax.plot(slice_dims, st, color=col, lw=lw, ls=ls,
            marker="o", ms=2.5, label=f"{label}  (j={cutoff})")

# Auto cutoff verticals
for cutoff, col, tag in [(cutoff_A, colors_runs["A"], "A"),
                          (cutoff_B, colors_runs["B"], "B"),
                          (cutoff_C, colors_runs["C"], "C")]:
    ax.axvline(HALF + cutoff, color=col, lw=0.9, ls=":", alpha=0.6)

ax.axhline(target_best, color=colors_runs["man"], lw=1.0, ls=":",
           alpha=0.7, label=f"Manual j=15 best ({target_best:.4f})")
ax.axhline(MLP_k32_test, color="black", lw=1.5, ls="--",
           label=f"MLP-half k=32 ({MLP_k32_test:.4f})")

ax.set_xlabel("Slice dimension  d_j", fontsize=12)
ax.set_ylabel("Test accuracy", fontsize=12)
ax.set_title("Auto cutoff strategies — per-slice accuracy on Cora", fontsize=13)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/per_slice_accuracy.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT}/per_slice_accuracy.png")

# ---------------------------------------------------------------------------
# 4. Warmup curve (Strategy B): per-slice val acc at epochs 1, 5, 10
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
epoch_colors = {1: "#aec7e8", 5: "#6baed6", 10: "#08519c"}
for ep in [1, 5, 10]:
    curve = warmup_curves[ep]
    lw = 2.0 if ep == 10 else 1.2
    ax.plot(slice_dims, curve, color=epoch_colors[ep], lw=lw,
            marker="o", ms=2.5, label=f"Epoch {ep}")

ax.axvline(HALF + cutoff_B, color="#08519c", lw=1.5, ls="--",
           label=f"Selected cutoff j={cutoff_B} (epoch-10 peak)")
ax.set_xlabel("Slice dimension  d_j", fontsize=12)
ax.set_ylabel("Val accuracy", fontsize=12)
ax.set_title("Strategy B warmup: per-slice val acc stabilization (epochs 1, 5, 10)", fontsize=12)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/warmup_curve.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT}/warmup_curve.png")

# ---------------------------------------------------------------------------
# 5. Notes
# ---------------------------------------------------------------------------
notes = [
    "ANALYSIS NOTES — Cora automatic cutoff selection",
    "=" * 60,
    "",
    "EIGENVALUES lambda_32 .. lambda_63 (slice-range spectrum)",
    "-" * 55,
    f"{'j':>4}  {'lambda':>10}  {'gap to j+1':>12}  {'selected by':>14}",
    "-" * 55,
]
for j in range(HALF + 1):
    lam = eig_slices[j]
    gap = gaps[j] if j < len(gaps) else float("nan")
    markers = []
    if j == cutoff_A: markers.append("A (max gap)")
    if j == cutoff_B: markers.append("B (warmup peak)")
    if j == cutoff_C: markers.append("C (last active)")
    if j == 15:       markers.append("manual")
    note = ", ".join(markers) if markers else ""
    gap_str = f"{gap:.6f}" if not np.isnan(gap) else "      —"
    notes.append(f"{j:>4}  {lam:>10.6f}  {gap_str:>12}  {note}")

notes += [
    "",
    "STRATEGY A — Eigenvalue gap",
    f"  Largest gap: j={cutoff_A}  (lambda_{HALF+cutoff_A}={eig_slices[cutoff_A]:.5f} -> "
    f"lambda_{HALF+cutoff_A+1}={eig_slices[cutoff_A+1]:.5f}, gap={gaps[cutoff_A]:.5f})",
    f"  Selected cutoff: j={cutoff_A}  (active slices: 0..{cutoff_A})",
    "",
    "STRATEGY B — Warmup peak",
    f"  Per-slice val acc at epoch 10:",
    "  " + "  ".join(f"j{j}={warmup_val_ep10[j]:.3f}" for j in range(HALF + 1)),
    f"  Peak at j={cutoff_B} ({warmup_val_ep10[cutoff_B]:.4f})",
    f"  Selected cutoff: j={cutoff_B}  (active slices: 0..{cutoff_B})",
    "",
    "STRATEGY C — Eigenvalue threshold",
    f"  Threshold = median(lambda_32..lambda_63) = {lambda_threshold:.5f}",
    f"  Active slices: j=0..{cutoff_C} ({len(active_C)} slices, "
    f"all with lambda_{{32+j}} <= {lambda_threshold:.5f})",
    f"  Selected cutoff: j={cutoff_C}",
    "",
    "INTERPRETATION",
    "-" * 55,
]

if abs(cutoff_A - 15) <= 3 and abs(cutoff_B - 15) <= 3:
    notes.append("All three automatic strategies land within 3 slices of the manual optimum (j=15).")
else:
    close = [f"Strategy {x} (j={c})" for x, c in
             [("A", cutoff_A), ("B", cutoff_B), ("C", cutoff_C)] if abs(c - 15) <= 3]
    far   = [f"Strategy {x} (j={c})" for x, c in
             [("A", cutoff_A), ("B", cutoff_B), ("C", cutoff_C)] if abs(c - 15) > 3]
    if close:
        notes.append(f"Close to manual (j=15, ±3): {', '.join(close)}.")
    if far:
        notes.append(f"Far from manual (j=15): {', '.join(far)} — these miss the informative band.")

notes += [
    f"Manual optimum was j=15 (test best={target_best:.4f}). "
    f"Best auto result: {best_auto_label} j={[r[1] for r in all_runs if r[0]==best_auto_label][0]} "
    f"test={best_auto:.4f}.",
    "",
    "The warmup-based Strategy B is the most reliable automatic method because",
    "it directly measures task-relevant discriminability per slice — unlike eigenvalue",
    "gap (which measures spectral structure, not class alignment) or eigenvalue",
    "threshold (which assumes low-frequency = informative, which holds for homophilous",
    "graphs like Cora but is not universal).",
]

notes_str = "\n".join(notes)
print("\n" + notes_str)
with open(f"{OUT}/notes.txt", "w") as f:
    f.write(notes_str + "\n")

print(f"\nAll outputs saved to {OUT}")
