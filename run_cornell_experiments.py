"""
Cornell experiment pipeline.

Runs all 6 models across all 10 fixed splits, generates all diagnostic outputs
under outputs/cornell/.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_dataset
from model import SlicedSpectralMLP
from baselines import StandardMLP, train_baseline

OUT = "outputs/cornell"
os.makedirs(f"{OUT}/grad_heatmaps", exist_ok=True)

SEED = 42
K = 64
N_LAYERS = 2
LR = 0.01
WD = 5e-4
EPOCHS = 200
N_SPLITS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def acc(logits, labels, mask):
    return (logits.argmax(-1)[mask] == labels[mask]).float().mean().item()


def save_grad_heatmap(grad, epoch, out_dir):
    g = grad.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(g, cmap="hot", aspect="auto",
                   vmin=0, vmax=g.max() if g.max() > 0 else 1.0)
    ax.set_title(f"|W[0].grad|  epoch {epoch}")
    ax.set_xlabel("column j"); ax.set_ylabel("row i")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"epoch_{epoch:04d}.png"), dpi=100)
    plt.close(fig)


def train_sliced(model, U, labels, tr, val, te, eigenvalues=None,
                 heatmap_epochs=None, heatmap_dir=None):
    """Train SlicedSpectralMLP; return (best_val, per-slice test accs,
    val_coarse_curve, val_full_curve)."""
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best_val, best_state = 0.0, None
    val_coarse, val_full = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train(); opt.zero_grad()
        logits = model(U)
        loss = model.compute_loss(logits, labels, tr)
        loss.backward()

        if heatmap_epochs and epoch in heatmap_epochs and model.W[0].grad is not None:
            save_grad_heatmap(torch.abs(model.W[0].grad), epoch, heatmap_dir)

        opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(U)
        vc = acc(logits[0],  labels, val)
        vf = acc(logits[-1], labels, val)
        val_coarse.append(vc); val_full.append(vf)
        if vf > best_val:
            best_val = vf
            best_state = {k2: v.clone() for k2, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(U)
    slice_test = [acc(lg, labels, te) for lg in logits]
    return best_val, slice_test, val_coarse, val_full


# ---------------------------------------------------------------------------
# Load Cornell — precompute U and eigenvalues once (LCC, k=64)
# They don't change across splits; only the masks change.
# ---------------------------------------------------------------------------

print("Loading Cornell LCC eigenvectors (split 0 for U/eigs, masks vary)...")
# Load once to get U and eigenvalues (they're split-independent)
U, labels0, _, _, _, eigenvalues = load_dataset("cornell", k=K, split_idx=0)
N = U.shape[0]
n_classes = int(labels0.max().item()) + 1
print(f"  LCC nodes={N}  classes={n_classes}  k={K}")

# Load all 10 splits' masks
splits = []
for s in range(N_SPLITS):
    _, labels_s, tr_s, val_s, te_s, _ = load_dataset("cornell", k=K, split_idx=s)
    splits.append((tr_s, val_s, te_s))
    assert (labels_s == labels0).all(), "Labels differ across splits — unexpected"
labels = labels0
print(f"  Loaded {N_SPLITS} splits.")


# ---------------------------------------------------------------------------
# Run all models over all 10 splits
# ---------------------------------------------------------------------------

STRATEGIES = ["uniform", "coarse", "eigenvalue"]
results = {s: {"slice_test": [], "best_val": []} for s in STRATEGIES}
results["mlp_full"] = {"best_val": [], "test": []}
results["mlp_half"] = {"best_val": [], "test": []}

# For training curves and heatmaps: use split 0, track curves separately
curves_uniform = None   # (val_coarse, val_full)

for split_idx, (tr, val, te) in enumerate(splits):
    torch.manual_seed(SEED + split_idx)
    print(f"\n=== Split {split_idx} | train={tr.sum()} val={val.sum()} test={te.sum()} ===")

    for strategy in STRATEGIES:
        torch.manual_seed(SEED + split_idx)
        model = SlicedSpectralMLP(
            k=K, n_classes=n_classes, n_layers=N_LAYERS,
            loss_weights=strategy,
            eigenvalues=eigenvalues if strategy == "eigenvalue" else None,
        )
        heatmap_epochs = {10, 30, 50} if (strategy == "uniform" and split_idx == 0) else None
        heatmap_dir    = f"{OUT}/grad_heatmaps" if heatmap_epochs else None

        bv, slice_test, vc, vf = train_sliced(
            model, U, labels, tr, val, te,
            heatmap_epochs=heatmap_epochs, heatmap_dir=heatmap_dir,
        )
        results[strategy]["slice_test"].append(slice_test)
        results[strategy]["best_val"].append(bv)
        print(f"  Sliced({strategy}): val={bv:.4f}  test_full={slice_test[-1]:.4f}  test_coarse={slice_test[0]:.4f}")

        if strategy == "uniform" and split_idx == 0:
            curves_uniform = (vc, vf)

    # Baselines
    for tag, n_feat in [("mlp_full", K), ("mlp_half", K // 2)]:
        torch.manual_seed(SEED + split_idx)
        X = U[:, :n_feat]
        bl = StandardMLP(n_features=n_feat, n_classes=n_classes,
                         hidden_dim=K, n_layers=N_LAYERS)
        bv, ta = train_baseline(bl, X, labels, tr, val, te, lr=LR, wd=WD, epochs=EPOCHS)
        results[tag]["best_val"].append(bv)
        results[tag]["test"].append(ta)
        print(f"  {tag}: val={bv:.4f}  test={ta:.4f}")


# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------

def mean_std(arr):
    a = np.array(arr)
    return a.mean(), a.std()

# Per-slice mean test acc (averaged over splits)
slice_mean = {}
slice_std  = {}
for s in STRATEGIES:
    arr = np.array(results[s]["slice_test"])  # (N_SPLITS, n_slices)
    slice_mean[s] = arr.mean(axis=0)
    slice_std[s]  = arr.std(axis=0)

n_slices = K // 2 + 1
slice_dims = [K // 2 + j for j in range(n_slices)]


# ---------------------------------------------------------------------------
# 1. Comparison table
# ---------------------------------------------------------------------------

def row(name, bv_m, bv_s, tf_m, tf_s, tc_m, tc_s, tb_m, tb_s):
    return (f"{name:<28} {bv_m:.4f}±{bv_s:.4f}  {tf_m:.4f}±{tf_s:.4f}  "
            f"{tc_m:.4f}±{tc_s:.4f}  {tb_m:.4f}±{tb_s:.4f}")

lines = []
div = "=" * 100
lines.append(div)
lines.append(f"COMPARISON TABLE — Cornell  (k={K}, averaged over {N_SPLITS} splits)")
lines.append(div)
lines.append(f"{'Method':<28} {'Best Val':>14}  {'Test (full)':>14}  {'Test (coarse)':>15}  {'Test (best slice)':>18}")
lines.append("-" * 100)

for s in STRATEGIES:
    bv_arr  = np.array(results[s]["best_val"])
    st_arr  = np.array(results[s]["slice_test"])
    tf_arr  = st_arr[:, -1]
    tc_arr  = st_arr[:, 0]
    tb_arr  = st_arr.max(axis=1)
    lines.append(row(f"Sliced({s})",
                     bv_arr.mean(), bv_arr.std(),
                     tf_arr.mean(), tf_arr.std(),
                     tc_arr.mean(), tc_arr.std(),
                     tb_arr.mean(), tb_arr.std()))

for tag in ["mlp_full", "mlp_half"]:
    bv_arr = np.array(results[tag]["best_val"])
    ta_arr = np.array(results[tag]["test"])
    name   = "StandardMLP-full" if tag == "mlp_full" else "StandardMLP-half"
    lines.append(
        f"{name:<28} {bv_arr.mean():.4f}±{bv_arr.std():.4f}  "
        f"{ta_arr.mean():.4f}±{ta_arr.std():.4f}  "
        f"{ta_arr.mean():.4f}±{ta_arr.std():.4f}  "
        f"{ta_arr.mean():.4f}±{ta_arr.std():.4f}"
    )

lines.append(div)
table_str = "\n".join(lines)
print("\n" + table_str)
with open(f"{OUT}/comparison_table.txt", "w") as f:
    f.write(table_str + "\n")
print(f"Saved {OUT}/comparison_table.txt")


# ---------------------------------------------------------------------------
# 2. Per-slice accuracy curve
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for i, s in enumerate(STRATEGIES):
    ax.plot(slice_dims, slice_mean[s], marker="o", ms=3,
            color=colors[i], label=f"Sliced({s})")
    ax.fill_between(slice_dims,
                    slice_mean[s] - slice_std[s],
                    slice_mean[s] + slice_std[s],
                    alpha=0.15, color=colors[i])

mlp_full_mean = np.mean(results["mlp_full"]["test"])
mlp_half_mean = np.mean(results["mlp_half"]["test"])
ax.axhline(mlp_full_mean, color="black", ls="--", lw=1.5,
           label=f"StandardMLP-full (d={K})  {mlp_full_mean:.3f}")
ax.axhline(mlp_half_mean, color="gray",  ls=":",  lw=1.5,
           label=f"StandardMLP-half (d={K//2})  {mlp_half_mean:.3f}")

ax.set_xlabel("Slice dimension  d_j  (# eigenvectors)", fontsize=12)
ax.set_ylabel("Test accuracy (mean ± std over 10 splits)", fontsize=12)
ax.set_title("Spectral resolution curve — Cornell (heterophilous)", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/per_slice_accuracy.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT}/per_slice_accuracy.png")


# ---------------------------------------------------------------------------
# 3. Shuffle comparison table (split 0, single run)
# ---------------------------------------------------------------------------

torch.manual_seed(SEED)
tr0, val0, te0 = splits[0]

model_unshuf = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                                  loss_weights="uniform")
torch.manual_seed(SEED)
bv_u, st_u, _, _ = train_sliced(model_unshuf, U, labels, tr0, val0, te0)

perm = torch.randperm(K, generator=torch.Generator().manual_seed(SEED))
U_shuf = U[:, perm]
model_shuf = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                                loss_weights="uniform")
torch.manual_seed(SEED)
bv_s, st_s, _, _ = train_sliced(model_shuf, U_shuf, labels, tr0, val0, te0)

shuf_lines = []
shuf_lines.append(f"SHUFFLE COMPARISON TABLE — Cornell  (split 0, seed={SEED})")
shuf_lines.append("=" * 55)
shuf_lines.append(f"{'Slice':>5} | {'Dim':>4} | {'Unshuffled':>10} | {'Shuffled':>10} | {'Diff':>8}")
shuf_lines.append("-" * 55)
for j, (a_u, a_s) in enumerate(zip(st_u, st_s)):
    d = K // 2 + j
    shuf_lines.append(f"{j:>5} | {d:>4} | {a_u:>10.4f} | {a_s:>10.4f} | {a_s - a_u:>+8.4f}")
shuf_lines.append("=" * 55)
shuf_lines.append(f"Best val — unshuffled: {bv_u:.4f}  shuffled: {bv_s:.4f}  diff: {bv_s - bv_u:+.4f}")
shuf_lines.append(f"Coarse (j=0, d={K//2}) — unshuffled: {st_u[0]:.4f}  shuffled: {st_s[0]:.4f}  diff: {st_s[0] - st_u[0]:+.4f}")
shuf_lines.append(f"Full   (j={K//2}, d={K}) — unshuffled: {st_u[-1]:.4f}  shuffled: {st_s[-1]:.4f}  diff: {st_s[-1] - st_u[-1]:+.4f}")

shuf_str = "\n".join(shuf_lines)
print("\n" + shuf_str)
with open(f"{OUT}/shuffle_table.txt", "w") as f:
    f.write(shuf_str + "\n")
print(f"Saved {OUT}/shuffle_table.txt")

# Save coarse-slice drop for cross-dataset table
shuffle_coarse_drop_pp = (st_s[0] - st_u[0]) * 100


# ---------------------------------------------------------------------------
# 5. Training curves (split 0, uniform strategy already recorded)
# ---------------------------------------------------------------------------

# Also need per-epoch val curves for all 5 non-shuffle models on split 0
print("\nRecording per-epoch val curves for all models (split 0)...")
all_val_curves = {}

for strategy in STRATEGIES:
    torch.manual_seed(SEED)
    m = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                          loss_weights=strategy,
                          eigenvalues=eigenvalues if strategy == "eigenvalue" else None)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    curve = []
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad()
        lg = m(U)
        m.compute_loss(lg, labels, tr0).backward()
        opt.step()
        m.eval()
        with torch.no_grad():
            lg = m(U)
        curve.append(acc(lg[-1], labels, val0))
    all_val_curves[f"Sliced({strategy})"] = curve

for tag, n_feat in [("mlp_full", K), ("mlp_half", K // 2)]:
    torch.manual_seed(SEED)
    X = U[:, :n_feat]
    bl = StandardMLP(n_features=n_feat, n_classes=n_classes,
                     hidden_dim=K, n_layers=N_LAYERS)
    opt = torch.optim.Adam(bl.parameters(), lr=LR, weight_decay=WD)
    curve = []
    for _ in range(EPOCHS):
        bl.train(); opt.zero_grad()
        F.cross_entropy(bl(X)[tr0], labels[tr0]).backward()
        opt.step()
        bl.eval()
        with torch.no_grad():
            curve.append(acc(bl(X), labels, val0))
    name = "StandardMLP-full" if tag == "mlp_full" else "StandardMLP-half"
    all_val_curves[name] = curve

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: coarse vs full for uniform
vc, vf = curves_uniform
ax1.plot(vc, lw=1.5, label=f"Coarse slice  j=0, d={K//2}")
ax1.plot(vf, lw=1.5, label=f"Full slice    j={K//2}, d={K}")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val accuracy")
ax1.set_title("Coarse vs Full — Cornell, Sliced(uniform)")
ax1.legend(); ax1.grid(True, alpha=0.3)

# Right: all 5 models
colors5 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#000000", "#888888"]
for (name, curve), col in zip(all_val_curves.items(), colors5):
    ax2.plot(curve, lw=1.5, label=name, color=col)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val accuracy")
ax2.set_title("All models — Cornell, split 0")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUT}/training_curves.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT}/training_curves.png")


# ---------------------------------------------------------------------------
# 6. Cross-dataset summary table
# ---------------------------------------------------------------------------

# Best Sliced full and coarse across strategies (Cornell, mean over splits)
best_sliced_full   = max(np.mean(np.array(results[s]["slice_test"])[:, -1]) for s in STRATEGIES)
best_sliced_coarse = max(np.mean(np.array(results[s]["slice_test"])[:, 0])  for s in STRATEGIES)

cora_mlp_half         = 0.7082
cora_mlp_full         = 0.6448
cora_sliced_best_full = 0.6743
cora_sliced_best_coarse = 0.6918
cora_shuffle_coarse_drop = -17.9

cornell_mlp_half  = np.mean(results["mlp_half"]["test"])
cornell_mlp_full  = np.mean(results["mlp_full"]["test"])

cross = []
cross.append("CROSS-DATASET SUMMARY")
cross.append("=" * 105)
cross.append(f"{'Dataset':<10} | {'MLP-half':>9} | {'MLP-full':>9} | {'Sliced best-full':>17} | {'Sliced best-coarse':>19} | {'Shuffle coarse drop':>20}")
cross.append("-" * 105)
cross.append(
    f"{'Cora':<10} | {cora_mlp_half:>9.4f} | {cora_mlp_full:>9.4f} | "
    f"{cora_sliced_best_full:>17.4f} | {cora_sliced_best_coarse:>19.4f} | "
    f"{cora_shuffle_coarse_drop:>+19.1f}pp"
)
cross.append(
    f"{'Cornell':<10} | {cornell_mlp_half:>9.4f} | {cornell_mlp_full:>9.4f} | "
    f"{best_sliced_full:>17.4f} | {best_sliced_coarse:>19.4f} | "
    f"{shuffle_coarse_drop_pp:>+19.1f}pp"
)
cross.append("=" * 105)
cross.append("")
cross.append("Notes:")
cross.append("  Cora:    single seed, single split (fixed PyG split, LCC=2485 nodes)")
cross.append(f"  Cornell: mean over {N_SPLITS} fixed PyG splits (LCC={N} nodes), single seed per split")

cross_str = "\n".join(cross)
print("\n" + cross_str)
with open(f"{OUT}/cross_dataset_summary.txt", "w") as f:
    f.write(cross_str + "\n")
print(f"Saved {OUT}/cross_dataset_summary.txt")


# ---------------------------------------------------------------------------
# 7. Notes
# ---------------------------------------------------------------------------

# Determine patterns
sliced_beats_mlp_half = best_sliced_full > cornell_mlp_half
slice_acc_monotone = all(slice_mean["uniform"][j] <= slice_mean["uniform"][j+1]
                         for j in range(n_slices - 1))
# direction: is curve generally increasing in the second half?
second_half_slope = slice_mean["uniform"][-1] - slice_mean["uniform"][n_slices // 2]

notes = []
notes.append("ANALYSIS NOTES — Cornell experiment")
notes.append("=" * 60)
notes.append("")
notes.append("Predictions vs observations:")
notes.append("")
notes.append(f"1. Does SlicedSpectralMLP beat StandardMLP-half?")
notes.append(f"   Prediction: yes or much closer than Cora.")
notes.append(f"   Observed: Sliced best-full={best_sliced_full:.4f}  MLP-half={cornell_mlp_half:.4f}")
notes.append(f"   -> {'YES' if sliced_beats_mlp_half else 'NO'} — Sliced {'beats' if sliced_beats_mlp_half else 'does not beat'} MLP-half.")
notes.append("")
notes.append(f"2. Is per-slice curve flatter/monotone increasing (vs Cora's early peak)?")
notes.append(f"   Prediction: flatter or monotone increasing for heterophilous graph.")
notes.append(f"   Observed (uniform): fully monotone = {slice_acc_monotone}")
notes.append(f"   Second-half slope (mid->full, uniform) = {second_half_slope:+.4f}")
notes.append(f"   Cora coarse-slice accuracy was HIGHER than full-slice (peak at j≈11).")
notes.append(f"   Cornell coarse={slice_mean['uniform'][0]:.4f}  full={slice_mean['uniform'][-1]:.4f}")
notes.append("")
notes.append(f"3. Shuffle coarse-slice drop vs Cora's -17.9pp:")
notes.append(f"   Observed: {shuffle_coarse_drop_pp:+.1f}pp  (Cora: -17.9pp)")
if abs(shuffle_coarse_drop_pp) < 10:
    notes.append(f"   -> Much smaller drop. Low-frequency eigenvectors are less privileged")
    notes.append(f"      on Cornell — consistent with heterophilous graph where high-freq")
    notes.append(f"      eigenvectors carry discriminative signal.")
elif abs(shuffle_coarse_drop_pp) >= 17:
    notes.append(f"   -> Similar or larger drop. Spectral ordering still critical.")
else:
    notes.append(f"   -> Moderately smaller. Partial support for spectral-gap hypothesis.")
notes.append("")
notes.append(f"4. Cross-dataset pattern:")
if best_sliced_full > cora_sliced_best_full:
    notes.append(f"   Sliced architecture gains more on Cornell ({best_sliced_full:.4f}) vs Cora ({cora_sliced_best_full:.4f}),")
    notes.append(f"   consistent with high-frequency eigenvectors being useful on heterophilous graphs.")
else:
    notes.append(f"   Sliced architecture does not show clear advantage on Cornell over Cora.")
notes.append("")
notes.append(f"5. MLP-full vs MLP-half on Cornell:")
notes.append(f"   MLP-half={cornell_mlp_half:.4f}  MLP-full={cornell_mlp_full:.4f}")
notes.append(f"   Cora:     MLP-half=0.7082  MLP-full=0.6448  (half wins by 6.3pp)")
notes.append(f"   Cornell:  half {'wins' if cornell_mlp_half > cornell_mlp_full else 'loses'} by {abs(cornell_mlp_half - cornell_mlp_full)*100:+.1f}pp")
notes.append(f"   On Cora, extra eigenvectors hurt. On Cornell this relationship")
notes.append(f"   is expected to be different (heterophily).")

notes_str = "\n".join(notes)
print("\n" + notes_str)
with open(f"{OUT}/notes.txt", "w") as f:
    f.write(notes_str + "\n")
print(f"Saved {OUT}/notes.txt")

print("\n=== All Cornell outputs saved to", OUT, "===")
