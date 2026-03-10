"""
Cora spectral truncation experiment.
Tests whether zeroing out loss for slices past a cutoff fixes the sharing tax.
"""
from __future__ import annotations
import os, numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_dataset
from model import SlicedSpectralMLP
from baselines import StandardMLP, train_baseline

OUT = "outputs/cora_truncation"
os.makedirs(OUT, exist_ok=True)

SEED = 42; K = 64; N_LAYERS = 2; LR = 0.01; WD = 5e-4; EPOCHS = 200

torch.manual_seed(SEED)
U, labels, tr, val, te, eigenvalues = load_dataset("cora", k=K)
n_classes = int(labels.max().item()) + 1
print(f"Cora LCC: N={U.shape[0]}  classes={n_classes}  train={tr.sum()}  val={val.sum()}  test={te.sum()}")


def acc(logits, labels, mask):
    return (logits.argmax(-1)[mask] == labels[mask]).float().mean().item()


def train_sliced(cutoff):
    torch.manual_seed(SEED)
    model = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                               loss_weights="uniform")
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best_val, best_state = 0.0, None
    val_coarse, val_cutoff = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train(); opt.zero_grad()
        lg = model(U)
        model.compute_loss(lg, labels, tr, loss_cutoff=cutoff).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            lg = model(U)
        vc = acc(lg[0], labels, val)
        # track val acc at the cutoff slice (or last slice if cutoff=32)
        active_j = cutoff if cutoff is not None else len(lg) - 1
        vcut = acc(lg[active_j], labels, val)
        val_coarse.append(vc); val_cutoff.append(vcut)
        if vcut > best_val:
            best_val = vcut
            best_state = {k2: v.clone() for k2, v in model.state_dict().items()}

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        lg = model(U)
    slice_test = [acc(logit, labels, te) for logit in lg]
    return best_val, slice_test, val_coarse, val_cutoff


# ---------------------------------------------------------------------------
# Run all cutoff variants
# ---------------------------------------------------------------------------
CUTOFFS = [7, 11, 15, 32]   # 32 = all slices active (baseline)
results = {}

for cutoff in CUTOFFS:
    label = f"cutoff-j{cutoff:02d}"
    print(f"\nRunning Sliced {label} ({'all active' if cutoff==32 else f'j=0..{cutoff} active'})...")
    bv, st, vc, vcut = train_sliced(cutoff)
    results[label] = {"best_val": bv, "slice_test": st,
                      "val_coarse": vc, "val_cutoff": vcut, "cutoff": cutoff}
    active_acc  = st[cutoff]
    best_acc    = max(st)
    best_j      = st.index(best_acc)
    print(f"  best_val={bv:.4f}  test@cutoff={active_acc:.4f}  "
          f"test_coarse={st[0]:.4f}  best_slice={best_acc:.4f} (j={best_j})")

# ---------------------------------------------------------------------------
# Run 5: StandardMLP-k43  (the actual peak from the untruncated run)
# ---------------------------------------------------------------------------
print("\nRunning StandardMLP-k43...")
torch.manual_seed(SEED)
bl43 = StandardMLP(n_features=43, n_classes=n_classes, hidden_dim=K, n_layers=N_LAYERS)
bv43, ta43 = train_baseline(bl43, U[:, :43], labels, tr, val, te,
                              lr=LR, wd=WD, epochs=EPOCHS)
print(f"  best_val={bv43:.4f}  test={ta43:.4f}")

# Prior baselines from the eval.py run (single seed, 200 epochs)
MLP_k32_val  = 0.7255; MLP_k32_test  = 0.7082
MLP_k64_val  = 0.6514; MLP_k64_test  = 0.6448

# ---------------------------------------------------------------------------
# 1. Comparison table
# ---------------------------------------------------------------------------
best_truncated_test = max(
    max(results[f"cutoff-j{c:02d}"]["slice_test"])
    for c in [7, 11, 15]
)
best_truncated_label = ""
for c in [7, 11, 15]:
    st = results[f"cutoff-j{c:02d}"]["slice_test"]
    if max(st) == best_truncated_test:
        best_truncated_label = f"cutoff-j{c:02d}"

header = [
    "",
    "  DOES TRUNCATED SLICED BEAT MLP-HALF ON CORA?",
    f"  MLP-half (k=32):          {MLP_k32_test:.4f}",
    f"  MLP-k43  (new):           {ta43:.4f}",
    f"  Best truncated Sliced:    {best_truncated_test:.4f}  ({best_truncated_label})",
    "",
]
if best_truncated_test > MLP_k32_test:
    header.append("  Answer: YES — truncation fixes the sharing tax.")
elif best_truncated_test > MLP_k32_test - 0.005:
    header.append("  Answer: MARGINAL — within 0.5pp of MLP-half.")
else:
    header.append("  Answer: NO — truncation does not recover MLP-half performance.")

if ta43 > best_truncated_test:
    header.append("  MLP-k43 > best truncated Sliced — sliced architecture adds no value beyond right truncation.")
else:
    header.append("  Truncated Sliced >= MLP-k43 — sliced structure provides benefit beyond simple truncation.")
header.append("")

div = "=" * 95
lines = header + [
    div,
    f"COMPARISON TABLE — Cora spectral truncation (k={K}, seed={SEED}, {EPOCHS} epochs)",
    div,
    f"{'Method':<26} {'Best Val':>9} {'Test@active':>12} {'Test(coarse)':>13} {'Test(best)':>11}",
    "-" * 95,
]

for c in CUTOFFS:
    label = f"cutoff-j{c:02d}"
    r = results[label]
    st = r["slice_test"]
    test_at_active = st[c]
    tag = f"Sliced-{label}"
    if c == 32:
        tag += " (baseline)"
    lines.append(
        f"{tag:<26} {r['best_val']:>9.4f} {test_at_active:>12.4f} "
        f"{st[0]:>13.4f} {max(st):>11.4f}"
    )

lines.append(f"{'StandardMLP-k43':<26} {bv43:>9.4f} {ta43:>12.4f} {'—':>13} {'—':>11}")
lines.append(f"{'StandardMLP-k32 (prior)':<26} {MLP_k32_val:>9.4f} {MLP_k32_test:>12.4f} {'—':>13} {'—':>11}")
lines.append(f"{'StandardMLP-k64 (prior)':<26} {MLP_k64_val:>9.4f} {MLP_k64_test:>12.4f} {'—':>13} {'—':>11}")
lines.append(div)
lines.append("")
lines.append("Notes:")
lines.append("  Test@active = test acc of the largest ACTIVE slice (j=cutoff).")
lines.append("  Test(best)  = best test acc across ALL 33 slices (including inactive).")
lines.append("  Inactive slices still run forward; their heads just received no loss signal.")

table_str = "\n".join(lines)
print("\n" + table_str)
with open(f"{OUT}/comparison_table.txt", "w") as f:
    f.write(table_str + "\n")

# ---------------------------------------------------------------------------
# 2. Per-slice accuracy curve
# ---------------------------------------------------------------------------
slice_dims = [K // 2 + j for j in range(K // 2 + 1)]
colors = {7: "#d62728", 11: "#1f77b4", 15: "#ff7f0e", 32: "#888888"}
labels_map = {7: "cutoff j=7",  11: "cutoff j=11",
              15: "cutoff j=15", 32: "cutoff j=32 (full, baseline)"}

fig, ax = plt.subplots(figsize=(11, 5))
for c in CUTOFFS:
    st = results[f"cutoff-j{c:02d}"]["slice_test"]
    lw = 2.0 if c != 32 else 1.2
    ls = "-" if c != 32 else "--"
    ax.plot(slice_dims, st, color=colors[c], lw=lw, ls=ls,
            marker="o", ms=2.5, label=labels_map[c])

# Cutoff vertical markers
for c, col in [(7, colors[7]), (11, colors[11]), (15, colors[15])]:
    d = K // 2 + c
    ax.axvline(d, color=col, lw=0.8, alpha=0.5, ls=":")

# Baseline horizontals
ax.axhline(MLP_k32_test, color="black", ls="--", lw=1.5,
           label=f"MLP-k32 (d=32)  {MLP_k32_test:.4f}")
ax.axhline(ta43, color="purple", ls="-.", lw=1.5,
           label=f"MLP-k43 (d=43)  {ta43:.4f}")

ax.set_xlabel("Slice dimension  d_j  (# eigenvectors)", fontsize=12)
ax.set_ylabel("Test accuracy", fontsize=12)
ax.set_title("Spectral truncation — Cora (do inactive slices collapse?)", fontsize=13)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/per_slice_accuracy.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT}/per_slice_accuracy.png")

# ---------------------------------------------------------------------------
# 3. Training curves (2×2 grid, one per cutoff run)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
axes = axes.flatten()
for i, c in enumerate(CUTOFFS):
    r = results[f"cutoff-j{c:02d}"]
    ax = axes[i]
    ax.plot(r["val_coarse"], lw=1.5, label=f"Coarse j=0, d={K//2}")
    ax.plot(r["val_cutoff"], lw=1.5,
            label=f"Cutoff j={c}, d={K//2+c}" + (" (full)" if c==32 else ""))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val accuracy")
    title = f"cutoff j={c}" + (" — baseline (all active)" if c==32 else f" — j=0..{c} active")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.suptitle("Training curves — Cora spectral truncation", fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(f"{OUT}/training_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT}/training_curves.png")

print("\nAll outputs saved to", OUT)
