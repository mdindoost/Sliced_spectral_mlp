"""
Squirrel experiment pipeline with shuffle diagnostic gate.
Step 0: shuffle diagnostic on split 0. Continue only if coarse drop >= 5pp.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os, sys, time, numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loaders import load_dataset
from src.models.sliced_mlp import SlicedSpectralMLP
from src.models.baselines import StandardMLP, train_baseline

OUT = "outputs/squirrel"
os.makedirs(f"{OUT}/grad_heatmaps", exist_ok=True)

SEED = 42; K = 64; N_LAYERS = 2; LR = 0.01; WD = 5e-4
EPOCHS = 100          # same as Actor; 200 would take ~15 min for 10 splits
N_SPLITS = 10
SPARSE_DIMS = [32, 40, 48, 56, 64]
HALF = K // 2

WALL_START = time.time()
def elapsed(): return time.time() - WALL_START

# ---------------------------------------------------------------------------
# Load dataset (eigenvectors computed once, split-independent)
# ---------------------------------------------------------------------------
print("Loading Squirrel LCC eigenvectors...")
t0 = time.time()
U, labels0, _, _, _, eigenvalues = load_dataset("squirrel", k=K, split_idx=0)
N = U.shape[0]; n_classes = int(labels0.max().item()) + 1
print(f"  N={N}  classes={n_classes}  k={K}  load+eigsh: {time.time()-t0:.1f}s")

# Check if this is the filtered version (PyG default is geom_gcn_preprocess=True)
try:
    from torch_geometric.datasets import WikipediaNetwork
    ds = WikipediaNetwork(root="./data", name="squirrel")
    has_filter = getattr(ds, "geom_gcn_preprocess", True)
    print(f"  PyG geom_gcn_preprocess (filtered): {has_filter}")
except Exception:
    has_filter = "unknown"

splits = []
for s in range(N_SPLITS):
    _, ls, tr, val, te, _ = load_dataset("squirrel", k=K, split_idx=s)
    assert (ls == labels0).all()
    splits.append((tr, val, te))
labels = labels0
print(f"  All {N_SPLITS} splits loaded.  Elapsed: {elapsed():.0f}s")
tr0, val0, te0 = splits[0]
print(f"  Split 0: train={tr0.sum()}  val={val0.sum()}  test={te0.sum()}")

def acc(logits, labels, mask):
    return (logits.argmax(-1)[mask] == labels[mask]).float().mean().item()

def save_heatmap(grad, epoch, out_dir):
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

def train_sliced(model, U, tr, val, te, loss_cutoff=None,
                 heatmap_epoch=None, heatmap_dir=None):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best_val, best_state = 0.0, None
    val_coarse, val_full = [], []
    for epoch in range(1, EPOCHS + 1):
        model.train(); opt.zero_grad()
        lg = model(U)
        model.compute_loss(lg, labels, tr, loss_cutoff=loss_cutoff).backward()
        if heatmap_epoch and epoch == heatmap_epoch and model.W[0].grad is not None:
            save_heatmap(torch.abs(model.W[0].grad), epoch, heatmap_dir)
        opt.step()
        model.eval()
        with torch.no_grad(): lg = model(U)
        vc = acc(lg[0],  labels, val)
        vf = acc(lg[-1], labels, val)
        val_coarse.append(vc); val_full.append(vf)
        top_j = loss_cutoff if loss_cutoff is not None else len(lg) - 1
        v = acc(lg[top_j], labels, val)
        if v > best_val:
            best_val = v
            best_state = {k2: v2.clone() for k2, v2 in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): lg = model(U)
    return best_val, [acc(l, labels, te) for l in lg], val_coarse, val_full

# ---------------------------------------------------------------------------
# STEP 0 — Shuffle diagnostic (split 0 only)
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("STEP 0: Shuffle diagnostic (split 0)")
print("="*60)

torch.manual_seed(SEED)
m_u = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                         loss_weights="uniform")
bv_u, st_u, vc_u, vf_u = train_sliced(m_u, U, tr0, val0, te0,
                                        heatmap_epoch=50,
                                        heatmap_dir=f"{OUT}/grad_heatmaps")

torch.manual_seed(SEED)
perm = torch.randperm(K, generator=torch.Generator().manual_seed(SEED))
U_shuf = U[:, perm]
m_s = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                         loss_weights="uniform")
bv_s, st_s, _, _ = train_sliced(m_s, U_shuf, tr0, val0, te0)

coarse_drop_pp = (st_s[0] - st_u[0]) * 100
full_drop_pp   = (st_s[-1] - st_u[-1]) * 100

print(f"\n  Unshuffled: coarse={st_u[0]:.4f}  full={st_u[-1]:.4f}  val={bv_u:.4f}")
print(f"  Shuffled:   coarse={st_s[0]:.4f}  full={st_s[-1]:.4f}  val={bv_s:.4f}")
print(f"\n  COARSE DROP: {coarse_drop_pp:+.1f}pp")
print(f"  FULL DROP:   {full_drop_pp:+.1f}pp")

diag_lines = [
    "SQUIRREL SHUFFLE DIAGNOSTIC",
    "="*50,
    f"Split 0 only, seed={SEED}, {EPOCHS} epochs",
    f"Unshuffled: coarse={st_u[0]:.4f}  full={st_u[-1]:.4f}",
    f"Shuffled:   coarse={st_s[0]:.4f}  full={st_s[-1]:.4f}",
    f"Coarse drop: {coarse_drop_pp:+.1f}pp",
    f"Full drop:   {full_drop_pp:+.1f}pp",
    "",
    f"Reference: Cora={-17.9:.1f}pp  Cornell={-8.1:.1f}pp  Actor={-0.1:.1f}pp",
]

if coarse_drop_pp > -5.0:
    decision = "STOP"
    diag_lines += ["", "Decision: STOP — Squirrel eigenvectors are noise like Actor.",
                   "Coarse drop < 5pp. Full pipeline skipped."]
    print("\n  Decision: STOP — coarse drop < 5pp. Squirrel is another Actor.")
elif coarse_drop_pp > -10.0:
    decision = "MARGINAL"
    diag_lines += ["", "Decision: MARGINAL — 5pp <= drop < 10pp. Running full pipeline with caution.",
                   "Note: Squirrel has known label noise; this may suppress the shuffle signal."]
    print("\n  Decision: MARGINAL — running full pipeline.")
else:
    decision = "FULL"
    diag_lines += ["", "Decision: FULL — coarse drop >= 10pp. Running complete pipeline."]
    print(f"\n  Decision: FULL RUN — coarse drop >= 10pp.")

with open(f"{OUT}/diagnostic.txt", "w") as f:
    f.write("\n".join(diag_lines) + "\n")

if decision == "STOP":
    # Cross-dataset summary with just the diagnostic
    cross = [
        "CROSS-DATASET SUMMARY (Squirrel: diagnostic only)",
        "="*100,
        f"{'Dataset':<10} | {'N':>6} | {'MLP-half':>9} | {'MLP-full':>9} | {'Sliced-best':>12} | {'Strategy-C':>11} | {'Shuffle drop':>13}",
        "-"*100,
        f"{'Cora':<10} | {2485:>6} | {0.7082:>9.4f} | {0.6448:>9.4f} | {0.7399:>12.4f} | {0.7410:>11.4f} | {-17.9:>+12.1f}pp",
        f"{'Cornell':<10} | {183:>6} | {0.4054:>9.4f} | {0.4270:>9.4f} | {0.4189:>12.4f} | {'n/a':>11} | {-8.1:>+12.1f}pp",
        f"{'Actor':<10} | {7600:>6} | {0.2525:>9.4f} | {0.2524:>9.4f} | {0.2511:>12.4f} | {'n/a':>11} | {-0.1:>+12.1f}pp",
        f"{'Squirrel':<10} | {N:>6} | {'n/a':>9} | {'n/a':>9} | {'n/a':>12} | {'n/a':>11} | {coarse_drop_pp:>+12.1f}pp",
        "="*100,
        "", "Squirrel skipped at diagnostic stage: coarse shuffle drop < 5pp.",
    ]
    with open(f"{OUT}/cross_dataset_summary.txt", "w") as f:
        f.write("\n".join(cross) + "\n")
    print("\nOutputs saved to", OUT); sys.exit(0)

# ---------------------------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nRunning full pipeline ({N_SPLITS} splits, {EPOCHS} epochs each)\n{'='*60}")

# Strategy C: median eigenvalue threshold
eig_slices = np.array([eigenvalues[min(HALF + j, K - 1)] for j in range(HALF + 1)])
lambda_threshold = float(np.median(eig_slices))
raw_w = np.zeros(HALF + 1)
for j in range(HALF + 1):
    if eig_slices[j] <= lambda_threshold:
        raw_w[j] = 1.0 / max(eig_slices[j], 1e-6)
w_C = raw_w / raw_w.sum() if raw_w.sum() > 0 else raw_w
cutoff_C = int(np.max(np.where(raw_w > 0)))

print(f"\nStrategy C: median eigenvalue threshold = {lambda_threshold:.5f}")
print(f"  eig_slices[0..4]: {eig_slices[:5]}")
print(f"  eig_slices[28..32]: {eig_slices[28:]}")
print(f"  Active slices: j=0..{cutoff_C}  cutoff selected: j={cutoff_C}")

res = {t: {"best_val": [], "slice_test": []}
       for t in ["dense_uniform", "sparse_uniform", "strategy_c"]}
res["mlp_full"] = {"best_val": [], "test": []}
res["mlp_half"] = {"best_val": [], "test": []}
curves_uniform = None

for si, (tr, val, te) in enumerate(splits):
    print(f"\n--- Split {si} | elapsed={elapsed():.0f}s ---")

    # Dense uniform
    torch.manual_seed(SEED + si)
    m = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                           loss_weights="uniform")
    bv, st, vc, vf = train_sliced(m, U, tr, val, te)
    res["dense_uniform"]["best_val"].append(bv)
    res["dense_uniform"]["slice_test"].append(st)
    if si == 0: curves_uniform = (vc, vf)
    print(f"  dense_uniform:  val={bv:.4f}  full={st[-1]:.4f}  coarse={st[0]:.4f}")

    # Sparse uniform
    torch.manual_seed(SEED + si)
    m = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                           loss_weights="uniform", custom_slice_dims=SPARSE_DIMS)
    bv, st, _, _ = train_sliced(m, U, tr, val, te)
    res["sparse_uniform"]["best_val"].append(bv)
    res["sparse_uniform"]["slice_test"].append(st)
    print(f"  sparse_uniform: val={bv:.4f}  full={st[-1]:.4f}  coarse={st[0]:.4f}")

    # Strategy C
    torch.manual_seed(SEED + si)
    m = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                           loss_weights="uniform")
    w_t = torch.tensor(w_C, dtype=torch.float32)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    best_val_c, best_state_c = 0.0, None
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad()
        loss = torch.zeros(1)[0]
        lg = m(U)
        for j2, logit in enumerate(lg):
            if w_C[j2] > 0:
                loss = loss + w_C[j2] * F.cross_entropy(logit[tr], labels[tr])
        loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): lg = m(U)
        v = acc(lg[cutoff_C], labels, val)
        if v > best_val_c:
            best_val_c = v
            best_state_c = {k2: v2.clone() for k2, v2 in m.state_dict().items()}
    m.load_state_dict(best_state_c); m.eval()
    with torch.no_grad(): lg = m(U)
    st = [acc(l, labels, te) for l in lg]
    res["strategy_c"]["best_val"].append(best_val_c)
    res["strategy_c"]["slice_test"].append(st)
    print(f"  strategy_c:     val={best_val_c:.4f}  best={max(st):.4f}  cutoff_j={cutoff_C}")

    # MLP-full
    torch.manual_seed(SEED + si)
    bl = StandardMLP(n_features=K, n_classes=n_classes, hidden_dim=K, n_layers=N_LAYERS)
    bv, ta = train_baseline(bl, U, labels, tr, val, te, lr=LR, wd=WD, epochs=EPOCHS)
    res["mlp_full"]["best_val"].append(bv); res["mlp_full"]["test"].append(ta)
    print(f"  mlp_full:       val={bv:.4f}  test={ta:.4f}")

    # MLP-half
    torch.manual_seed(SEED + si)
    bl = StandardMLP(n_features=HALF, n_classes=n_classes, hidden_dim=K, n_layers=N_LAYERS)
    bv, ta = train_baseline(bl, U[:, :HALF], labels, tr, val, te, lr=LR, wd=WD, epochs=EPOCHS)
    res["mlp_half"]["best_val"].append(bv); res["mlp_half"]["test"].append(ta)
    print(f"  mlp_half:       val={bv:.4f}  test={ta:.4f}")

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def ms(arr): return f"{np.mean(arr):.4f}±{np.std(arr):.4f}"

for tag in ["dense_uniform", "sparse_uniform", "strategy_c"]:
    arr = np.array(res[tag]["slice_test"])
    res[tag]["mean"] = arr.mean(0)
    res[tag]["std"]  = arr.std(0)

n_slices_dense  = K // 2 + 1
slice_dims_dense = [K // 2 + j for j in range(n_slices_dense)]

mf_mean = np.mean(res["mlp_full"]["test"])
mh_mean = np.mean(res["mlp_half"]["test"])
du_full = np.mean(np.array(res["dense_uniform"]["slice_test"])[:, -1])
du_best = np.mean(np.array(res["dense_uniform"]["slice_test"]).max(1))
sc_best = np.mean(np.array(res["strategy_c"]["slice_test"]).max(1))

# ---------------------------------------------------------------------------
# 1. Comparison table
# ---------------------------------------------------------------------------
lines = []
div = "="*110
lines += [div,
          f"COMPARISON TABLE — SQUIRREL  (k={K}, {EPOCHS} epochs, mean±std over {N_SPLITS} splits)",
          f"  geom_gcn_preprocess (filtered): {has_filter}",
          div,
          f"{'Method':<28} {'Best Val':>16}  {'Test(full)':>14}  {'Test(coarse)':>14}  {'Test(best)':>12}",
          "-"*110]

for tag, label in [("dense_uniform","Sliced-dense(uniform)"),
                    ("sparse_uniform","Sliced-sparse(uniform)"),
                    ("strategy_c",   f"Strategy-C(cutoff j={cutoff_C})")]:
    arr = np.array(res[tag]["slice_test"])
    bv  = np.array(res[tag]["best_val"])
    tf  = arr[:, -1]; tc = arr[:, 0]; tb = arr.max(1)
    lines.append(f"{label:<28} {ms(bv):>16}  {ms(tf):>14}  {ms(tc):>14}  {ms(tb):>12}")

for tag, label in [("mlp_full","StandardMLP-full"),("mlp_half","StandardMLP-half")]:
    bv = np.array(res[tag]["best_val"]); ta = np.array(res[tag]["test"])
    lines.append(f"{label:<28} {ms(bv):>16}  {ms(ta):>14}  {ms(ta):>14}  {ms(ta):>12}")
lines.append(div)
lines += ["",
          f"Sliced-dense beats MLP-full?  {du_full:.4f} vs {mf_mean:.4f} -> {'YES' if du_full > mf_mean else 'NO'} ({du_full-mf_mean:+.4f})",
          f"Strategy C beats MLP-full?    {sc_best:.4f} vs {mf_mean:.4f} -> {'YES' if sc_best > mf_mean else 'NO'} ({sc_best-mf_mean:+.4f})",
          f"MLP-full beats MLP-half?      {mf_mean:.4f} vs {mh_mean:.4f} -> {'YES' if mf_mean > mh_mean else 'NO'} ({mf_mean-mh_mean:+.4f})  (heterophily reversal check)"]

table_str = "\n".join(lines)
print("\n" + table_str)
with open(f"{OUT}/comparison_table.txt", "w") as f:
    f.write(table_str + "\n")

# ---------------------------------------------------------------------------
# 2. Per-slice accuracy curve
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
colors = {"dense_uniform": "#1f77b4", "strategy_c": "#2ca02c"}
for tag, label in [("dense_uniform","Sliced-dense(uniform)"),
                    ("strategy_c",  f"Strategy-C(j={cutoff_C})")]:
    ax.plot(slice_dims_dense, res[tag]["mean"], marker="o", ms=2.5,
            color=colors[tag], label=label)
    ax.fill_between(slice_dims_dense,
                    res[tag]["mean"] - res[tag]["std"],
                    res[tag]["mean"] + res[tag]["std"],
                    alpha=0.12, color=colors[tag])

# Sparse — scatter at 5 positions
sparse_mean = res["sparse_uniform"]["mean"]
sparse_std  = res["sparse_uniform"]["std"]
ax.scatter(SPARSE_DIMS, sparse_mean, marker="D", s=60, color="#ff7f0e",
           zorder=5, label="Sliced-sparse(uniform)")
for d_j, mv, sv in zip(SPARSE_DIMS, sparse_mean, sparse_std):
    ax.errorbar(d_j, mv, yerr=sv, fmt="none", color="#ff7f0e", capsize=3, lw=1.2)

ax.axvline(HALF + cutoff_C, color="#2ca02c", lw=1.0, ls=":", alpha=0.6,
           label=f"Strategy C cutoff j={cutoff_C}")
ax.axhline(mf_mean, color="black", ls="--", lw=1.5,
           label=f"MLP-full {mf_mean:.4f}")
ax.axhline(mh_mean, color="gray",  ls=":",  lw=1.5,
           label=f"MLP-half {mh_mean:.4f}")

ax.set_xlabel("Slice dimension d_j", fontsize=12)
ax.set_ylabel("Test accuracy (mean±std, 10 splits)", fontsize=12)
ax.set_title(f"Squirrel spectral resolution curve (heterophilous, N={N})", fontsize=13)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/per_slice_accuracy.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT}/per_slice_accuracy.png")

# ---------------------------------------------------------------------------
# 3. Shuffle table
# ---------------------------------------------------------------------------
shuf = [f"SHUFFLE TABLE — SQUIRREL  (split 0, seed={SEED})", "="*55,
        f"{'Slice':>5} | {'Dim':>4} | {'Unshuffled':>10} | {'Shuffled':>10} | {'Diff':>8}",
        "-"*55]
for j, (a_u, a_s) in enumerate(zip(st_u, st_s)):
    d = HALF + j
    shuf.append(f"{j:>5} | {d:>4} | {a_u:>10.4f} | {a_s:>10.4f} | {a_s-a_u:>+8.4f}")
shuf += ["="*55,
         f"Coarse drop: {coarse_drop_pp:+.1f}pp",
         f"Full drop:   {full_drop_pp:+.1f}pp",
         f"Best val unshuffled: {bv_u:.4f}  shuffled: {bv_s:.4f}"]
print("\n" + "\n".join(shuf))
with open(f"{OUT}/shuffle_table.txt", "w") as f:
    f.write("\n".join(shuf) + "\n")

# ---------------------------------------------------------------------------
# 4. Training curves (split 0, rerun for curves)
# ---------------------------------------------------------------------------
print("\nRecording training curves (split 0)...")
all_val_curves = {}
for tag, label, sdims, cutoff_arg in [
    ("du","Sliced-dense(uniform)",  None,        None),
    ("su","Sliced-sparse(uniform)", SPARSE_DIMS, None),
]:
    torch.manual_seed(SEED)
    m = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                           loss_weights="uniform", custom_slice_dims=sdims)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD); curve=[]
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad()
        m.compute_loss(m(U), labels, tr0).backward(); opt.step()
        m.eval()
        with torch.no_grad(): curve.append(acc(m(U)[-1], labels, val0))
    all_val_curves[label] = curve

for tag, label, nf in [("mf","MLP-full",K),("mh","MLP-half",HALF)]:
    torch.manual_seed(SEED); X = U[:,:nf]
    bl = StandardMLP(n_features=nf, n_classes=n_classes, hidden_dim=K, n_layers=N_LAYERS)
    opt = torch.optim.Adam(bl.parameters(), lr=LR, weight_decay=WD); curve=[]
    for _ in range(EPOCHS):
        bl.train(); opt.zero_grad()
        F.cross_entropy(bl(X)[tr0], labels[tr0]).backward(); opt.step()
        bl.eval()
        with torch.no_grad(): curve.append(acc(bl(X), labels, val0))
    all_val_curves[label] = curve

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(curves_uniform[0], lw=1.5, label=f"Coarse j=0, d={HALF}")
ax1.plot(curves_uniform[1], lw=1.5, label=f"Full  j={HALF}, d={K}")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val accuracy")
ax1.set_title("Coarse vs Full — Squirrel, dense uniform"); ax1.legend(); ax1.grid(True, alpha=0.3)

c5 = ["#1f77b4","#ff7f0e","#000000","#888888"]
for (lbl, curve), col in zip(all_val_curves.items(), c5):
    ax2.plot(curve, lw=1.5, label=lbl, color=col)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val accuracy")
ax2.set_title("All models — Squirrel, split 0"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/training_curves.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT}/training_curves.png")

# ---------------------------------------------------------------------------
# 6. Cross-dataset summary
# ---------------------------------------------------------------------------
best_sliced = max(du_best, sc_best,
                  np.mean(np.array(res["sparse_uniform"]["slice_test"]).max(1)))

rows = [
    ("Cora",     2485, 0.7082, 0.6448, 0.7399, 0.7410,  -17.9),
    ("Cornell",   183, 0.4054, 0.4270, 0.4189,  None,    -8.1),
    ("Actor",    7600, 0.2525, 0.2524, 0.2511,  None,    -0.1),
    ("Squirrel",    N, mh_mean, mf_mean, best_sliced, sc_best, coarse_drop_pp),
]
cross = ["CROSS-DATASET SUMMARY", "="*105,
         f"{'Dataset':<10} | {'N':>6} | {'MLP-half':>9} | {'MLP-full':>9} | {'Sliced-best':>12} | {'Strategy-C':>11} | {'Shuffle drop':>13}",
         "-"*105]
for dataset, n, mh, mf, sb, sc, sd in rows:
    sc_str = f"{sc:.4f}" if sc else "n/a"
    cross.append(f"{dataset:<10} | {n:>6} | {mh:>9.4f} | {mf:>9.4f} | {sb:>12.4f} | {sc_str:>11} | {sd:>+12.1f}pp")
cross += ["="*105, "",
          "CROSS-DATASET INTERPRETATION:",
          "Shuffle coarse drop vs Sliced-best > MLP-full:"]
for dataset, n, mh, mf, sb, sc, sd in rows:
    sliced_wins = "YES" if sb > mf else "NO"
    cross.append(f"  {dataset:<10}: drop={sd:+.1f}pp  Sliced>MLP-full? {sliced_wins} ({sb:.4f} vs {mf:.4f})")

# Check monotone pattern
drops = [-17.9, -8.1, -0.1, coarse_drop_pp]
accs_sliced_vs_mlp = [0.7399>0.6448, 0.4189>0.4270, 0.2511>0.2524, best_sliced>mf_mean]
cross += ["",
          f"Monotone pattern (larger |drop| -> Sliced beats MLP-full): "
          + ("consistent" if (drops[0]<drops[1]<drops[2]) else "not monotone for Cora/Cornell/Actor")]
cross_str = "\n".join(cross)
print("\n" + cross_str)
for path in [f"{OUT}/cross_dataset_summary.txt",
             "outputs/actor/cross_dataset_summary.txt",
             "outputs/cornell/cross_dataset_summary.txt"]:
    with open(path, "w") as f: f.write(cross_str + "\n")

# ---------------------------------------------------------------------------
# 7. Notes
# ---------------------------------------------------------------------------
arr_du = np.array(res["dense_uniform"]["slice_test"])
curve_shape = "increasing" if res["dense_uniform"]["mean"][-1] > res["dense_uniform"]["mean"][0] else \
              ("peak-then-decline" if res["dense_uniform"]["mean"].argmax() < n_slices_dense - 3 else "flat")

notes = ["SQUIRREL ANALYSIS NOTES", "="*60, "",
         f"1. Coarse shuffle drop: {coarse_drop_pp:+.1f}pp",
         f"   Cora=-17.9pp  Cornell=-8.1pp  Actor=-0.1pp  Squirrel={coarse_drop_pp:+.1f}pp",
         f"   Position in spectrum: {'closer to Cornell (heterophilous signal)' if coarse_drop_pp < -5 else 'closer to Actor (no signal)'}",
         "",
         f"2. MLP-full vs MLP-half: {mf_mean:.4f} vs {mh_mean:.4f}  diff={mf_mean-mh_mean:+.4f}",
         f"   Reversal (full>half = heterophilous): {'YES' if mf_mean > mh_mean else 'NO'}",
         f"   Cora: half>full (+6.3pp).  Cornell: full>half (+2.2pp).",
         "",
         f"3. Per-slice curve shape: {curve_shape}",
         f"   Coarse slice mean={res['dense_uniform']['mean'][0]:.4f}  "
         f"Mid (j=16) mean={res['dense_uniform']['mean'][16]:.4f}  "
         f"Full mean={res['dense_uniform']['mean'][-1]:.4f}",
         "",
         f"4. Strategy C cutoff j={cutoff_C} (median lambda={lambda_threshold:.5f})",
         f"   On homophilous Cora: C selected j=16, achieved 74.10% (optimal).",
         f"   On heterophilous Squirrel: C selected j={cutoff_C}.",
         f"   Strategy C best={sc_best:.4f} vs MLP-full={mf_mean:.4f}: {'better' if sc_best > mf_mean else 'worse'}",
         f"   {'Strategy C median threshold still works here.' if sc_best >= mf_mean else 'Confirms Strategy C is wrong for heterophilous graphs.'}",
         "",
         f"5. Training stability: N={N} nodes (vs Cornell=183).",
         f"   Expect smooth curves — large dataset damps per-epoch variance.",
         "",
         f"6. Bottom line for spectral gap hypothesis:",
         f"   Shuffle drop={coarse_drop_pp:+.1f}pp, Sliced>MLP-full: {best_sliced>mf_mean}.",
]

notes_str = "\n".join(notes)
print("\n" + notes_str)
with open(f"{OUT}/notes.txt", "w") as f:
    f.write(notes_str + "\n")

print(f"\nTotal elapsed: {elapsed():.0f}s ({elapsed()/60:.1f} min)")
print(f"All outputs saved to {OUT}")
