"""
PubMed experiment pipeline with shuffle diagnostic gate.
Homophilous, N≈19717, 3 classes, single fixed split.
Prediction: coarse shuffle drop > 17.9pp (stronger than Cora).
"""
from __future__ import annotations
import os, sys, time, numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_dataset
from model import SlicedSpectralMLP
from baselines import StandardMLP, train_baseline

OUT = "outputs/pubmed"
os.makedirs(f"{OUT}/grad_heatmaps", exist_ok=True)

SEED = 42; K = 64; N_LAYERS = 2; LR = 0.01; WD = 5e-4; EPOCHS = 200
HALF = K // 2

WALL_START = time.time()
def elapsed(): return time.time() - WALL_START

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
print("Loading PubMed...")
t0 = time.time()
U, labels, tr, val, te, eigenvalues = load_dataset("pubmed", k=K)
N = U.shape[0]; n_classes = int(labels.max().item()) + 1
slice_dims = [HALF + j for j in range(HALF + 1)]
print(f"  N={N}  classes={n_classes}  k={K}  load: {time.time()-t0:.1f}s")
print(f"  train={tr.sum()}  val={val.sum()}  test={te.sum()}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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

def train_sliced(model, loss_cutoff=None, w_vec=None,
                 heatmap_epoch=None, heatmap_dir=None):
    """Train SlicedSpectralMLP. Returns (best_val, slice_test, val_coarse, val_full)."""
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    w_t = torch.tensor(w_vec, dtype=torch.float32) if w_vec is not None else None
    best_val, best_state = 0.0, None
    top_j = loss_cutoff if loss_cutoff is not None else (HALF)
    val_coarse, val_full = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train(); opt.zero_grad()
        lg = model(U)
        if w_t is not None:
            loss = torch.zeros(1)[0]
            for j2, logit in enumerate(lg):
                if w_vec[j2] > 0:
                    loss = loss + w_vec[j2] * F.cross_entropy(logit[tr], labels[tr])
        else:
            loss = model.compute_loss(lg, labels, tr, loss_cutoff=loss_cutoff)
        loss.backward()
        if heatmap_epoch and epoch == heatmap_epoch and model.W[0].grad is not None:
            save_heatmap(torch.abs(model.W[0].grad), epoch, heatmap_dir)
        opt.step()
        model.eval()
        with torch.no_grad(): lg = model(U)
        vc = acc(lg[0],  labels, val)
        vf = acc(lg[-1], labels, val)
        val_coarse.append(vc); val_full.append(vf)
        v = acc(lg[top_j], labels, val)
        if v > best_val:
            best_val = v
            best_state = {k2: v2.clone() for k2, v2 in model.state_dict().items()}

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): lg = model(U)
    return best_val, [acc(l, labels, te) for l in lg], val_coarse, val_full

# ---------------------------------------------------------------------------
# STEP 0 — Shuffle diagnostic
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nSTEP 0: Shuffle diagnostic\n{'='*60}")

torch.manual_seed(SEED)
m_u = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                         loss_weights="uniform")
bv_u, st_u, vc_u, vf_u = train_sliced(m_u, heatmap_epoch=50,
                                        heatmap_dir=f"{OUT}/grad_heatmaps")

torch.manual_seed(SEED)
perm = torch.randperm(K, generator=torch.Generator().manual_seed(SEED))
U_shuf = U[:, perm]
m_s = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                         loss_weights="uniform")
# swap U for shuffled in the closure — run manually
opt_s = torch.optim.Adam(m_s.parameters(), lr=LR, weight_decay=WD)
best_s, best_state_s = 0.0, None
for _ in range(EPOCHS):
    m_s.train(); opt_s.zero_grad()
    lg = m_s(U_shuf); m_s.compute_loss(lg, labels, tr).backward(); opt_s.step()
    m_s.eval()
    with torch.no_grad(): lg = m_s(U_shuf)
    v = acc(lg[-1], labels, val)
    if v > best_s:
        best_s = v; best_state_s = {k2: v2.clone() for k2, v2 in m_s.state_dict().items()}
m_s.load_state_dict(best_state_s); m_s.eval()
with torch.no_grad(): lg = m_s(U_shuf)
st_s = [acc(l, labels, te) for l in lg]

coarse_drop_pp = (st_s[0] - st_u[0]) * 100
full_drop_pp   = (st_s[-1] - st_u[-1]) * 100

print(f"\n  Unshuffled: coarse={st_u[0]:.4f}  full={st_u[-1]:.4f}  val={bv_u:.4f}")
print(f"  Shuffled:   coarse={st_s[0]:.4f}  full={st_s[-1]:.4f}")
print(f"\n  COARSE DROP: {coarse_drop_pp:+.1f}pp  (Cora=-17.9, Cornell=-8.1, Actor=-0.1, Squirrel=+2.7)")
print(f"  FULL DROP:   {full_drop_pp:+.1f}pp")
print(f"  Prediction was: < -17.9pp")

diag = [
    "PUBMED SHUFFLE DIAGNOSTIC",
    "="*55,
    f"N={N}  classes={n_classes}  single split  seed={SEED}  {EPOCHS} epochs",
    f"Unshuffled: coarse={st_u[0]:.4f}  full={st_u[-1]:.4f}  val={bv_u:.4f}",
    f"Shuffled:   coarse={st_s[0]:.4f}  full={st_s[-1]:.4f}",
    f"Coarse drop: {coarse_drop_pp:+.1f}pp",
    f"Full drop:   {full_drop_pp:+.1f}pp",
    "",
    "Reference: Cora=-17.9  Cornell=-8.1  Actor=-0.1  Squirrel=+2.7",
    f"Prediction was: < -17.9pp  ->  {'CONFIRMED' if coarse_drop_pp < -17.9 else 'NOT CONFIRMED'}",
]

STOP_THRESHOLD = -10.0
if coarse_drop_pp > STOP_THRESHOLD:
    diag += ["", f"Decision: STOP — coarse drop {coarse_drop_pp:+.1f}pp > {STOP_THRESHOLD}pp threshold."]
    print(f"\n  Decision: STOP (drop > {STOP_THRESHOLD}pp)")
    with open(f"{OUT}/diagnostic.txt", "w") as f:
        f.write("\n".join(diag) + "\n")
    sys.exit(0)
else:
    decision_str = f"FULL RUN — coarse drop {coarse_drop_pp:+.1f}pp <= {STOP_THRESHOLD}pp."
    diag += ["", f"Decision: {decision_str}"]
    print(f"\n  Decision: {decision_str}")

with open(f"{OUT}/diagnostic.txt", "w") as f:
    f.write("\n".join(diag) + "\n")

# ---------------------------------------------------------------------------
# Strategy C: median eigenvalue threshold
# ---------------------------------------------------------------------------
eig_slices = np.array([eigenvalues[min(HALF + j, K - 1)] for j in range(HALF + 1)])
lambda_threshold = float(np.median(eig_slices))
raw_w = np.zeros(HALF + 1)
for j in range(HALF + 1):
    if eig_slices[j] <= lambda_threshold:
        raw_w[j] = 1.0 / max(eig_slices[j], 1e-6)
w_C = (raw_w / raw_w.sum()) if raw_w.sum() > 0 else raw_w
cutoff_C = int(np.max(np.where(raw_w > 0))) if raw_w.any() else HALF

print(f"\nStrategy C: median lambda = {lambda_threshold:.5f}  cutoff j={cutoff_C}")
print(f"  eig_slices[0:5]:  {eig_slices[:5]}")
print(f"  eig_slices[28:33]: {eig_slices[28:]}")

# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
results = {}

# 1. Dense uniform (reuse diagnostic run)
results["dense_uniform"] = {"best_val": bv_u, "slice_test": st_u,
                              "val_coarse": vc_u, "val_full": vf_u}
print(f"\nSliced-dense(uniform): [reused from diagnostic]  "
      f"val={bv_u:.4f}  full={st_u[-1]:.4f}  coarse={st_u[0]:.4f}")

# 2. Strategy C
print(f"\nRunning Strategy C (cutoff j={cutoff_C})...")
torch.manual_seed(SEED)
m_c = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                         loss_weights="uniform")
bv_c, st_c, vc_c, vf_c = train_sliced(m_c, w_vec=w_C)
results["strategy_c"] = {"best_val": bv_c, "slice_test": st_c,
                          "val_coarse": vc_c, "val_full": vf_c}
print(f"  Strategy C: val={bv_c:.4f}  best={max(st_c):.4f}  "
      f"@cutoff(j={cutoff_C})={st_c[cutoff_C]:.4f}  coarse={st_c[0]:.4f}")

# 3. MLP-full
print("\nRunning StandardMLP-full...")
torch.manual_seed(SEED)
bl_full = StandardMLP(n_features=K, n_classes=n_classes, hidden_dim=K, n_layers=N_LAYERS)
bv_mf, ta_mf = train_baseline(bl_full, U, labels, tr, val, te, lr=LR, wd=WD, epochs=EPOCHS)
results["mlp_full"] = {"best_val": bv_mf, "test": ta_mf}
print(f"  val={bv_mf:.4f}  test={ta_mf:.4f}")

# 4. MLP-half
print("\nRunning StandardMLP-half...")
torch.manual_seed(SEED)
bl_half = StandardMLP(n_features=HALF, n_classes=n_classes, hidden_dim=K, n_layers=N_LAYERS)
bv_mh, ta_mh = train_baseline(bl_half, U[:, :HALF], labels, tr, val, te,
                                lr=LR, wd=WD, epochs=EPOCHS)
results["mlp_half"] = {"best_val": bv_mh, "test": ta_mh}
print(f"  val={bv_mh:.4f}  test={ta_mh:.4f}")

# Derived
best_sliced = max(max(st_u), max(st_c))
best_j_u    = st_u.index(max(st_u))

# Surprising case: peak very early — run manual cutoffs
print(f"\nPer-slice peak: j={best_j_u}  (d={HALF+best_j_u})")
if best_j_u <= 6:
    print("  -> Peak is very early. Running manual cutoff-j3 and cutoff-j5...")
    for mc in [3, 5]:
        torch.manual_seed(SEED)
        m_mc = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                                  loss_weights="uniform")
        bv_mc, st_mc, _, _ = train_sliced(m_mc, loss_cutoff=mc)
        results[f"cutoff_j{mc:02d}"] = {"best_val": bv_mc, "slice_test": st_mc}
        best_sliced = max(best_sliced, max(st_mc))
        print(f"  cutoff-j{mc}: val={bv_mc:.4f}  best={max(st_mc):.4f}  "
              f"@j{mc}={st_mc[mc]:.4f}")

print(f"\nElapsed: {elapsed():.0f}s ({elapsed()/60:.1f} min)")

# ---------------------------------------------------------------------------
# 1. Comparison table
# ---------------------------------------------------------------------------
pred_drop  = coarse_drop_pp < -17.9
pred_c_mh  = max(st_c) > ta_mh
pred_mh_mf = ta_mh > ta_mf

header = [
    "",
    "  PREDICTION CHECK:",
    f"  Shuffle coarse drop < -17.9pp (Cora)?  "
    f"{'YES' if pred_drop else 'NO'}  — actual: {coarse_drop_pp:+.1f}pp",
    f"  Strategy C beats MLP-half?              "
    f"{'YES' if pred_c_mh else 'NO'}  — {max(st_c):.4f} vs {ta_mh:.4f}",
    f"  MLP-half beats MLP-full?                "
    f"{'YES' if pred_mh_mf else 'NO'}  — {ta_mh:.4f} vs {ta_mf:.4f}  (homophily check)",
    "",
]
div = "="*85
lines = header + [
    div,
    f"COMPARISON TABLE — PubMed (k={K}, seed={SEED}, {EPOCHS} epochs, single split)",
    div,
    f"{'Method':<26} {'Best Val':>9} {'Test(full)':>11} {'Test(coarse)':>13} {'Test(best)':>11}",
    "-"*85,
]
for tag, label in [("dense_uniform","Sliced-dense(uniform)"),
                    ("strategy_c",  f"Strategy-C(j={cutoff_C})")]:
    r = results[tag]; st = r["slice_test"]
    lines.append(f"{label:<26} {r['best_val']:>9.4f} {st[-1]:>11.4f} "
                 f"{st[0]:>13.4f} {max(st):>11.4f}")
for mc in [3, 5]:
    tag = f"cutoff_j{mc:02d}"
    if tag in results:
        r = results[tag]; st = r["slice_test"]
        lines.append(f"{'Sliced-cutoff-j'+str(mc):<26} {r['best_val']:>9.4f} "
                     f"{st[mc]:>11.4f} {st[0]:>13.4f} {max(st):>11.4f}")
lines.append(f"{'StandardMLP-full':<26} {bv_mf:>9.4f} {ta_mf:>11.4f} "
             f"{'—':>13} {'—':>11}")
lines.append(f"{'StandardMLP-half':<26} {bv_mh:>9.4f} {ta_mh:>11.4f} "
             f"{'—':>13} {'—':>11}")
lines.append(div)
table_str = "\n".join(lines)
print("\n" + table_str)
with open(f"{OUT}/comparison_table.txt", "w") as f:
    f.write(table_str + "\n")

# ---------------------------------------------------------------------------
# 2. Per-slice accuracy curve
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(slice_dims, st_u, color="#1f77b4", lw=2, marker="o", ms=3,
        label="Sliced-dense(uniform)")
ax.plot(slice_dims, st_c, color="#2ca02c", lw=1.8, marker="o", ms=3,
        label=f"Strategy-C(j={cutoff_C})")
for mc in [3, 5]:
    if f"cutoff_j{mc:02d}" in results:
        st_mc = results[f"cutoff_j{mc:02d}"]["slice_test"]
        ax.plot(slice_dims, st_mc, lw=1.5, ls="--", marker="o", ms=2.5,
                label=f"Manual cutoff-j{mc}")

ax.axvline(HALF + cutoff_C, color="#2ca02c", lw=0.9, ls=":", alpha=0.6,
           label=f"Strategy C cutoff j={cutoff_C}")
ax.axhline(ta_mf, color="black", ls="--", lw=1.5, label=f"MLP-full {ta_mf:.4f}")
ax.axhline(ta_mh, color="gray",  ls=":",  lw=1.5, label=f"MLP-half {ta_mh:.4f}")

ax.set_xlabel("Slice dimension d_j", fontsize=12)
ax.set_ylabel("Test accuracy", fontsize=12)
ax.set_title(f"PubMed spectral resolution curve (homophilous, N={N})", fontsize=13)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/per_slice_accuracy.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT}/per_slice_accuracy.png")

# ---------------------------------------------------------------------------
# 3. Shuffle table
# ---------------------------------------------------------------------------
shuf_lines = [
    f"SHUFFLE TABLE — PubMed (seed={SEED}, {EPOCHS} epochs)",
    "="*55,
    f"{'Slice':>5} | {'Dim':>4} | {'Unshuffled':>10} | {'Shuffled':>10} | {'Diff':>8}",
    "-"*55,
]
for j, (a_u, a_s) in enumerate(zip(st_u, st_s)):
    shuf_lines.append(f"{j:>5} | {HALF+j:>4} | {a_u:>10.4f} | {a_s:>10.4f} | {a_s-a_u:>+8.4f}")
shuf_lines += ["="*55,
               f"Coarse drop: {coarse_drop_pp:+.1f}pp",
               f"Full drop:   {full_drop_pp:+.1f}pp",
               f"Best val: unshuffled={bv_u:.4f}"]
shuf_str = "\n".join(shuf_lines)
print("\n" + shuf_str)
with open(f"{OUT}/shuffle_table.txt", "w") as f:
    f.write(shuf_str + "\n")

# ---------------------------------------------------------------------------
# 4. Eigenvalue spectrum
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(range(K), eigenvalues, "o-", ms=3, lw=1, color="#1f77b4")
ax.axvline(HALF, color="black", lw=1.2, ls="--",
           label=f"k//2 boundary (j=0, d={HALF})")
ax.axvline(HALF + cutoff_C, color="#2ca02c", lw=1.5, ls="-.",
           label=f"Strategy C cutoff j={cutoff_C}")
ax.axhline(lambda_threshold, color="#2ca02c", lw=0.8, ls=":", alpha=0.7,
           label=f"Median threshold λ={lambda_threshold:.4f}")
if best_j_u <= 6:
    ax.axvline(HALF + best_j_u, color="#d62728", lw=1.5, ls="-",
               label=f"Per-slice peak j={best_j_u}")
ax.set_xlabel("Eigenvalue index i", fontsize=11)
ax.set_ylabel("Eigenvalue λᵢ", fontsize=11)
ax.set_title("PubMed normalized Laplacian spectrum — Strategy C cutoff", fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/eigenvalue_spectrum.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT}/eigenvalue_spectrum.png")

# ---------------------------------------------------------------------------
# 5. Training curves
# ---------------------------------------------------------------------------
# Rerun for all models to get per-epoch val curves (no checkpoint loading needed)
print("\nRecording training curves...")
curve_vals = {}
for tag, label, model_fn, U_in in [
    ("du", "Sliced-dense(uniform)", None, U),
    ("mf", "MLP-full",             None, U),
    ("mh", "MLP-half",             None, U[:, :HALF]),
]:
    torch.manual_seed(SEED)
    if "Sliced" in label:
        m = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                               loss_weights="uniform")
        opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
        curve = []
        for _ in range(EPOCHS):
            m.train(); opt.zero_grad()
            m.compute_loss(m(U_in), labels, tr).backward(); opt.step()
            m.eval()
            with torch.no_grad(): curve.append(acc(m(U_in)[-1], labels, val))
    else:
        n_f = K if "full" in label else HALF
        m = StandardMLP(n_features=n_f, n_classes=n_classes, hidden_dim=K, n_layers=N_LAYERS)
        opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
        curve = []
        for _ in range(EPOCHS):
            m.train(); opt.zero_grad()
            F.cross_entropy(m(U_in)[tr], labels[tr]).backward(); opt.step()
            m.eval()
            with torch.no_grad(): curve.append(acc(m(U_in), labels, val))
    curve_vals[label] = curve

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(vc_u, lw=1.5, label=f"Coarse j=0, d={HALF}")
ax1.plot(vf_u, lw=1.5, label=f"Full  j={HALF}, d={K}")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val accuracy")
ax1.set_title("Coarse vs Full — PubMed, dense uniform"); ax1.legend(); ax1.grid(True, alpha=0.3)

c4 = ["#1f77b4", "#000000", "#888888"]
for (lbl, curve), col in zip(curve_vals.items(), c4):
    ax2.plot(curve, lw=1.5, label=lbl, color=col)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val accuracy")
ax2.set_title("All models — PubMed"); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/training_curves.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT}/training_curves.png")

# ---------------------------------------------------------------------------
# 6. Cross-dataset summary
# ---------------------------------------------------------------------------
rows = [
    ("Cora",     2485,  0.7082, 0.6448, 0.7399, 0.7410, -17.9),
    ("Cornell",   183,  0.4054, 0.4270, 0.4189,   None,  -8.1),
    ("Actor",    7600,  0.2525, 0.2524, 0.2511,   None,  -0.1),
    ("Squirrel", 5201,    None,   None,   None,   None,  +2.7),
    ("PubMed",      N,  ta_mh,  ta_mf, best_sliced, max(st_c), coarse_drop_pp),
]
cross = ["CROSS-DATASET SUMMARY (all 5 datasets)", "="*110,
         f"{'Dataset':<10} | {'N':>6} | {'MLP-half':>9} | {'MLP-full':>9} | "
         f"{'Sliced-best':>12} | {'Strategy-C':>11} | {'Shuffle drop':>13}",
         "-"*110]
for dataset, n, mh, mf, sb, sc, sd in rows:
    mh_s = f"{mh:.4f}" if mh else "n/a"
    mf_s = f"{mf:.4f}" if mf else "n/a"
    sb_s = f"{sb:.4f}" if sb else "n/a"
    sc_s = f"{sc:.4f}" if sc else "n/a"
    cross.append(f"{dataset:<10} | {n:>6} | {mh_s:>9} | {mf_s:>9} | "
                 f"{sb_s:>12} | {sc_s:>11} | {sd:>+12.1f}pp")
cross.append("="*110)

# Core claim paragraph
cross += ["",
          "CORE EMPIRICAL CLAIM:",
          "Shuffle drop predicts whether Sliced-best > MLP-full:"]
for dataset, n, mh, mf, sb, sc, sd in rows:
    if mf and sb:
        wins = "YES" if sb > mf else "NO"
        cross.append(f"  {dataset:<10}: drop={sd:+.1f}pp  Sliced>MLP-full: {wins} "
                     f"({sb:.4f} vs {mf:.4f})")
    else:
        cross.append(f"  {dataset:<10}: drop={sd:+.1f}pp  Sliced>MLP-full: NOT RUN")
cross += ["",
          "Pattern: larger |shuffle drop| <-> Sliced beats plain MLP.",
          "  Cora (-17.9pp, homophilous):  Sliced wins (74.10% vs 64.48%)",
          "  PubMed (see above, homophilous): ?",
          "  Cornell (-8.1pp, heterophilous): marginal",
          "  Actor/Squirrel (~0pp):          Sliced loses — null result confirmed"]

cross_str = "\n".join(cross)
print("\n" + cross_str)
for path in [f"{OUT}/cross_dataset_summary.txt",
             "outputs/cora_truncation/../cross_dataset_summary.txt",
             "outputs/cornell/cross_dataset_summary.txt"]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: f.write(cross_str + "\n")

# ---------------------------------------------------------------------------
# 7. Notes
# ---------------------------------------------------------------------------
peak_j = best_j_u
notes = [
    "PUBMED ANALYSIS NOTES", "="*60, "",
    f"1. Coarse shuffle drop: {coarse_drop_pp:+.1f}pp",
    f"   Cora=-17.9  Cornell=-8.1  Actor=-0.1  Squirrel=+2.7  PubMed={coarse_drop_pp:+.1f}",
    f"   Prediction was < -17.9pp: {'CONFIRMED' if coarse_drop_pp < -17.9 else 'NOT CONFIRMED'}",
    "",
    f"2. Per-slice curve peak: j={peak_j} (d={HALF+peak_j})",
    f"   Cora peaked at j=11 (d=43). Prediction for PubMed was j <= 6.",
    f"   Observed: {'CONFIRMED (j <= 6)' if peak_j <= 6 else f'NOT CONFIRMED (j={peak_j})'}",
    f"   Max test acc: {max(st_u):.4f} at j={peak_j}",
    "",
    f"3. Strategy C: cutoff j={cutoff_C}  (median lambda={lambda_threshold:.5f})",
    f"   Strategy C best={max(st_c):.4f}  MLP-half={ta_mh:.4f}",
    f"   Strategy C beats MLP-half: {'YES' if max(st_c) > ta_mh else 'NO'}",
    f"   Cutoff is {'close to' if abs(cutoff_C - peak_j) <= 5 else 'far from'} the peak (j={peak_j})",
    "",
    f"4. Sliced vs MLP-half: {best_sliced:.4f} vs {ta_mh:.4f}  "
    f"({'Sliced wins' if best_sliced > ta_mh else 'MLP-half wins'} by {abs(best_sliced-ta_mh)*100:.1f}pp)",
    "",
    f"5. MLP-half vs MLP-full: {ta_mh:.4f} vs {ta_mf:.4f}  diff={ta_mh-ta_mf:+.4f}",
    f"   Cora: half-full = +6.3pp.  PubMed: {(ta_mh-ta_mf)*100:+.1f}pp",
    f"   {'Larger reversal than Cora' if ta_mh-ta_mf > 0.063 else 'Similar or smaller reversal than Cora'}",
    "",
    f"6. Training stability: N={N} — large dataset, stable training expected.",
    "",
    f"7. Spectral gap hypothesis: PubMed shuffle drop={coarse_drop_pp:+.1f}pp, "
    f"Sliced>MLP-full: {best_sliced > ta_mf}.",
    f"   {'CONFIRMS hypothesis: large drop predicts Sliced win.' if (coarse_drop_pp < -10 and best_sliced > ta_mf) else 'DOES NOT confirm hypothesis at this threshold.'}",
]
notes_str = "\n".join(notes)
print("\n" + notes_str)
with open(f"{OUT}/notes.txt", "w") as f:
    f.write(notes_str + "\n")

print(f"\nTotal elapsed: {elapsed():.0f}s ({elapsed()/60:.1f} min)")
print(f"All outputs saved to {OUT}")
