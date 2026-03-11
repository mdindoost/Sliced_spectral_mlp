"""
Actor (and optionally Squirrel) experiment pipeline.
Stops and reports after Actor if total runtime > 10 minutes.

Outputs: outputs/actor/  and  outputs/squirrel/
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loaders import load_dataset
from src.models.sliced_mlp import SlicedSpectralMLP
from src.models.baselines import StandardMLP, train_baseline

WALL_START = time.time()
WALL_LIMIT  = 10 * 60   # 10 minutes hard stop
ACTOR_LIMIT =  5 * 60   # run Squirrel only if Actor finishes before this

SEED      = 42
K         = 64
N_LAYERS  = 2
LR        = 0.01
WD        = 5e-4
EPOCHS    = 100          # 200 is too slow; user approved 100 for large datasets
N_SPLITS  = 10
SPARSE_DIMS = [32, 40, 48, 56, 64]   # 5-slice sparse variant


def elapsed():
    return time.time() - WALL_START


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


def train_sliced(model, U, labels, tr, val, te,
                 heatmap_epoch=None, heatmap_dir=None):
    """Returns (best_val, slice_test_accs, val_coarse_curve, val_full_curve)."""
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best_val, best_state = 0.0, None
    val_coarse, val_full = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train(); opt.zero_grad()
        lg = model(U)
        model.compute_loss(lg, labels, tr).backward()

        if heatmap_epoch and epoch == heatmap_epoch and model.W[0].grad is not None:
            save_heatmap(torch.abs(model.W[0].grad), epoch, heatmap_dir)

        opt.step()
        model.eval()
        with torch.no_grad():
            lg = model(U)
        vc = acc(lg[0],  labels, val)
        vf = acc(lg[-1], labels, val)
        val_coarse.append(vc); val_full.append(vf)
        if vf > best_val:
            best_val = vf
            best_state = {k2: v.clone() for k2, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        lg = model(U)
    slice_test = [acc(logit, labels, te) for logit in lg]
    return best_val, slice_test, val_coarse, val_full


def run_dataset(name, out_dir):
    """Run all models on one dataset. Returns results dict."""
    os.makedirs(f"{out_dir}/grad_heatmaps", exist_ok=True)
    t_ds_start = time.time()

    # ----------------------------------------------------------------
    # Load eigenvectors once (split-independent)
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Loading {name.upper()} LCC eigenvectors...")
    U, labels0, _, _, _, eigenvalues = load_dataset(name, k=K, split_idx=0)
    N = U.shape[0]
    n_classes = int(labels0.max().item()) + 1
    print(f"  N={N}  classes={n_classes}  k={K}")
    print(f"  Eigenvectors loaded in {time.time()-t_ds_start:.1f}s")

    # Load all 10 splits
    splits = []
    for s in range(N_SPLITS):
        _, ls, tr, val, te, _ = load_dataset(name, k=K, split_idx=s)
        assert (ls == labels0).all()
        splits.append((tr, val, te))
    labels = labels0
    print(f"  All {N_SPLITS} splits loaded. Elapsed: {elapsed():.0f}s")

    # ----------------------------------------------------------------
    # Containers
    # ----------------------------------------------------------------
    res = {
        "dense_uniform":    {"best_val": [], "slice_test": []},
        "sparse_uniform":   {"best_val": [], "slice_test": []},
        "dense_eigenvalue": {"best_val": [], "slice_test": []},
        "mlp_full":         {"best_val": [], "test": []},
        "mlp_half":         {"best_val": [], "test": []},
    }
    curves_uniform = None   # (val_coarse, val_full) from split 0

    # ----------------------------------------------------------------
    # Training loop over splits
    # ----------------------------------------------------------------
    for si, (tr, val, te) in enumerate(splits):
        print(f"\n--- Split {si} | train={tr.sum()} val={val.sum()} test={te.sum()} | elapsed={elapsed():.0f}s ---")

        # Dense uniform
        torch.manual_seed(SEED + si)
        m = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                               loss_weights="uniform")
        hm_epoch = 50 if si == 0 else None
        bv, st, vc, vf = train_sliced(m, U, labels, tr, val, te,
                                       heatmap_epoch=hm_epoch,
                                       heatmap_dir=f"{out_dir}/grad_heatmaps")
        res["dense_uniform"]["best_val"].append(bv)
        res["dense_uniform"]["slice_test"].append(st)
        if si == 0: curves_uniform = (vc, vf)
        print(f"  dense_uniform:    val={bv:.4f}  test_full={st[-1]:.4f}  test_coarse={st[0]:.4f}")

        # Sparse uniform
        torch.manual_seed(SEED + si)
        m = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                               loss_weights="uniform",
                               custom_slice_dims=SPARSE_DIMS)
        bv, st, _, _ = train_sliced(m, U, labels, tr, val, te)
        res["sparse_uniform"]["best_val"].append(bv)
        res["sparse_uniform"]["slice_test"].append(st)
        print(f"  sparse_uniform:   val={bv:.4f}  test_full={st[-1]:.4f}  test_coarse={st[0]:.4f}")

        # Dense eigenvalue
        torch.manual_seed(SEED + si)
        m = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                               loss_weights="eigenvalue", eigenvalues=eigenvalues)
        bv, st, _, _ = train_sliced(m, U, labels, tr, val, te)
        res["dense_eigenvalue"]["best_val"].append(bv)
        res["dense_eigenvalue"]["slice_test"].append(st)
        print(f"  dense_eigenvalue: val={bv:.4f}  test_full={st[-1]:.4f}")

        # MLP-full
        torch.manual_seed(SEED + si)
        bl = StandardMLP(n_features=K, n_classes=n_classes, hidden_dim=K, n_layers=N_LAYERS)
        bv, ta = train_baseline(bl, U, labels, tr, val, te, lr=LR, wd=WD, epochs=EPOCHS)
        res["mlp_full"]["best_val"].append(bv); res["mlp_full"]["test"].append(ta)
        print(f"  mlp_full:         val={bv:.4f}  test={ta:.4f}")

        # MLP-half
        torch.manual_seed(SEED + si)
        bl = StandardMLP(n_features=K//2, n_classes=n_classes, hidden_dim=K, n_layers=N_LAYERS)
        bv, ta = train_baseline(bl, U[:, :K//2], labels, tr, val, te, lr=LR, wd=WD, epochs=EPOCHS)
        res["mlp_half"]["best_val"].append(bv); res["mlp_half"]["test"].append(ta)
        print(f"  mlp_half:         val={bv:.4f}  test={ta:.4f}")

    # ----------------------------------------------------------------
    # Shuffle comparison — split 0 only
    # ----------------------------------------------------------------
    tr0, val0, te0 = splits[0]
    torch.manual_seed(SEED)
    m_u = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS, loss_weights="uniform")
    bv_u, st_u, _, _ = train_sliced(m_u, U, labels, tr0, val0, te0)

    torch.manual_seed(SEED)
    perm = torch.randperm(K, generator=torch.Generator().manual_seed(SEED))
    U_shuf = U[:, perm]
    m_s = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS, loss_weights="uniform")
    bv_s, st_s, _, _ = train_sliced(m_s, U_shuf, labels, tr0, val0, te0)

    # ----------------------------------------------------------------
    # Aggregate
    # ----------------------------------------------------------------
    n_slices_dense  = K // 2 + 1
    slice_dims_dense = [K // 2 + j for j in range(n_slices_dense)]

    for tag in ["dense_uniform", "dense_eigenvalue"]:
        arr = np.array(res[tag]["slice_test"])
        res[tag]["mean"] = arr.mean(0)
        res[tag]["std"]  = arr.std(0)

    # sparse: collapse to single test-acc per slice position
    res["sparse_uniform"]["mean"] = np.mean(res["sparse_uniform"]["slice_test"], axis=0)
    res["sparse_uniform"]["std"]  = np.std( res["sparse_uniform"]["slice_test"], axis=0)

    # ----------------------------------------------------------------
    # 1. Comparison table
    # ----------------------------------------------------------------
    def ms(arr): m=np.mean(arr); s=np.std(arr); return f"{m:.4f}±{s:.4f}"

    lines = []
    div = "=" * 110
    lines.append(div)
    lines.append(f"COMPARISON TABLE — {name.upper()}  (k={K}, {EPOCHS} epochs, mean over {N_SPLITS} splits)")
    lines.append(div)
    lines.append(f"{'Method':<30} {'Best Val':>16}  {'Test (full)':>14}  {'Test (coarse)':>15}  {'Test (best slice)':>18}")
    lines.append("-" * 110)

    for tag, label in [
        ("dense_uniform",    "Sliced-dense(uniform)"),
        ("sparse_uniform",   "Sliced-sparse(uniform)"),
        ("dense_eigenvalue", "Sliced-dense(eigenvalue)"),
    ]:
        arr_st = np.array(res[tag]["slice_test"])
        bv_arr = np.array(res[tag]["best_val"])
        tf = arr_st[:, -1]; tc = arr_st[:, 0]; tb = arr_st.max(1)
        lines.append(f"{label:<30} {ms(bv_arr):>16}  {ms(tf):>14}  {ms(tc):>15}  {ms(tb):>18}")

    for tag, label in [("mlp_full","StandardMLP-full"), ("mlp_half","StandardMLP-half")]:
        bv_arr = np.array(res[tag]["best_val"]); ta_arr = np.array(res[tag]["test"])
        lines.append(f"{label:<30} {ms(bv_arr):>16}  {ms(ta_arr):>14}  {ms(ta_arr):>15}  {ms(ta_arr):>18}")

    lines.append(div)
    table_str = "\n".join(lines)
    print("\n" + table_str)
    with open(f"{out_dir}/comparison_table.txt", "w") as f:
        f.write(table_str + "\n")

    # ----------------------------------------------------------------
    # 2. Per-slice accuracy curve
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"dense_uniform": "#1f77b4", "dense_eigenvalue": "#2ca02c"}
    for tag, label in [("dense_uniform","Sliced-dense(uniform)"),
                        ("dense_eigenvalue","Sliced-dense(eigenvalue)")]:
        ax.plot(slice_dims_dense, res[tag]["mean"], marker="o", ms=2.5,
                color=colors[tag], label=label)
        ax.fill_between(slice_dims_dense,
                        res[tag]["mean"] - res[tag]["std"],
                        res[tag]["mean"] + res[tag]["std"],
                        alpha=0.12, color=colors[tag])

    # Sparse: scatter points at 5 positions
    ax.scatter(SPARSE_DIMS, res["sparse_uniform"]["mean"],
               marker="D", s=60, color="#ff7f0e", zorder=5,
               label="Sliced-sparse(uniform)")
    for d_j, m_val, s_val in zip(SPARSE_DIMS,
                                   res["sparse_uniform"]["mean"],
                                   res["sparse_uniform"]["std"]):
        ax.errorbar(d_j, m_val, yerr=s_val, fmt="none",
                    color="#ff7f0e", capsize=3, lw=1.2)

    mf = np.mean(res["mlp_full"]["test"]); mh = np.mean(res["mlp_half"]["test"])
    ax.axhline(mf, color="black", ls="--", lw=1.5,
               label=f"StandardMLP-full (d={K})  {mf:.3f}")
    ax.axhline(mh, color="gray",  ls=":",  lw=1.5,
               label=f"StandardMLP-half (d={K//2})  {mh:.3f}")
    ax.set_xlabel("Slice dimension  d_j  (# eigenvectors)", fontsize=12)
    ax.set_ylabel("Test accuracy (mean ± std over 10 splits)", fontsize=12)
    ax.set_title(f"Spectral resolution curve — {name.capitalize()} (heterophilous)", fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/per_slice_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir}/per_slice_accuracy.png")

    # ----------------------------------------------------------------
    # 3. Shuffle table
    # ----------------------------------------------------------------
    shuf_lines = []
    shuf_lines.append(f"SHUFFLE COMPARISON — {name.upper()}  (split 0, seed={SEED})")
    shuf_lines.append("=" * 55)
    shuf_lines.append(f"{'Slice':>5} | {'Dim':>4} | {'Unshuffled':>10} | {'Shuffled':>10} | {'Diff':>8}")
    shuf_lines.append("-" * 55)
    for j, (a_u, a_s) in enumerate(zip(st_u, st_s)):
        d = K // 2 + j
        shuf_lines.append(f"{j:>5} | {d:>4} | {a_u:>10.4f} | {a_s:>10.4f} | {a_s-a_u:>+8.4f}")
    shuf_lines.append("=" * 55)
    shuf_lines.append(f"Best val  — unshuffled: {bv_u:.4f}  shuffled: {bv_s:.4f}  diff: {bv_s-bv_u:+.4f}")
    shuf_lines.append(f"Coarse (j=0, d={K//2}) — unshuffled: {st_u[0]:.4f}  shuffled: {st_s[0]:.4f}  diff: {st_s[0]-st_u[0]:+.4f}")
    shuf_lines.append(f"Full   (j={K//2}, d={K}) — unshuffled: {st_u[-1]:.4f}  shuffled: {st_s[-1]:.4f}  diff: {st_s[-1]-st_u[-1]:+.4f}")
    shuf_str = "\n".join(shuf_lines)
    print("\n" + shuf_str)
    with open(f"{out_dir}/shuffle_table.txt", "w") as f:
        f.write(shuf_str + "\n")

    # ----------------------------------------------------------------
    # 4. Training curves
    # ----------------------------------------------------------------
    # Need per-epoch val curves for all models on split 0
    print("\nRecording val curves (split 0) for training_curves.png...")
    tr0, val0, te0 = splits[0]
    all_curves = {}

    for tag, label, sdims in [
        ("dense_uniform",  "Sliced-dense(uniform)",  None),
        ("sparse_uniform", "Sliced-sparse(uniform)", SPARSE_DIMS),
    ]:
        torch.manual_seed(SEED)
        m = SlicedSpectralMLP(k=K, n_classes=n_classes, n_layers=N_LAYERS,
                               loss_weights="uniform",
                               custom_slice_dims=sdims)
        opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
        curve = []
        for _ in range(EPOCHS):
            m.train(); opt.zero_grad()
            m.compute_loss(m(U), labels, tr0).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                curve.append(acc(m(U)[-1], labels, val0))
        all_curves[label] = curve

    for tag, label, n_feat in [("mlp_full","StandardMLP-full",K),
                                 ("mlp_half","StandardMLP-half",K//2)]:
        torch.manual_seed(SEED)
        X = U[:, :n_feat]
        bl = StandardMLP(n_features=n_feat, n_classes=n_classes, hidden_dim=K, n_layers=N_LAYERS)
        opt = torch.optim.Adam(bl.parameters(), lr=LR, weight_decay=WD)
        curve = []
        for _ in range(EPOCHS):
            bl.train(); opt.zero_grad()
            F.cross_entropy(bl(X)[tr0], labels[tr0]).backward(); opt.step()
            bl.eval()
            with torch.no_grad():
                curve.append(acc(bl(X), labels, val0))
        all_curves[label] = curve

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    vc, vf = curves_uniform
    ax1.plot(vc, lw=1.5, label=f"Coarse j=0, d={K//2}")
    ax1.plot(vf, lw=1.5, label=f"Full   j={K//2}, d={K}")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val accuracy")
    ax1.set_title(f"Coarse vs Full — {name.capitalize()}, dense uniform")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    c5 = ["#1f77b4","#ff7f0e","#000000","#888888"]
    for (lbl, curve), col in zip(all_curves.items(), c5):
        ax2.plot(curve, lw=1.5, label=lbl, color=col)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val accuracy")
    ax2.set_title(f"All models — {name.capitalize()}, split 0")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/training_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir}/training_curves.png")

    t_total = time.time() - t_ds_start
    print(f"\n{name.upper()} finished in {t_total:.0f}s ({t_total/60:.1f} min)")

    # Return key scalars for cross-dataset table
    return {
        "mlp_half":           np.mean(res["mlp_half"]["test"]),
        "mlp_full":           np.mean(res["mlp_full"]["test"]),
        "sliced_best_full":   max(np.mean(np.array(res[t]["slice_test"])[:, -1])
                                  for t in ["dense_uniform","sparse_uniform","dense_eigenvalue"]),
        "sliced_best_coarse": max(np.mean(np.array(res[t]["slice_test"])[:, 0])
                                  for t in ["dense_uniform","sparse_uniform","dense_eigenvalue"]),
        "shuffle_coarse_drop_pp": (st_s[0] - st_u[0]) * 100,
        "n_nodes": N,
    }


# ---------------------------------------------------------------------------
# Run Actor
# ---------------------------------------------------------------------------
actor_results = run_dataset("actor", "outputs/actor")

actor_elapsed = elapsed()
print(f"\nActor done. Total elapsed: {actor_elapsed:.0f}s ({actor_elapsed/60:.1f} min)")

# ---------------------------------------------------------------------------
# Run Squirrel (only if Actor finished fast enough)
# ---------------------------------------------------------------------------
squirrel_results = None
if actor_elapsed < ACTOR_LIMIT:
    print(f"\nActor finished in {actor_elapsed/60:.1f} min < 5 min — proceeding to Squirrel.")
    squirrel_results = run_dataset("squirrel", "outputs/squirrel")
else:
    print(f"\nActor took {actor_elapsed/60:.1f} min >= 5 min — skipping Squirrel.")

# ---------------------------------------------------------------------------
# Cross-dataset summary
# ---------------------------------------------------------------------------
rows = [
    ("Cora",     0.7082, 0.6448, 0.6743, 0.6918, -17.9, 2485),
    ("Cornell",  0.4054, 0.4270, 0.4189, 0.4243,  -8.1,  183),
]
for name, r in [("Actor", actor_results)] + \
               ([("Squirrel", squirrel_results)] if squirrel_results else []):
    rows.append((name,
                 r["mlp_half"], r["mlp_full"],
                 r["sliced_best_full"], r["sliced_best_coarse"],
                 r["shuffle_coarse_drop_pp"], r["n_nodes"]))

cross = []
cross.append("CROSS-DATASET SUMMARY")
cross.append("=" * 115)
cross.append(f"{'Dataset':<10} | {'N nodes':>8} | {'MLP-half':>9} | {'MLP-full':>9} | "
             f"{'Sliced best-full':>17} | {'Sliced best-coarse':>19} | {'Shuffle coarse drop':>20}")
cross.append("-" * 115)
for dataset, mh, mf, sbf, sbc, scd, n in rows:
    cross.append(f"{dataset:<10} | {n:>8} | {mh:>9.4f} | {mf:>9.4f} | "
                 f"{sbf:>17.4f} | {sbc:>19.4f} | {scd:>+19.1f}pp")
cross.append("=" * 115)
cross.append("")
cross.append("Notes:")
cross.append("  Cora/Cornell: single seed per split, 200 epochs")
cross.append("  Actor/Squirrel: single seed per split, 100 epochs, mean over 10 splits")

cross_str = "\n".join(cross)
print("\n" + cross_str)

# Save to both actor dir and root
for path in ["outputs/actor/cross_dataset_summary.txt",
             "outputs/cornell/cross_dataset_summary.txt"]:
    with open(path, "w") as f:
        f.write(cross_str + "\n")
if squirrel_results:
    with open("outputs/squirrel/cross_dataset_summary.txt", "w") as f:
        f.write(cross_str + "\n")

print(f"\nTotal wall time: {elapsed():.0f}s ({elapsed()/60:.1f} min)")
print("Done.")
