"""
Row normalisation experiments for SlicedSpectralMLP.

Experiments
-----------
  A  Shuffle diagnostic: original U vs row-normalised U_tilde
  B  Per-slice curve: Sliced (orig) vs Sliced (row-norm input)
  C  Full comparison table (7 models)
  D  Cutoff sensitivity: j* scan for orig vs row-norm input

Models
------
  StandardMLP-half / full   — existing baseline (with bias, no sphere norm)
  RowNormMLP-half / full     — no bias, row norm at input + after each layer
  Sliced-orig                — SlicedSpectralMLP, no row-norm input
  Sliced-rownorm             — SlicedSpectralMLP, use_row_norm_input=True
  Sliced-rownorm-nosph       — SlicedSpectralMLP, row-norm input + use_sphere_norm=False
  Sliced-stratC-orig         — Strategy C cutoff, original input
  Sliced-stratC-rownorm      — Strategy C cutoff (same j*), row-norm input

Usage
-----
  python scripts/run_rownorm.py [--dataset cora] [--epochs 200]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import load_dataset
from src.evaluation.metrics import accuracy, per_slice_accuracy
from src.evaluation.shuffle import run_shuffle_diagnostic
from src.models.sliced_mlp import SlicedSpectralMLP
from src.models.baselines import StandardMLP, RowNormMLP, train_baseline
from src.cutoff.strategy_c import select_cutoff as strategy_c_cutoff


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS = ["cora", "citeseer", "pubmed", "cornell"]
CUTOFF_SCAN = [0, 3, 7, 11, 15, 20, 31]

_SUMMARY_FIELDS = [
    "dataset", "model", "use_row_norm_input", "use_sphere_norm",
    "test_best", "test_coarse", "shuffle_drop", "optimal_j_star",
]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_sliced(
    model: SlicedSpectralMLP,
    U: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.01,
    wd: float = 5e-4,
    loss_cutoff: int | None = None,
) -> tuple[float, float, list[float]]:
    """
    Train SlicedSpectralMLP and return (best_val, test_full, slice_test_accs).

    When loss_cutoff is set, model selection uses val accuracy at slice
    loss_cutoff (last active slice) to avoid tracking undertrained dims.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val = 0.0
    best_state: dict | None = None
    select_j = loss_cutoff if loss_cutoff is not None else -1

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        lg = model(U)
        model.compute_loss(lg, labels, train_mask, loss_cutoff=loss_cutoff).backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            lg = model(U)
        v = accuracy(lg[select_j], labels, val_mask)
        if v > best_val:
            best_val = v
            best_state = {k: vv.clone() for k, vv in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        lg = model(U)
    slice_accs = [accuracy(l, labels, test_mask) for l in lg]
    return best_val, slice_accs[-1], slice_accs


def _make_sliced(k, n_classes, use_row_norm_input=False, use_sphere_norm=True, seed=42):
    torch.manual_seed(seed)
    return SlicedSpectralMLP(
        k=k, n_classes=n_classes, n_layers=2, loss_weights="uniform",
        use_row_norm_input=use_row_norm_input,
        use_sphere_norm=use_sphere_norm,
    )


def _run_baseline(name, X, labels, train_mask, val_mask, test_mask,
                  n_classes, k, seed=42, epochs=200, lr=0.01, wd=5e-4):
    n_feat = X.shape[1]
    torch.manual_seed(seed)
    if name.startswith("rownorm"):
        model = RowNormMLP(input_dim=n_feat, hidden_dim=k, output_dim=n_classes)
    else:
        model = StandardMLP(n_features=n_feat, n_classes=n_classes,
                            hidden_dim=k, n_layers=2)
    bv, ta = train_baseline(model, X, labels, train_mask, val_mask, test_mask,
                            lr=lr, wd=wd, epochs=epochs)
    return bv, ta


# ---------------------------------------------------------------------------
# Experiment A: shuffle diagnostic
# ---------------------------------------------------------------------------

def exp_a(U, U_tilde, labels, train_mask, val_mask, test_mask, k, n_classes,
          epochs, seed):
    """Compare shuffle diagnostic on original vs row-normalised input."""
    print("\n  [Exp A] Shuffle diagnostic — original U")
    diag_orig = run_shuffle_diagnostic(
        U, labels, train_mask, val_mask, test_mask,
        k=k, n_classes=n_classes, seed=seed, epochs=epochs,
    )

    print("  [Exp A] Shuffle diagnostic — row-norm U_tilde")
    diag_rn = run_shuffle_diagnostic(
        U_tilde, labels, train_mask, val_mask, test_mask,
        k=k, n_classes=n_classes, seed=seed, epochs=epochs,
    )

    return diag_orig, diag_rn


# ---------------------------------------------------------------------------
# Experiment B: per-slice curve
# ---------------------------------------------------------------------------

def exp_b(U, U_tilde, labels, train_mask, val_mask, test_mask, k, n_classes,
          epochs, lr, wd, seed):
    """Train Sliced (orig) and Sliced (row-norm input), record per-slice curves."""
    print("\n  [Exp B] Per-slice curve — original input")
    m_orig = _make_sliced(k, n_classes, use_row_norm_input=False, seed=seed)
    _, _, curve_orig = _train_sliced(
        m_orig, U, labels, train_mask, val_mask, test_mask,
        epochs=epochs, lr=lr, wd=wd,
    )

    print("  [Exp B] Per-slice curve — row-norm input")
    m_rn = _make_sliced(k, n_classes, use_row_norm_input=True, seed=seed)
    _, _, curve_rn = _train_sliced(
        m_rn, U, labels, train_mask, val_mask, test_mask,
        epochs=epochs, lr=lr, wd=wd,
    )

    j_star_orig = int(np.argmax(curve_orig))
    j_star_rn   = int(np.argmax(curve_rn))

    print(f"    orig  : j*={j_star_orig}  peak={max(curve_orig):.4f}")
    print(f"    rownorm: j*={j_star_rn}   peak={max(curve_rn):.4f}")

    return curve_orig, curve_rn, j_star_orig, j_star_rn


# ---------------------------------------------------------------------------
# Experiment C: full comparison table
# ---------------------------------------------------------------------------

def exp_c(U, U_tilde, labels, train_mask, val_mask, test_mask, k, n_classes,
          eigenvalues, epochs, lr, wd, seed):
    """Train all 7 model variants and record test accuracy."""
    results = {}

    # --- Standard MLP baselines ---
    print("\n  [Exp C] StandardMLP-half")
    bv, ta = _run_baseline("std_half", U[:, :k//2], labels, train_mask,
                            val_mask, test_mask, n_classes, k, seed, epochs, lr, wd)
    results["StandardMLP-half"] = {"bv": bv, "test": ta,
                                    "use_rn": False, "use_sph": True}

    print("  [Exp C] StandardMLP-full")
    bv, ta = _run_baseline("std_full", U, labels, train_mask,
                            val_mask, test_mask, n_classes, k, seed, epochs, lr, wd)
    results["StandardMLP-full"] = {"bv": bv, "test": ta,
                                    "use_rn": False, "use_sph": True}

    # --- RowNormMLP baselines ---
    print("  [Exp C] RowNormMLP-half")
    bv, ta = _run_baseline("rownorm_half", U[:, :k//2], labels, train_mask,
                            val_mask, test_mask, n_classes, k, seed, epochs, lr, wd)
    results["RowNormMLP-half"] = {"bv": bv, "test": ta,
                                   "use_rn": True, "use_sph": True}

    print("  [Exp C] RowNormMLP-full")
    bv, ta = _run_baseline("rownorm_full", U, labels, train_mask,
                            val_mask, test_mask, n_classes, k, seed, epochs, lr, wd)
    results["RowNormMLP-full"] = {"bv": bv, "test": ta,
                                   "use_rn": True, "use_sph": True}

    # --- Sliced variants ---
    print("  [Exp C] Sliced-orig (no row-norm)")
    m = _make_sliced(k, n_classes, use_row_norm_input=False,
                     use_sphere_norm=True, seed=seed)
    bv, ta, slices = _train_sliced(m, U, labels, train_mask, val_mask, test_mask,
                                    epochs=epochs, lr=lr, wd=wd)
    results["Sliced-orig"] = {"bv": bv, "test": max(slices),
                               "test_coarse": slices[0], "test_full": ta,
                               "use_rn": False, "use_sph": True,
                               "j_star": int(np.argmax(slices))}

    print("  [Exp C] Sliced-rownorm (row-norm input, sphere norm inside)")
    m = _make_sliced(k, n_classes, use_row_norm_input=True,
                     use_sphere_norm=True, seed=seed)
    bv, ta, slices = _train_sliced(m, U, labels, train_mask, val_mask, test_mask,
                                    epochs=epochs, lr=lr, wd=wd)
    results["Sliced-rownorm"] = {"bv": bv, "test": max(slices),
                                  "test_coarse": slices[0], "test_full": ta,
                                  "use_rn": True, "use_sph": True,
                                  "j_star": int(np.argmax(slices))}

    print("  [Exp C] Sliced-rownorm-nosph (row-norm input, NO sphere norm inside)")
    m = _make_sliced(k, n_classes, use_row_norm_input=True,
                     use_sphere_norm=False, seed=seed)
    bv, ta, slices = _train_sliced(m, U, labels, train_mask, val_mask, test_mask,
                                    epochs=epochs, lr=lr, wd=wd)
    results["Sliced-rownorm-nosph"] = {"bv": bv, "test": max(slices),
                                        "test_coarse": slices[0], "test_full": ta,
                                        "use_rn": True, "use_sph": False,
                                        "j_star": int(np.argmax(slices))}

    # --- Strategy C: same j* for both (eigenvalues unchanged by row norm) ---
    j_c = strategy_c_cutoff(eigenvalues, k)
    print(f"  [Exp C] Strategy C cutoff: j*={j_c} (same for orig and row-norm)")

    print("  [Exp C] Sliced-stratC-orig")
    m = _make_sliced(k, n_classes, use_row_norm_input=False,
                     use_sphere_norm=True, seed=seed)
    bv, ta, slices = _train_sliced(m, U, labels, train_mask, val_mask, test_mask,
                                    epochs=epochs, lr=lr, wd=wd, loss_cutoff=j_c)
    results["Sliced-stratC-orig"] = {"bv": bv, "test": max(slices),
                                      "test_coarse": slices[0], "test_full": ta,
                                      "use_rn": False, "use_sph": True,
                                      "j_star": j_c}

    print("  [Exp C] Sliced-stratC-rownorm")
    m = _make_sliced(k, n_classes, use_row_norm_input=True,
                     use_sphere_norm=True, seed=seed)
    bv, ta, slices = _train_sliced(m, U, labels, train_mask, val_mask, test_mask,
                                    epochs=epochs, lr=lr, wd=wd, loss_cutoff=j_c)
    results["Sliced-stratC-rownorm"] = {"bv": bv, "test": max(slices),
                                         "test_coarse": slices[0], "test_full": ta,
                                         "use_rn": True, "use_sph": True,
                                         "j_star": j_c}

    return results, j_c


# ---------------------------------------------------------------------------
# Experiment D: cutoff sensitivity scan
# ---------------------------------------------------------------------------

def exp_d(U, labels, train_mask, val_mask, test_mask, k, n_classes,
          cutoffs, epochs, lr, wd, seed, use_row_norm_input: bool):
    """Run truncated loss training for each cutoff in cutoffs list."""
    results = {}
    tag = "rownorm" if use_row_norm_input else "orig"
    for j_cut in cutoffs:
        j_cut_clamped = min(j_cut, k // 2)
        print(f"    [{tag}] j_cut={j_cut_clamped}")
        m = _make_sliced(k, n_classes, use_row_norm_input=use_row_norm_input,
                         use_sphere_norm=True, seed=seed)
        _, _, slices = _train_sliced(
            m, U, labels, train_mask, val_mask, test_mask,
            epochs=epochs, lr=lr, wd=wd, loss_cutoff=j_cut_clamped,
        )
        results[j_cut_clamped] = {
            "test_best":   round(max(slices), 4),
            "test_coarse": round(slices[0], 4),
            "j_star":      int(np.argmax(slices)),
        }
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _make_figures(dataset, out_dir,
                  diag_orig, diag_rn,
                  curve_orig, curve_rn,
                  cutoff_orig, cutoff_rn):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Figure 1: shuffle drop comparison
    fig, ax = plt.subplots(figsize=(5, 3))
    labels_bar = ["Original", "Row-norm"]
    drops = [diag_orig["coarse_drop_pp"], diag_rn["coarse_drop_pp"]]
    colors = ["#1f77b4", "#ff7f0e"]
    bars = ax.bar(labels_bar, drops, color=colors, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Coarse-slice drop under shuffling (pp)", fontsize=10)
    ax.set_title(f"{dataset.upper()} — shuffle diagnostic", fontsize=11)
    for bar, drop in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width() / 2, drop - 0.5,
                f"{drop:+.1f}pp", ha="center", va="top", fontsize=9,
                color="white" if drop < -2 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "shuffle_comparison.png"), dpi=200)
    plt.close(fig)

    # Figure 2: per-slice curves
    fig, ax = plt.subplots(figsize=(6, 4))
    x = list(range(len(curve_orig)))
    ax.plot(x, curve_orig, color="#1f77b4", label="Original input", linewidth=1.8)
    ax.plot(x, curve_rn,   color="#ff7f0e", label="Row-norm input", linewidth=1.8,
            linestyle="--")
    j_orig = int(np.argmax(curve_orig))
    j_rn   = int(np.argmax(curve_rn))
    ax.axvline(j_orig, color="#1f77b4", linewidth=0.8, linestyle=":")
    ax.axvline(j_rn,   color="#ff7f0e", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Slice j", fontsize=11)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_title(f"{dataset.upper()} — per-slice accuracy", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_slice_comparison.png"), dpi=200)
    plt.close(fig)

    # Figure 3: cutoff sensitivity
    cutoffs = sorted(cutoff_orig.keys())
    best_orig = [cutoff_orig[j]["test_best"] for j in cutoffs]
    best_rn   = [cutoff_rn[j]["test_best"]   for j in cutoffs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cutoffs, best_orig, "o-", color="#1f77b4", label="Original input",
            linewidth=1.8, markersize=5)
    ax.plot(cutoffs, best_rn,   "s--", color="#ff7f0e", label="Row-norm input",
            linewidth=1.8, markersize=5)
    ax.set_xlabel("Loss cutoff j*", fontsize=11)
    ax.set_ylabel("Best-slice test accuracy", fontsize=11)
    ax.set_title(f"{dataset.upper()} — cutoff sensitivity", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cutoff_sensitivity.png"), dpi=200)
    plt.close(fig)

    print(f"  Figures saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Main dataset runner
# ---------------------------------------------------------------------------

def run_dataset(dataset, epochs, lr, wd, seed, out_root, summary_rows):
    k = 64
    print(f"\n{'='*65}")
    print(f"Dataset: {dataset.upper()}  (k={k}, epochs={epochs}, seed={seed})")
    print(f"{'='*65}")

    U, labels, train_mask, val_mask, test_mask, eigenvalues = load_dataset(
        dataset, k=k, split_idx=0
    )
    n_classes = int(labels.max().item()) + 1
    print(f"  N={U.shape[0]}, n_classes={n_classes}, "
          f"train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

    U_tilde = F.normalize(U, p=2, dim=1)

    out_dir = os.path.join(out_root, dataset)
    os.makedirs(out_dir, exist_ok=True)

    # -- A: shuffle --
    diag_orig, diag_rn = exp_a(
        U, U_tilde, labels, train_mask, val_mask, test_mask,
        k, n_classes, epochs, seed,
    )
    print(f"\n  Shuffle drop — orig:    {diag_orig['coarse_drop_pp']:+.1f}pp  "
          f"(coarse={diag_orig['unshuffled_coarse']:.4f})")
    print(f"  Shuffle drop — rownorm: {diag_rn['coarse_drop_pp']:+.1f}pp  "
          f"(coarse={diag_rn['unshuffled_coarse']:.4f})")

    # -- B: per-slice --
    curve_orig, curve_rn, j_orig, j_rn = exp_b(
        U, U_tilde, labels, train_mask, val_mask, test_mask,
        k, n_classes, epochs, lr, wd, seed,
    )
    print(f"\n  Per-slice j* — orig={j_orig} (peak={max(curve_orig):.4f})  "
          f"rownorm={j_rn} (peak={max(curve_rn):.4f})")
    if j_rn < j_orig:
        print("  => j* shifted LEFT: row norm removes high-freq noise more effectively")
    elif j_rn > j_orig:
        print("  => j* shifted RIGHT: magnitude carried useful info that norm discards")
    else:
        print("  => j* unchanged: magnitude does not affect spectral task structure")

    # -- C: full table --
    c_results, j_c = exp_c(
        U, U_tilde, labels, train_mask, val_mask, test_mask,
        k, n_classes, eigenvalues, epochs, lr, wd, seed,
    )

    # -- D: cutoff sensitivity --
    cutoffs_clamped = [min(j, k // 2) for j in CUTOFF_SCAN]
    cutoffs_clamped = sorted(set(cutoffs_clamped))

    print("\n  [Exp D] Cutoff scan — original input")
    cd_orig = exp_d(U, labels, train_mask, val_mask, test_mask, k, n_classes,
                    cutoffs_clamped, epochs, lr, wd, seed, use_row_norm_input=False)

    print("\n  [Exp D] Cutoff scan — row-norm input")
    cd_rn = exp_d(U, labels, train_mask, val_mask, test_mask, k, n_classes,
                  cutoffs_clamped, epochs, lr, wd, seed, use_row_norm_input=True)

    # -- Figures --
    _make_figures(dataset, out_dir, diag_orig, diag_rn,
                  curve_orig, curve_rn, cd_orig, cd_rn)

    # -- Per-dataset CSV --
    csv_path = os.path.join(out_dir, "results_table.csv")
    _write_results_csv(csv_path, dataset, c_results,
                       diag_orig, diag_rn, j_orig, j_rn)

    # -- Print summary table --
    _print_table(dataset, c_results, diag_orig, diag_rn,
                 j_orig, j_rn, j_c, cd_orig, cd_rn)

    # -- Collect for global summary --
    _collect_summary_rows(summary_rows, dataset, c_results,
                          diag_orig, diag_rn, j_orig, j_rn)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_results_csv(path, dataset, c_results, diag_orig, diag_rn, j_orig, j_rn):
    rows = []
    for model_name, r in c_results.items():
        rows.append({
            "dataset":    dataset,
            "model":      model_name,
            "use_rn":     r["use_rn"],
            "use_sph":    r["use_sph"],
            "best_val":   round(r["bv"], 4),
            "test_best":  round(r["test"], 4),
            "test_coarse": round(r.get("test_coarse", r["test"]), 4),
            "test_full":   round(r.get("test_full", r["test"]), 4),
            "j_star":      r.get("j_star", "—"),
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_table(dataset, c_results, diag_orig, diag_rn,
                 j_orig, j_rn, j_c, cd_orig, cd_rn):
    div = "=" * 72
    print(f"\n{div}")
    print(f"RESULTS — {dataset.upper()}")
    print(div)
    print(f"{'Model':<28} {'TestBest':>9} {'TestFull':>9} {'TestCoarse':>11} {'j*':>4}")
    print("-" * 72)
    for name, r in c_results.items():
        tb   = round(r["test"], 4)
        tf   = round(r.get("test_full", r["test"]), 4)
        tc   = round(r.get("test_coarse", r["test"]), 4)
        j_st = r.get("j_star", "—")
        print(f"  {name:<26} {tb:>9.4f} {tf:>9.4f} {tc:>11.4f} {str(j_st):>4}")
    print(div)
    print(f"\n  Shuffle drop: orig={diag_orig['coarse_drop_pp']:+.1f}pp  "
          f"rownorm={diag_rn['coarse_drop_pp']:+.1f}pp")
    print(f"  Per-slice j*: orig={j_orig}  rownorm={j_rn}  stratC={j_c}")

    print(f"\n  Cutoff sensitivity:")
    print(f"  {'j_cut':>6}  {'orig_best':>10}  {'rn_best':>10}  "
          f"{'orig_j*':>8}  {'rn_j*':>8}")
    for jc in sorted(cd_orig.keys()):
        o = cd_orig[jc]
        r = cd_rn[jc]
        print(f"  {jc:>6}  {o['test_best']:>10.4f}  {r['test_best']:>10.4f}  "
              f"{o['j_star']:>8}  {r['j_star']:>8}")

    # Key conclusions
    print("\n  --- Key findings ---")
    m2 = c_results.get("Sliced-rownorm", {})
    m3 = c_results.get("Sliced-rownorm-nosph", {})
    if m2 and m3:
        diff = abs(m2["test"] - m3["test"])
        if diff < 0.005:
            print("  [3 vs 2] Internal sphere norm is REDUNDANT given row-norm input "
                  f"(diff={diff:.4f})")
        else:
            better = "Model 2 (sphere norm)" if m2["test"] > m3["test"] else "Model 3 (no sphere norm)"
            print(f"  [3 vs 2] {better} wins by {diff:.4f} — sphere norm adds value")

    m_orig = c_results.get("Sliced-orig", {})
    m_rn   = c_results.get("Sliced-rownorm", {})
    if m_orig and m_rn:
        diff = m_rn["test"] - m_orig["test"]
        print(f"  [Row norm effect] Sliced-rownorm vs Sliced-orig: {diff:+.4f}")


def _collect_summary_rows(summary_rows, dataset, c_results, diag_orig, diag_rn,
                           j_orig, j_rn):
    for model_name, r in c_results.items():
        use_rn  = r["use_rn"]
        use_sph = r["use_sph"]
        drop    = diag_rn["coarse_drop_pp"] if use_rn else diag_orig["coarse_drop_pp"]
        j_star  = r.get("j_star", j_rn if use_rn else j_orig)
        summary_rows.append({
            "dataset":           dataset,
            "model":             model_name,
            "use_row_norm_input": use_rn,
            "use_sphere_norm":   use_sph,
            "test_best":         round(r["test"], 4),
            "test_coarse":       round(r.get("test_coarse", r["test"]), 4),
            "shuffle_drop":      round(drop, 2),
            "optimal_j_star":    r.get("j_star", "—"),
        })


def _write_summary_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Row normalisation experiments")
    p.add_argument("--dataset", type=str, default=None,
                   help="Single dataset (default: all 4)")
    p.add_argument("--epochs",  type=int, default=200)
    p.add_argument("--lr",      type=float, default=0.01)
    p.add_argument("--wd",      type=float, default=5e-4)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--out-dir", type=str, default="outputs/rownorm")
    return p.parse_args()


def main():
    args = _parse_args()
    datasets = [args.dataset] if args.dataset else DATASETS

    summary_rows = []

    for ds in datasets:
        run_dataset(
            dataset=ds,
            epochs=args.epochs,
            lr=args.lr,
            wd=args.wd,
            seed=args.seed,
            out_root=args.out_dir,
            summary_rows=summary_rows,
        )

    summary_path = os.path.join(args.out_dir, "summary.csv")
    _write_summary_csv(summary_path, summary_rows)
    print(f"\nSaved {summary_path}")


if __name__ == "__main__":
    main()
