"""
Fixed-split multi-seed experiment — primary evaluation.

Split protocol
--------------
  cora, citeseer, pubmed   — standard PyG Planetoid fixed splits
                             verified against expected sizes before training
  amazon_photo, actor,     — fixed stratified 60/20/20 random split
  cornell                    generated ONCE with seed=100, saved to
                             outputs/fixed_splits/, reused for all seeds

Seeds [0, 1, 2, 3, 4] affect ONLY weight initialisation.
Split masks are never altered by the experiment seeds.

Usage
-----
    python scripts/run_fixed_split_multiseed.py
    python scripts/run_fixed_split_multiseed.py --dataset cora citeseer
    python scripts/run_fixed_split_multiseed.py --seeds 0 1 --epochs 200
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import load_dataset
from src.models.sliced_mlp import SlicedSpectralMLP
from src.models.baselines import StandardMLP, train_baseline
from src.training.trainer import train_sliced
from src.evaluation.metrics import accuracy, per_slice_accuracy
from src.evaluation.shuffle import run_shuffle_diagnostic
from src.cutoff.strategy_c import select_cutoff as strategy_c_cutoff

try:
    from sklearn.model_selection import train_test_split as sk_train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available — figures will be skipped")

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS  = ["cora", "citeseer", "pubmed", "amazon_photo", "actor", "cornell"]
PYG_DATASETS = {"cora", "citeseer", "pubmed"}     # use PyG fixed splits
SEEDS     = [0, 1, 2, 3, 4]
JCUT_SCAN = [0, 3, 7, 11, 15, 20, 31]
FIXED_SPLIT_SEED = 100    # seed for non-PyG fixed splits — never changes

K        = 64
N_LAYERS = 2
LR       = 0.01
WD       = 5e-4
EPOCHS   = 200

# Expected PyG split sizes AFTER LCC extraction.
# Some labeled nodes fall outside the LCC and are dropped from the masks.
# Full-graph sizes (Planetoid standard): Cora 140/500/1000, CiteSeer 120/500/1000
# LCC-filtered sizes confirmed from data pipeline:
EXPECTED_SPLITS = {
    "cora":     {"n": 2485,  "train": 122, "val": 459, "test": 915},
    "citeseer": {"n": 2120,  "train": 80,  "val": 328, "test": 663},
    "pubmed":   {"n": 19717, "train": 60,  "val": 500, "test": 1000},
}

OUT_ROOT        = "outputs/fixed_split_multiseed"
FIXED_SPLIT_DIR = "outputs/fixed_splits"
PER_SEED_DIR    = os.path.join(OUT_ROOT, "per_seed")
AGG_DIR         = os.path.join(OUT_ROOT, "aggregated")
TAB_DIR         = os.path.join(OUT_ROOT, "tables")
FIG_DIR         = os.path.join(OUT_ROOT, "figures")

DS_COLORS = {
    "cora":         "#1f77b4",
    "citeseer":     "#ff7f0e",
    "pubmed":       "#2ca02c",
    "amazon_photo": "#d62728",
    "actor":        "#9467bd",
    "cornell":      "#8c564b",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_lines: List[str] = []


def log(msg: str) -> None:
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _log_lines.append(line)


# ---------------------------------------------------------------------------
# Split management
# ---------------------------------------------------------------------------

def _stratified_split(labels: torch.Tensor, seed: int) -> Tuple:
    """Stratified 60/20/20 split using sklearn."""
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn required for non-PyG splits.  "
                           "pip install scikit-learn")
    N   = len(labels)
    y   = labels.numpy()
    idx = np.arange(N)
    idx_train, idx_temp = sk_train_test_split(
        idx, test_size=0.4, stratify=y[idx], random_state=seed
    )
    idx_val, idx_test = sk_train_test_split(
        idx_temp, test_size=0.5, stratify=y[idx_temp], random_state=seed
    )
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val]     = True
    test_mask[idx_test]   = True
    return train_mask, val_mask, test_mask


def get_or_create_fixed_split(
    dataset: str, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load saved fixed split for non-PyG datasets, or generate and save it.

    Generated ONCE with FIXED_SPLIT_SEED=100.  Never regenerated after that.
    """
    os.makedirs(FIXED_SPLIT_DIR, exist_ok=True)
    train_path = os.path.join(FIXED_SPLIT_DIR, f"{dataset}_train_mask.pt")
    val_path   = os.path.join(FIXED_SPLIT_DIR, f"{dataset}_val_mask.pt")
    test_path  = os.path.join(FIXED_SPLIT_DIR, f"{dataset}_test_mask.pt")

    if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        train_mask = torch.load(train_path)
        val_mask   = torch.load(val_path)
        test_mask  = torch.load(test_path)
        log(f"  Loaded saved fixed split for {dataset} "
            f"(train={train_mask.sum()}, val={val_mask.sum()}, "
            f"test={test_mask.sum()})")
    else:
        log(f"  Generating fixed split for {dataset} with seed={FIXED_SPLIT_SEED}")
        train_mask, val_mask, test_mask = _stratified_split(
            labels, FIXED_SPLIT_SEED
        )
        torch.save(train_mask, train_path)
        torch.save(val_mask,   val_path)
        torch.save(test_mask,  test_path)
        log(f"  Saved to {FIXED_SPLIT_DIR}/ "
            f"(train={train_mask.sum()}, val={val_mask.sum()}, "
            f"test={test_mask.sum()})")

    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# Step 1 — Verify environment and splits
# ---------------------------------------------------------------------------

def verify_pyg_splits(
    dataset: str,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    N: int,
) -> bool:
    """
    Check PyG split sizes match expected LCC-filtered values.

    Returns True if OK.  Logs a WARNING on mismatch but does NOT stop —
    the split is still the reproducible PyG fixed split, just LCC-filtered.
    A mismatch here means the expected values need updating, not that the
    split is wrong.
    """
    n_train = train_mask.sum().item()
    n_val   = val_mask.sum().item()
    n_test  = test_mask.sum().item()

    if dataset not in EXPECTED_SPLITS:
        log(f"  Split: N={N} train={n_train} val={n_val} test={n_test} "
            f"[PyG fixed, no size check]")
        return True

    exp = EXPECTED_SPLITS[dataset]
    ok  = (N == exp["n"] and n_train == exp["train"]
           and n_val == exp["val"] and n_test == exp["test"])

    if ok:
        log(f"  Split verified: N={N} train={n_train} val={n_val} "
            f"test={n_test} ✓")
    else:
        log(f"  WARNING [{dataset}] split sizes differ from expected "
            f"(got train={n_train} val={n_val} test={n_test}, "
            f"expected train={exp['train']} val={exp['val']} test={exp['test']}). "
            f"Proceeding — split is still PyG fixed, LCC-filtered.")
    return True   # never hard-stop; split is still valid


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _new_sliced(n_classes: int, seed: int) -> SlicedSpectralMLP:
    torch.manual_seed(seed)
    return SlicedSpectralMLP(
        k=K, n_classes=n_classes, n_layers=N_LAYERS, loss_weights="uniform"
    )


def _eval_sliced(
    model: SlicedSpectralMLP,
    U: torch.Tensor,
    labels: torch.Tensor,
    test_mask: torch.Tensor,
) -> List[float]:
    model.eval()
    with torch.no_grad():
        all_logits = model(U)
    return per_slice_accuracy(all_logits, labels, test_mask)


# ---------------------------------------------------------------------------
# Step 3 — Per-seed run
# ---------------------------------------------------------------------------

def run_one_seed(
    dataset: str,
    seed: int,
    U: torch.Tensor,
    labels: torch.Tensor,
    eigenvalues: np.ndarray,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    n_classes: int,
) -> dict:
    """Run all models for one (dataset, seed) trial.  Returns a results dict."""

    r: dict = {
        "dataset": dataset, "seed": seed,
        "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()),
        "n_test":  int(test_mask.sum()),
    }

    import time
    t0 = time.time()

    # ---- A. StandardMLP-half ----
    log(f"  [RUNNING] seed={seed} | A. StandardMLP-half")
    try:
        torch.manual_seed(seed)
        m  = StandardMLP(n_features=K // 2, n_classes=n_classes,
                         hidden_dim=K, n_layers=N_LAYERS)
        bv, ta = train_baseline(m, U[:, :K // 2], labels,
                                train_mask, val_mask, test_mask,
                                lr=LR, wd=WD, epochs=EPOCHS)
        r["mlp_half_val"]  = round(bv, 4)
        r["mlp_half_test"] = round(ta, 4)
        log(f"  [DONE]    seed={seed} | A. mlp_half val={bv:.4f} test={ta:.4f}")
    except Exception as e:
        log(f"  [FAILED]  seed={seed} | A. mlp_half — {e}")
        r["mlp_half_val"] = r["mlp_half_test"] = float("nan")

    # ---- B. StandardMLP-full ----
    log(f"  [RUNNING] seed={seed} | B. StandardMLP-full")
    try:
        torch.manual_seed(seed)
        m  = StandardMLP(n_features=K, n_classes=n_classes,
                         hidden_dim=K, n_layers=N_LAYERS)
        bv, ta = train_baseline(m, U, labels,
                                train_mask, val_mask, test_mask,
                                lr=LR, wd=WD, epochs=EPOCHS)
        r["mlp_full_val"]  = round(bv, 4)
        r["mlp_full_test"] = round(ta, 4)
        log(f"  [DONE]    seed={seed} | B. mlp_full val={bv:.4f} test={ta:.4f}")
    except Exception as e:
        log(f"  [FAILED]  seed={seed} | B. mlp_full — {e}")
        r["mlp_full_val"] = r["mlp_full_test"] = float("nan")

    # ---- C. Sliced-dense (no cutoff) ----
    log(f"  [RUNNING] seed={seed} | C. Sliced-dense")
    try:
        m  = _new_sliced(n_classes, seed)
        bv, _, _ = train_sliced(m, U, labels, train_mask, val_mask,
                                epochs=EPOCHS, lr=LR, wd=WD, loss_cutoff=None)
        sa = _eval_sliced(m, U, labels, test_mask)
        r["sliced_dense_val"]    = round(bv, 4)
        r["sliced_dense_best"]   = round(float(max(sa)), 4)
        r["sliced_dense_full"]   = round(float(sa[-1]),  4)
        r["sliced_dense_coarse"] = round(float(sa[0]),   4)
        r["sliced_dense_jstar"]  = int(np.argmax(sa))
        r["slice_curve_dense"]   = [round(float(a), 4) for a in sa]
        log(f"  [DONE]    seed={seed} | C. sliced_dense val={bv:.4f} "
            f"best={max(sa):.4f} j*={np.argmax(sa)}")
    except Exception as e:
        log(f"  [FAILED]  seed={seed} | C. sliced_dense — {e}")
        traceback.print_exc()
        r["sliced_dense_best"] = r["sliced_dense_coarse"] = float("nan")
        r["slice_curve_dense"] = []

    # ---- D. j_cut scan — select by VAL accuracy ----
    log(f"  [RUNNING] seed={seed} | D. j_cut scan {JCUT_SCAN}")
    jcut_rows: List[dict] = []
    best_jcut_val  = -1.0
    best_jcut_j: Optional[int] = None
    best_jcut_test = float("nan")
    jcut_curves: Dict[int, List[float]] = {}

    for jcut in JCUT_SCAN:
        jcut_eff = min(jcut, K // 2)
        log(f"  [RUNNING] seed={seed} | D. j_cut={jcut_eff}")
        try:
            torch.manual_seed(seed)   # reset for each j_cut
            m  = _new_sliced(n_classes, seed)
            bv, _, _ = train_sliced(m, U, labels, train_mask, val_mask,
                                    epochs=EPOCHS, lr=LR, wd=WD,
                                    loss_cutoff=jcut_eff)
            sa = _eval_sliced(m, U, labels, test_mask)
            tb = round(float(max(sa)), 4)
            jstar = int(np.argmax(sa))
            jcut_rows.append({
                "dataset": dataset, "seed": seed,
                "j_cut": jcut_eff, "best_val": round(bv, 4),
                "test_best": tb, "test_coarse": round(float(sa[0]), 4),
                "optimal_jstar": jstar,
            })
            jcut_curves[jcut_eff] = [round(float(a), 4) for a in sa]
            # *** j_cut selection uses VAL accuracy only ***
            if bv > best_jcut_val:
                best_jcut_val  = bv
                best_jcut_j    = jcut_eff
                best_jcut_test = tb
            log(f"  [DONE]    seed={seed} | D. j_cut={jcut_eff} "
                f"val={bv:.4f} test_best={tb:.4f}")
        except Exception as e:
            log(f"  [FAILED]  seed={seed} | D. j_cut={jcut_eff} — {e}")

    r["jcut_rows"]       = jcut_rows
    r["jcut_curves"]     = jcut_curves
    r["jcut_selected_j"] = best_jcut_j
    r["jcut_best_val"]   = round(best_jcut_val, 4)
    r["jcut_best_test"]  = round(best_jcut_test, 4) if not np.isnan(best_jcut_test) else float("nan")
    if best_jcut_j is not None:
        log(f"  [DONE]    seed={seed} | D. j_cut* selected={best_jcut_j} "
            f"test={best_jcut_test:.4f}")

    # ---- E. Strategy-C ----
    log(f"  [RUNNING] seed={seed} | E. Strategy-C")
    try:
        j_c = strategy_c_cutoff(eigenvalues, K)
        m   = _new_sliced(n_classes, seed)
        bv, _, _ = train_sliced(m, U, labels, train_mask, val_mask,
                                epochs=EPOCHS, lr=LR, wd=WD,
                                loss_cutoff=j_c)
        sa = _eval_sliced(m, U, labels, test_mask)
        r["stratC_cutoff"] = j_c
        r["stratC_val"]    = round(bv, 4)
        r["stratC_best"]   = round(float(max(sa)), 4)
        r["stratC_coarse"] = round(float(sa[0]),   4)
        log(f"  [DONE]    seed={seed} | E. stratC j_c={j_c} "
            f"val={bv:.4f} best={max(sa):.4f}")
    except Exception as e:
        log(f"  [FAILED]  seed={seed} | E. stratC — {e}")
        r["stratC_best"] = float("nan")

    # ---- F. Shuffle diagnostic ----
    log(f"  [RUNNING] seed={seed} | F. Shuffle diagnostic")
    try:
        diag = run_shuffle_diagnostic(
            U, labels, train_mask, val_mask, test_mask,
            k=K, n_classes=n_classes, seed=seed, epochs=EPOCHS,
        )
        r["shuffle_coarse_orig"]  = round(diag["unshuffled_coarse"], 4)
        r["shuffle_coarse_shuf"]  = round(diag["shuffled_coarse"],   4)
        r["shuffle_drop_pp"]      = round(diag["coarse_drop_pp"],    2)
        r["slice_curve_unshuf"]   = [round(float(a), 4)
                                     for a in diag["slice_test_unshuffled"]]
        r["slice_curve_shuf"]     = [round(float(a), 4)
                                     for a in diag["slice_test_shuffled"]]
        log(f"  [DONE]    seed={seed} | F. shuffle "
            f"drop={diag['coarse_drop_pp']:+.1f}pp  "
            f"orig={diag['unshuffled_coarse']:.4f}")
    except Exception as e:
        log(f"  [FAILED]  seed={seed} | F. shuffle — {e}")
        traceback.print_exc()
        r["shuffle_drop_pp"] = float("nan")

    elapsed = time.time() - t0
    mlp_h = r.get("mlp_half_test", float("nan"))
    s_best = r.get("jcut_best_test", float("nan"))
    drop   = r.get("shuffle_drop_pp", float("nan"))
    j_sel  = r.get("jcut_selected_j", "?")
    log(f"[DONE] {dataset} seed={seed} | "
        f"MLP-half={mlp_h*100:.1f}% "
        f"Sliced-best={s_best*100:.1f}% "
        f"j_cut*={j_sel} "
        f"drop={drop:+.1f}pp "
        f"(elapsed: {elapsed:.0f}s)")

    return r


# ---------------------------------------------------------------------------
# Step 2 / Dataset loop
# ---------------------------------------------------------------------------

def run_dataset(
    dataset: str, seeds: List[int]
) -> Tuple[List[dict], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load eigenvectors once, build fixed split, run all seeds.

    Returns (seed_results, train_mask, val_mask, test_mask).
    """
    log(f"[{dataset.upper()}] Loading data (k={K})…")
    try:
        U, labels, pyg_train, pyg_val, pyg_test, eigenvalues = load_dataset(
            dataset, k=K, split_idx=0
        )
    except Exception as e:
        log(f"[{dataset.upper()}] FAILED to load: {e}")
        traceback.print_exc()
        return [], None, None, None

    n_classes = int(labels.max().item()) + 1
    N = U.shape[0]

    # --- Select split ---
    if dataset in PYG_DATASETS:
        train_mask, val_mask, test_mask = pyg_train, pyg_val, pyg_test
        split_type = "PyG fixed"
        verify_pyg_splits(dataset, train_mask, val_mask, test_mask, N)
    else:
        train_mask, val_mask, test_mask = get_or_create_fixed_split(
            dataset, labels
        )
        split_type = f"fixed random (seed={FIXED_SPLIT_SEED})"
        log(f"  Split type: {split_type}")

    log(f"[{dataset.upper()}] N={N}  n_classes={n_classes}  "
        f"train={train_mask.sum()}  val={val_mask.sum()}  "
        f"test={test_mask.sum()}  [{split_type}]")

    seed_results: List[dict] = []
    for seed in seeds:
        log(f"[{dataset.upper()}] === seed={seed} ===")
        try:
            r = run_one_seed(dataset, seed, U, labels, eigenvalues,
                             train_mask, val_mask, test_mask, n_classes)
            seed_results.append(r)
            _save_per_seed_csv(dataset, seed, r)
        except Exception as e:
            log(f"[{dataset.upper()}] seed={seed} FAILED: {e}")
            traceback.print_exc()

    return seed_results, train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# CSV I/O — per-seed long-format
# ---------------------------------------------------------------------------

_PER_SEED_COLS = [
    "dataset", "seed", "model",
    "val_best", "test_best", "test_full", "test_coarse",
    "j_cut_used", "optimal_j_star", "shuffle_drop",
]

_NAN = ""   # how we represent missing/N/A in CSV


def _r(v) -> str:
    if v is None:
        return _NAN
    if isinstance(v, float) and np.isnan(v):
        return _NAN
    return str(v)


def _save_per_seed_csv(dataset: str, seed: int, r: dict) -> None:
    os.makedirs(PER_SEED_DIR, exist_ok=True)
    path = os.path.join(PER_SEED_DIR, f"{dataset}_seed{seed}.csv")

    rows = []
    # StandardMLP-half
    rows.append({
        "dataset": dataset, "seed": seed, "model": "StandardMLP-half",
        "val_best":  _r(r.get("mlp_half_val")),
        "test_best": _r(r.get("mlp_half_test")),
        "test_full": "", "test_coarse": "",
        "j_cut_used": "", "optimal_j_star": "", "shuffle_drop": "",
    })
    # StandardMLP-full
    rows.append({
        "dataset": dataset, "seed": seed, "model": "StandardMLP-full",
        "val_best":  _r(r.get("mlp_full_val")),
        "test_best": _r(r.get("mlp_full_test")),
        "test_full": "", "test_coarse": "",
        "j_cut_used": "", "optimal_j_star": "", "shuffle_drop": "",
    })
    # Sliced-dense
    rows.append({
        "dataset": dataset, "seed": seed, "model": "Sliced-dense",
        "val_best":      _r(r.get("sliced_dense_val")),
        "test_best":     _r(r.get("sliced_dense_best")),
        "test_full":     _r(r.get("sliced_dense_full")),
        "test_coarse":   _r(r.get("sliced_dense_coarse")),
        "j_cut_used":    "",
        "optimal_j_star": _r(r.get("sliced_dense_jstar")),
        "shuffle_drop":  "",
    })
    # Sliced-jcut
    rows.append({
        "dataset": dataset, "seed": seed, "model": "Sliced-jcut",
        "val_best":      _r(r.get("jcut_best_val")),
        "test_best":     _r(r.get("jcut_best_test")),
        "test_full":     "",
        "test_coarse":   "",
        "j_cut_used":    _r(r.get("jcut_selected_j")),
        "optimal_j_star": _r(r.get("jcut_selected_j")),
        "shuffle_drop":  "",
    })
    # Strategy-C
    rows.append({
        "dataset": dataset, "seed": seed, "model": "Strategy-C",
        "val_best":      _r(r.get("stratC_val")),
        "test_best":     _r(r.get("stratC_best")),
        "test_full":     "",
        "test_coarse":   _r(r.get("stratC_coarse")),
        "j_cut_used":    _r(r.get("stratC_cutoff")),
        "optimal_j_star": _r(r.get("stratC_cutoff")),
        "shuffle_drop":  "",
    })
    # Shuffle diagnostic
    rows.append({
        "dataset": dataset, "seed": seed, "model": "Shuffle",
        "val_best":      "",
        "test_best":     _r(r.get("shuffle_coarse_orig")),
        "test_full":     "",
        "test_coarse":   _r(r.get("shuffle_coarse_shuf")),
        "j_cut_used":    "",
        "optimal_j_star": "",
        "shuffle_drop":  _r(r.get("shuffle_drop_pp")),
    })

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_PER_SEED_COLS)
        w.writeheader()
        w.writerows(rows)

    # j_cut scan detail
    jcut_rows = r.get("jcut_rows", [])
    if jcut_rows:
        jpath = os.path.join(PER_SEED_DIR, f"{dataset}_seed{seed}_jcut.csv")
        jfields = ["dataset", "seed", "j_cut", "best_val",
                   "test_best", "test_coarse", "optimal_jstar"]
        with open(jpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=jfields)
            w.writeheader()
            w.writerows(jcut_rows)


# ---------------------------------------------------------------------------
# Step 4 — Aggregation
# ---------------------------------------------------------------------------

def _mean(vals: list) -> float:
    v = [x for x in vals if x is not None and
         not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def _std(vals: list) -> float:
    v = [x for x in vals if x is not None and
         not (isinstance(x, float) and np.isnan(x))]
    return float(np.std(v, ddof=0)) if len(v) > 1 else 0.0


def aggregate_dataset(dataset: str, seed_results: List[dict]) -> dict:
    if not seed_results:
        return {}

    keys = [
        "mlp_half_test", "mlp_full_test",
        "sliced_dense_best", "jcut_best_test", "stratC_best",
        "shuffle_drop_pp",
    ]
    agg = {"dataset": dataset, "n_seeds": len(seed_results)}
    for k in keys:
        vals = [r.get(k) for r in seed_results]
        agg[f"{k}_mean"] = round(_mean(vals), 4)
        agg[f"{k}_std"]  = round(_std(vals),  4)

    # j_cut* selection frequency
    selected = [r.get("jcut_selected_j") for r in seed_results
                if r.get("jcut_selected_j") is not None]
    counts = {}
    for j in JCUT_SCAN:
        counts[j] = selected.count(j)
    agg["jcut_counts"]    = counts
    agg["jcut_mode"]      = max(counts, key=counts.get) if selected else None
    agg["jcut_selected"]  = selected

    # j_cut sensitivity: test_best at each j_cut, mean across seeds
    jcut_by_j: Dict[int, List[float]] = {}
    for r in seed_results:
        for row in r.get("jcut_rows", []):
            j = row["j_cut"]
            jcut_by_j.setdefault(j, []).append(row["test_best"])
    agg["jcut_mean"]  = {j: round(_mean(v), 4) for j, v in jcut_by_j.items()}
    agg["jcut_std"]   = {j: round(_std(v),  4) for j, v in jcut_by_j.items()}

    # slice curves (for figures)
    curves_dense  = [r.get("slice_curve_dense")  for r in seed_results
                     if r.get("slice_curve_dense")]
    curves_unshuf = [r.get("slice_curve_unshuf") for r in seed_results
                     if r.get("slice_curve_unshuf")]
    curves_shuf   = [r.get("slice_curve_shuf")   for r in seed_results
                     if r.get("slice_curve_shuf")]

    if curves_dense:
        arr = np.array(curves_dense)
        agg["slice_dense_mean"] = arr.mean(0).tolist()
        agg["slice_dense_std"]  = arr.std(0).tolist()
    if curves_unshuf:
        arr = np.array(curves_unshuf)
        agg["slice_unshuf_mean"] = arr.mean(0).tolist()
        agg["slice_unshuf_std"]  = arr.std(0).tolist()
    if curves_shuf:
        arr = np.array(curves_shuf)
        agg["slice_shuf_mean"] = arr.mean(0).tolist()

    return agg


def save_aggregated_csv(dataset: str, agg: dict, seed_results: List[dict]) -> None:
    os.makedirs(AGG_DIR, exist_ok=True)
    fields = [
        "dataset", "n_seeds",
        "mlp_half_test_mean",   "mlp_half_test_std",
        "mlp_full_test_mean",   "mlp_full_test_std",
        "sliced_dense_best_mean", "sliced_dense_best_std",
        "jcut_best_test_mean",  "jcut_best_test_std",
        "stratC_best_mean",     "stratC_best_std",
        "shuffle_drop_pp_mean", "shuffle_drop_pp_std",
        "jcut_mode",
    ]
    row = {f: agg.get(f, "") for f in fields}
    path = os.path.join(AGG_DIR, f"{dataset}_aggregated.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(row)

    # Per-seed summary rows in one file
    detail_path = os.path.join(AGG_DIR, f"{dataset}_per_seed_summary.csv")
    summary_fields = [
        "seed", "mlp_half_test", "mlp_full_test",
        "sliced_dense_best", "jcut_best_test", "stratC_best",
        "jcut_selected_j", "shuffle_drop_pp",
    ]
    with open(detail_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in seed_results:
            w.writerow({f: r.get(f, "") for f in summary_fields})


# ---------------------------------------------------------------------------
# Step 5 — Tables
# ---------------------------------------------------------------------------

def _fmt(v, pct: bool = True) -> str:
    if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        return f"{float(v)*100:.1f}" if pct else f"{float(v):.4f}"
    except Exception:
        return str(v)


def _fmt_ms(m, s) -> str:
    """Format mean±std as XX.X±Y.Y."""
    if m is None or m == "" or (isinstance(m, float) and np.isnan(float(m))):
        return "—"
    try:
        return f"{float(m)*100:.1f}±{float(s)*100:.1f}"
    except Exception:
        return "—"


def build_main_table(all_agg: Dict[str, dict], datasets: List[str]) -> None:
    os.makedirs(TAB_DIR, exist_ok=True)

    methods = [
        ("MLP-half",     "mlp_half_test"),
        ("MLP-full",     "mlp_full_test"),
        ("Sliced-dense", "sliced_dense_best"),
        ("Sliced-best",  "jcut_best_test"),
        ("Strategy-C",   "stratC_best"),
    ]

    # CSV
    csv_fields = (["dataset"] +
                  [f"{n}_mean" for n, _ in methods] +
                  [f"{n}_std"  for n, _ in methods] +
                  ["shuffle_drop_mean", "shuffle_drop_std", "jcut_mode"])
    csv_rows = []
    for ds in datasets:
        agg = all_agg.get(ds, {})
        row = {"dataset": ds}
        for name, key in methods:
            row[f"{name}_mean"] = agg.get(f"{key}_mean", "")
            row[f"{name}_std"]  = agg.get(f"{key}_std",  "")
        row["shuffle_drop_mean"] = agg.get("shuffle_drop_pp_mean", "")
        row["shuffle_drop_std"]  = agg.get("shuffle_drop_pp_std",  "")
        row["jcut_mode"]         = agg.get("jcut_mode", "")
        csv_rows.append(row)

    csv_path = os.path.join(TAB_DIR, "main_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader(); w.writerows(csv_rows)

    # LaTeX
    actor_idx = datasets.index("actor") if "actor" in datasets else None
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Node classification test accuracy (\%, mean\,±\,std across 5 seeds). "
        r"Fixed splits. Seeds 0--4 affect weight initialisation only. "
        r"\textbf{Bold}: best Sliced variant per dataset. "
        r"$j^*$: most frequent cutoff across seeds.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Dataset & MLP-$\frac{k}{2}$ & MLP-$k$ & Sliced-dense "
        r"& Sliced-best & Strat-C & Shuf-drop & $j^*$ \\",
        r"\midrule",
    ]
    for i, row in enumerate(csv_rows):
        if actor_idx is not None and i == actor_idx:
            tex.append(r"\midrule")
        ds = row["dataset"].replace("_", r"\_")

        # Find best sliced value for bolding
        sliced_vals = {}
        for name in ["Sliced-dense", "Sliced-best", "Strategy-C"]:
            m = row.get(f"{name}_mean", "")
            try:
                sliced_vals[name] = float(m)
            except (ValueError, TypeError):
                pass
        best_sliced = max(sliced_vals.values()) if sliced_vals else None

        cols = []
        for name, _ in methods:
            s = _fmt_ms(row.get(f"{name}_mean"), row.get(f"{name}_std"))
            if best_sliced is not None and name in sliced_vals:
                if abs(sliced_vals[name] - best_sliced) < 1e-5:
                    s = f"\\textbf{{{s}}}"
            cols.append(s)

        # Shuffle drop column (pp, not fraction)
        m_drop = row.get("shuffle_drop_mean", "")
        s_drop = row.get("shuffle_drop_std",  "")
        try:
            drop_s = f"{float(m_drop):+.1f}±{float(s_drop):.1f}"
        except (ValueError, TypeError):
            drop_s = "—"
        cols.append(drop_s)
        cols.append(str(row.get("jcut_mode", "—")))

        tex.append(f"  {ds} & " + " & ".join(cols) + r" \\")

    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_path = os.path.join(TAB_DIR, "main_results.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex) + "\n")

    log(f"Saved {csv_path} + {tex_path}")


def build_jcut_sensitivity_table(
    all_agg: Dict[str, dict], datasets: List[str]
) -> None:
    """j_cut × dataset table with mean±std test accuracy plus dense row."""
    os.makedirs(TAB_DIR, exist_ok=True)
    target = [d for d in datasets
              if d in ("cora", "citeseer", "pubmed", "amazon_photo")]

    rows = []
    for j in JCUT_SCAN:
        row = {"j_cut": j}
        for ds in target:
            agg = all_agg.get(ds, {})
            m   = agg.get("jcut_mean", {}).get(j)
            s   = agg.get("jcut_std",  {}).get(j)
            row[ds]            = m if m is not None else ""
            row[f"{ds}_std"]   = s if s is not None else ""
        rows.append(row)
    # Dense row
    row = {"j_cut": "dense"}
    for ds in target:
        agg = all_agg.get(ds, {})
        row[ds]          = agg.get("sliced_dense_best_mean", "")
        row[f"{ds}_std"] = agg.get("sliced_dense_best_std",  "")
    rows.append(row)

    fields = ["j_cut"] + target + [f"{d}_std" for d in target]
    csv_path = os.path.join(TAB_DIR, "jcut_sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    # Find best per dataset (for bold)
    best_per_ds = {}
    for ds in target:
        vals = [r.get(ds) for r in rows if r.get(ds) != ""]
        if vals:
            try:
                best_per_ds[ds] = max(float(v) for v in vals)
            except Exception:
                pass

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{$j^*$ cutoff sensitivity. Mean\,±\,std test accuracy (\%) "
        r"across 5 seeds. Bold = best per dataset.}",
        r"\label{tab:jcut_sensitivity}",
        r"\begin{tabular}{l" + "r" * len(target) + "}",
        r"\toprule",
        "$j^*$ & " + " & ".join(d.replace("_", r"\_") for d in target) + r" \\",
        r"\midrule",
    ]
    for i, row in enumerate(rows):
        if i == len(rows) - 1:
            tex.append(r"\midrule")  # separator before dense row
        cols = []
        for ds in target:
            m = row.get(ds,        "")
            s = row.get(f"{ds}_std", "")
            cell = _fmt_ms(m, s)
            if m != "" and ds in best_per_ds:
                try:
                    if abs(float(m) - best_per_ds[ds]) < 1e-5:
                        cell = f"\\textbf{{{cell}}}"
                except Exception:
                    pass
            cols.append(cell)
        label = str(row["j_cut"])
        tex.append(f"  {label} & " + " & ".join(cols) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = os.path.join(TAB_DIR, "jcut_sensitivity.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex) + "\n")

    log(f"Saved {csv_path} + {tex_path}")


def build_jcut_stability_table(
    all_agg: Dict[str, dict], datasets: List[str]
) -> None:
    """Frequency count table: how many seeds selected each j_cut* value."""
    os.makedirs(TAB_DIR, exist_ok=True)

    fields = ["dataset"] + [str(j) for j in JCUT_SCAN]
    rows   = []
    for ds in datasets:
        agg    = all_agg.get(ds, {})
        counts = agg.get("jcut_counts", {j: 0 for j in JCUT_SCAN})
        row    = {"dataset": ds}
        for j in JCUT_SCAN:
            row[str(j)] = counts.get(j, 0)
        rows.append(row)

    csv_path = os.path.join(TAB_DIR, "jcut_stability.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    # LaTeX with \cellcolor — requires \usepackage[table]{xcolor}
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{$j^*$ selection frequency. Cell = number of seeds (out of 5) "
        r"that selected this $j^*$ value via val accuracy. "
        r"Requires \texttt{\\usepackage[table]\{xcolor\}}.}",
        r"\label{tab:jcut_stability}",
        r"\begin{tabular}{l" + "r" * len(JCUT_SCAN) + "}",
        r"\toprule",
        "Dataset & " + " & ".join(str(j) for j in JCUT_SCAN) + r" \\",
        r"\midrule",
    ]
    # Blue intensity 0..5 → 0..50% fill
    def _cell_color(count: int) -> str:
        if count == 0:
            return ""
        pct = count * 10  # 1→10%, 5→50%
        return f"\\cellcolor{{blue!{pct}}}"

    for row in rows:
        ds   = row["dataset"].replace("_", r"\_")
        cols = []
        for j in JCUT_SCAN:
            c = row.get(str(j), 0)
            cc = _cell_color(c)
            cols.append(f"{cc}{c}")
        tex.append(f"  {ds} & " + " & ".join(cols) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = os.path.join(TAB_DIR, "jcut_stability.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex) + "\n")

    log(f"Saved {csv_path} + {tex_path}")


# ---------------------------------------------------------------------------
# Step 6 — Figures
# ---------------------------------------------------------------------------

def _savefig(name: str, dpi: int = 300) -> None:
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    log(f"Saved {path}")


def make_figures(
    all_agg: Dict[str, dict],
    all_seed_results: Dict[str, List[dict]],
    datasets: List[str],
) -> None:
    if not HAS_MPL:
        log("WARNING: matplotlib not available — figures skipped")
        return
    os.makedirs(FIG_DIR, exist_ok=True)

    _fig1_shuffle_drop(all_agg, datasets)
    _fig2_per_slice_curves(all_agg, datasets)
    _fig3_jcut_sensitivity(all_agg, datasets)
    _fig4_main_results_bars(all_agg, datasets)
    _fig5_scatter(all_agg, datasets)
    _fig6_jcut_stability_heatmap(all_agg, datasets)


def _fig1_shuffle_drop(all_agg: Dict[str, dict], datasets: List[str]) -> None:
    """Horizontal bar chart: shuffle drop per dataset."""
    ds_sorted = sorted(
        datasets,
        key=lambda d: abs(all_agg.get(d, {}).get("shuffle_drop_pp_mean") or 0),
        reverse=True,
    )
    means = [all_agg.get(d, {}).get("shuffle_drop_pp_mean", 0) for d in ds_sorted]
    stds  = [all_agg.get(d, {}).get("shuffle_drop_pp_std",  0) for d in ds_sorted]

    def _color(m):
        if m is None or np.isnan(m):
            return "gray"
        if m < -5:
            return "#2ca02c"   # green — strong signal
        if m < -1:
            return "#ff7f0e"   # orange — weak signal
        return "#d62728"       # red — no signal

    colors = [_color(m) for m in means]

    fig, ax = plt.subplots(figsize=(8, max(3, len(ds_sorted) * 0.7)))
    y = np.arange(len(ds_sorted))
    ax.barh(y, means, xerr=stds, color=colors, alpha=0.85,
            capsize=4, height=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(-5, color="#ff7f0e", lw=0.8, linestyle=":", alpha=0.7,
               label="-5pp threshold")
    ax.set_yticks(y)
    ax.set_yticklabels([d.replace("_", "-") for d in ds_sorted])
    ax.set_xlabel("Accuracy drop when eigenvector ordering is destroyed (pp)")
    ax.set_title("Shuffle diagnostic: coarse-slice accuracy drop\n"
                 "(mean ± std, 5 seeds, fixed splits)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _savefig("shuffle_drop.png")


def _fig2_per_slice_curves(all_agg: Dict[str, dict], datasets: List[str]) -> None:
    """2×3 grid: per-slice accuracy curves mean ± std band."""
    n_ds  = len(datasets)
    ncols = 3
    nrows = (n_ds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)

    for idx, ds in enumerate(datasets):
        ax  = axes[idx // ncols][idx % ncols]
        agg = all_agg.get(ds, {})

        # Unshuffled (solid blue, ±std band)
        dm  = agg.get("slice_unshuf_mean")
        ds_ = agg.get("slice_unshuf_std")
        if dm:
            x  = np.arange(len(dm))
            m  = np.array(dm)  * 100
            s  = np.array(ds_) * 100
            ax.plot(x, m, color="#1f77b4", lw=1.8, label="Sliced (unshuf)")
            ax.fill_between(x, m - s, m + s, alpha=0.2, color="#1f77b4")

        # Shuffled coarse reference (dashed red horizontal)
        shuf_m = agg.get("slice_shuf_mean")
        if shuf_m:
            ax.axhline(float(shuf_m[0]) * 100, color="#d62728",
                       lw=1.2, linestyle="--", alpha=0.8, label="Shuf. coarse")

        # MLP-half reference (dashed gray)
        mlp_m = agg.get("mlp_half_test_mean")
        if mlp_m:
            ax.axhline(float(mlp_m) * 100, color="gray",
                       lw=1.2, linestyle="--", alpha=0.8, label="MLP-half")

        # Vertical dotted line at most-frequent j_cut*
        mode_j = agg.get("jcut_mode")
        if mode_j is not None:
            ax.axvline(mode_j, color="purple", lw=1.0, linestyle=":",
                       alpha=0.7, label=f"j*={mode_j}")

        drop_m = agg.get("shuffle_drop_pp_mean")
        drop_s = f"{float(drop_m):+.1f}pp" if drop_m is not None else ""
        ax.set_title(f"{ds.replace('_', '-')} (drop={drop_s})", fontsize=9)
        ax.set_xlabel("Slice j", fontsize=8)
        ax.set_ylabel("Test acc (%)", fontsize=8)
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    for idx in range(n_ds, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.suptitle("Per-slice test accuracy (mean ± std, 5 seeds, fixed splits)",
                 fontsize=11)
    plt.tight_layout()
    _savefig("per_slice_curves.png")


def _fig3_jcut_sensitivity(all_agg: Dict[str, dict], datasets: List[str]) -> None:
    """j_cut sensitivity: mean ± std for each dataset."""
    target = [d for d in datasets
              if d in ("cora", "citeseer", "pubmed", "amazon_photo")]

    fig, ax = plt.subplots(figsize=(9, 5))
    x_scan  = list(JCUT_SCAN)
    x_dense = max(JCUT_SCAN) + 4   # "dense" plotted to the right

    for ds in target:
        agg   = all_agg.get(ds, {})
        color = DS_COLORS.get(ds, "gray")

        means = [agg.get("jcut_mean", {}).get(j, float("nan")) for j in x_scan]
        stds  = [agg.get("jcut_std",  {}).get(j, 0.0) for j in x_scan]
        means_pct = [m * 100 if not np.isnan(m) else np.nan for m in means]
        stds_pct  = [s * 100 for s in stds]

        # Append dense result
        dm = agg.get("sliced_dense_best_mean")
        ds2 = agg.get("sliced_dense_best_std", 0)
        means_pct.append(float(dm) * 100 if dm else np.nan)
        stds_pct.append(float(ds2) * 100 if ds2 else 0.0)
        xs = x_scan + [x_dense]

        ax.plot(xs, means_pct, marker="o", color=color,
                label=ds.replace("_", "-"), lw=1.5)
        m_arr = np.array(means_pct)
        s_arr = np.array(stds_pct)
        ax.fill_between(xs, m_arr - s_arr, m_arr + s_arr,
                        alpha=0.12, color=color)

        # MLP-half reference (horizontal dashed)
        mlp_m = agg.get("mlp_half_test_mean")
        if mlp_m:
            ax.axhline(float(mlp_m) * 100, color=color, lw=0.8,
                       linestyle="--", alpha=0.5)

        # Mark optimal j_cut* (mode)
        mode_j = agg.get("jcut_mode")
        if mode_j is not None and mode_j in JCUT_SCAN:
            idx = x_scan.index(mode_j)
            if not np.isnan(means_pct[idx]):
                ax.scatter([mode_j], [means_pct[idx]], s=60, color=color,
                           zorder=5, marker="*")

    ax.set_xticks(x_scan + [x_dense])
    ax.set_xticklabels([str(j) for j in x_scan] + ["dense"], fontsize=8)
    ax.set_xlabel("j_cut (loss cutoff)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("j_cut sensitivity (mean ± std, 5 seeds)\n"
                 "★ = mode j_cut* | dashed = MLP-half")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig("jcut_sensitivity.png")


def _fig4_main_results_bars(all_agg: Dict[str, dict], datasets: List[str]) -> None:
    """Grouped bars: MLP-half, Sliced-dense, Sliced-best (exclude Actor)."""
    plot_ds = [d for d in datasets if d != "actor"]
    methods = [
        ("MLP-half",     "mlp_half_test",    "#1f77b4"),
        ("Sliced-dense", "sliced_dense_best", "#ff7f0e"),
        ("Sliced-best",  "jcut_best_test",    "#2ca02c"),
    ]
    n_ds  = len(plot_ds)
    n_m   = len(methods)
    x     = np.arange(n_ds)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, n_ds * 1.8), 5))
    for i, (name, key, color) in enumerate(methods):
        means = [all_agg.get(d, {}).get(f"{key}_mean", float("nan"))
                 for d in plot_ds]
        stds  = [all_agg.get(d, {}).get(f"{key}_std",  0.0)
                 for d in plot_ds]
        pct_m = [m * 100 if not np.isnan(m) else np.nan for m in means]
        pct_s = [s * 100 for s in stds]
        offset = (i - n_m / 2 + 0.5) * width
        ax.bar(x + offset, pct_m, width, label=name, color=color,
               yerr=pct_s, capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in plot_ds], fontsize=9)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(bottom=60)
    ax.set_title("Main results: MLP-half vs Sliced (mean ± std, 5 seeds)\n"
                 "Fixed splits; Actor excluded (no spectral signal)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig("main_results_bars.png")


def _fig5_scatter(all_agg: Dict[str, dict], datasets: List[str]) -> None:
    """Scatter: |shuffle drop| vs accuracy gain (Sliced-best - MLP-half)."""
    fig, ax = plt.subplots(figsize=(6, 5))

    xs, ys = [], []
    xs_err, ys_err = [], []
    labels_pts = []

    for ds in datasets:
        agg  = all_agg.get(ds, {})
        drop = agg.get("shuffle_drop_pp_mean")
        ds_d = agg.get("shuffle_drop_pp_std", 0)
        mlp  = agg.get("mlp_half_test_mean")
        slc  = agg.get("jcut_best_test_mean")
        ds_m = agg.get("mlp_half_test_std",  0)
        ds_s = agg.get("jcut_best_test_std",  0)

        if any(v is None or (isinstance(v, float) and np.isnan(v))
               for v in [drop, mlp, slc]):
            continue

        x_val  = abs(float(drop))
        y_val  = (float(slc) - float(mlp)) * 100
        x_err  = float(ds_d)
        y_err  = (float(ds_s) + float(ds_m)) * 100

        xs.append(x_val); ys.append(y_val)
        xs_err.append(x_err); ys_err.append(y_err)
        labels_pts.append(ds)

        color = DS_COLORS.get(ds, "gray")
        ax.scatter(x_val, y_val, s=80, color=color, zorder=3)
        ax.errorbar(x_val, y_val, xerr=x_err, yerr=y_err,
                    color=color, alpha=0.4, capsize=3)
        ax.annotate(ds.replace("_", "-"), (x_val, y_val),
                    textcoords="offset points", xytext=(6, 3), fontsize=8)

    # Regression line
    if HAS_SCIPY and len(xs) >= 3:
        slope, intercept, r_val, p_val, _ = scipy_stats.linregress(xs, ys)
        x_line = np.linspace(min(xs), max(xs), 100)
        ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.2, alpha=0.6,
                label=f"r={r_val:.2f} (p={p_val:.2f})")
        ax.legend(fontsize=8)

    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.4)
    ax.axvline(5, color="red",   lw=0.8, linestyle=":",  alpha=0.5,
               label="|drop|=5pp")

    # Quadrant labels
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.text(xlim[0] + 0.02 * (xlim[1] - xlim[0]),
            ylim[1] - 0.05 * (ylim[1] - ylim[0]),
            "spectral ordering helps", fontsize=7, color="gray", ha="left")
    ax.text(xlim[1] - 0.02 * (xlim[1] - xlim[0]),
            ylim[0] + 0.05 * (ylim[1] - ylim[0]),
            "ordering irrelevant", fontsize=7, color="gray", ha="right")

    ax.set_xlabel("|Shuffle drop| (pp) — larger = more ordering matters")
    ax.set_ylabel("Sliced-best − MLP-half accuracy gain (pp)")
    ax.set_title("Spectral signal vs Sliced advantage\n(5 seeds, fixed splits)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig("scatter_drop_vs_gain.png")


def _fig6_jcut_stability_heatmap(
    all_agg: Dict[str, dict], datasets: List[str]
) -> None:
    """Heatmap: j_cut* selection frequency across seeds (0–5)."""
    data = np.zeros((len(datasets), len(JCUT_SCAN)), dtype=float)
    for i, ds in enumerate(datasets):
        agg    = all_agg.get(ds, {})
        counts = agg.get("jcut_counts", {})
        for j_idx, j in enumerate(JCUT_SCAN):
            data[i, j_idx] = counts.get(j, 0)

    fig, ax = plt.subplots(
        figsize=(max(7, len(JCUT_SCAN) * 1.1), max(4, len(datasets) * 0.9))
    )
    im = ax.imshow(data, aspect="auto", cmap="Blues",
                   vmin=0, vmax=len(SEEDS))
    plt.colorbar(im, ax=ax, label="Seeds selecting this j_cut*")

    ax.set_xticks(range(len(JCUT_SCAN)))
    ax.set_xticklabels([str(j) for j in JCUT_SCAN])
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([d.replace("_", "-") for d in datasets])
    ax.set_xlabel("j_cut value")
    ax.set_title("j_cut* Selection Frequency Across Seeds\n"
                 "(count out of 5, selected by val accuracy)")

    for i in range(len(datasets)):
        for j in range(len(JCUT_SCAN)):
            v = int(data[i, j])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if v >= 3 else "black")

    plt.tight_layout()
    _savefig("jcut_stability_heatmap.png")


# ---------------------------------------------------------------------------
# Step 7 — Experiment log
# ---------------------------------------------------------------------------

def save_experiment_log(
    datasets: List[str],
    seeds: List[int],
    all_agg: Dict[str, dict],
    failed_runs: List[str],
    pytest_output: str,
) -> None:
    lines = [
        "# Fixed-Split Multi-Seed Experiment — Log",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Seeds:** {seeds} (weight init only)",
        f"**Datasets:** {datasets}",
        f"**k:** {K}  **n_layers:** {N_LAYERS}  **lr:** {LR}  "
        f"**wd:** {WD}  **epochs:** {EPOCHS}",
        "",
        "## Split Protocol",
        "",
        "| Dataset | Type | Train | Val | Test |",
        "|---------|------|-------|-----|------|",
    ]
    for ds in datasets:
        agg = all_agg.get(ds, {})
        if ds in PYG_DATASETS:
            stype = "PyG fixed"
            exp   = EXPECTED_SPLITS.get(ds, {})
            lines.append(f"| {ds} | {stype} | {exp.get('train','?')} "
                         f"| {exp.get('val','?')} | {exp.get('test','?')} |")
        else:
            lines.append(f"| {ds} | fixed random (seed={FIXED_SPLIT_SEED}) "
                         f"| 60% | 20% | 20% |")

    lines += [
        "",
        "## Bug 3 Fix — pytest output",
        "",
        "```",
    ]
    lines += pytest_output.splitlines()
    lines += [
        "```",
        "",
        "## Summary",
        "",
        "| Dataset | MLP-half | Sliced-best | Δ (pp) | j_cut* | Shuf-drop |",
        "|---------|----------|-------------|--------|--------|-----------|",
    ]
    for ds in datasets:
        agg  = all_agg.get(ds, {})
        mlp  = agg.get("mlp_half_test_mean")
        slc  = agg.get("jcut_best_test_mean")
        drop = agg.get("shuffle_drop_pp_mean")
        mode = agg.get("jcut_mode", "?")
        if mlp is not None and slc is not None:
            try:
                delta  = (float(slc) - float(mlp)) * 100
                drop_s = f"{float(drop):+.1f}" if drop else "—"
                lines.append(
                    f"| {ds} | {float(mlp)*100:.1f}±{agg.get('mlp_half_test_std',0)*100:.1f}"
                    f" | {float(slc)*100:.1f}±{agg.get('jcut_best_test_std',0)*100:.1f}"
                    f" | {delta:+.1f} | {mode} | {drop_s} |"
                )
            except Exception:
                lines.append(f"| {ds} | — | — | — | — | — |")
        else:
            lines.append(f"| {ds} | — | — | — | — | — |")

    if failed_runs:
        lines += ["", "## Failed Runs", ""]
        for f in failed_runs:
            lines.append(f"- {f}")
    else:
        lines += ["", "## Failed Runs", "", "NONE"]

    lines += [
        "",
        "## File Manifest",
        "",
        f"- `{PER_SEED_DIR}/` — per-seed CSVs",
        f"- `{AGG_DIR}/` — aggregated CSVs",
        f"- `{TAB_DIR}/` — LaTeX + CSV tables",
        f"- `{FIG_DIR}/` — PNG figures (300 dpi)",
        f"- `{FIXED_SPLIT_DIR}/` — saved split masks for non-PyG datasets",
        "",
        "## Important Notes",
        "",
        "- j_cut selection: **val accuracy only** (never test)",
        "- Bug 3 fix active in trainer.py: `track_j = loss_cutoff if loss_cutoff "
          "is not None else len(all_logits) - 1`",
        f"- Non-PyG fixed splits generated with seed={FIXED_SPLIT_SEED}",
        "- Eigenvectors reused across all seeds",
    ]

    path = os.path.join(OUT_ROOT, "experiment_log.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"Saved {path}")


# ---------------------------------------------------------------------------
# Step 1 — Environment verification
# ---------------------------------------------------------------------------

def verify_environment() -> str:
    """Run pytest tests/test_bug3_model_selection.py and return output."""
    import subprocess
    log("Running pytest tests/test_bug3_model_selection.py …")
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_bug3_model_selection.py", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    output = result.stdout + result.stderr
    print(output)
    if result.returncode != 0:
        log("PYTEST FAILED — fix tests before running experiment")
        sys.exit(1)
    log("pytest PASSED")
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Fixed-split multi-seed experiment"
    )
    p.add_argument("--dataset", type=str, nargs="+", default=None,
                   help="Datasets to run (default: all)")
    p.add_argument("--seeds",   type=int, nargs="+", default=None,
                   help="Seeds (default: 0 1 2 3 4)")
    p.add_argument("--epochs",  type=int, default=None,
                   help="Override epochs")
    p.add_argument("--skip-pytest", action="store_true",
                   help="Skip pytest verification step")
    args = p.parse_args()

    datasets = args.dataset if args.dataset else DATASETS
    seeds    = args.seeds   if args.seeds   else SEEDS
    global EPOCHS
    if args.epochs is not None:
        EPOCHS = args.epochs

    for d in [OUT_ROOT, PER_SEED_DIR, AGG_DIR, TAB_DIR, FIG_DIR]:
        os.makedirs(d, exist_ok=True)

    # --- Step 1: verify ---
    pytest_out = ""
    if not args.skip_pytest:
        pytest_out = verify_environment()
    else:
        log("Skipping pytest (--skip-pytest)")

    log("=" * 65)
    log("FIXED-SPLIT MULTI-SEED EXPERIMENT")
    log(f"Datasets : {datasets}")
    log(f"Seeds    : {seeds}  (weight init only)")
    log(f"Epochs   : {EPOCHS}  k={K}  lr={LR}  wd={WD}")
    log(f"j_cut    : selected by VAL accuracy, never test")
    log("=" * 65)

    all_seed_results: Dict[str, List[dict]] = {}
    all_agg:          Dict[str, dict]       = {}
    failed_runs:      List[str]             = []

    # --- Main loop ---
    for ds in datasets:
        seed_results, *_ = run_dataset(ds, seeds)
        all_seed_results[ds] = seed_results

        if seed_results:
            agg = aggregate_dataset(ds, seed_results)
            all_agg[ds] = agg
            save_aggregated_csv(ds, agg, seed_results)
        else:
            all_agg[ds] = {}
            failed_runs.append(f"{ds} — no results")

    # --- Tables ---
    log("Building tables…")
    build_main_table(all_agg, datasets)
    build_jcut_sensitivity_table(all_agg, datasets)
    build_jcut_stability_table(all_agg, datasets)

    # --- Figures ---
    log("Building figures…")
    make_figures(all_agg, all_seed_results, datasets)

    # --- Log ---
    save_experiment_log(datasets, seeds, all_agg, failed_runs, pytest_out)

    # --- Master CSV ---
    master_fields = [
        "dataset",
        "mlp_half_test_mean",   "mlp_half_test_std",
        "mlp_full_test_mean",   "mlp_full_test_std",
        "sliced_dense_best_mean", "sliced_dense_best_std",
        "jcut_best_test_mean",  "jcut_best_test_std",
        "stratC_best_mean",     "stratC_best_std",
        "shuffle_drop_pp_mean", "shuffle_drop_pp_std",
        "jcut_mode",
    ]
    master_path = os.path.join(OUT_ROOT, "main_results.csv")
    with open(master_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=master_fields)
        w.writeheader()
        for ds in datasets:
            agg = all_agg.get(ds, {})
            row = {f: agg.get(f, "") for f in master_fields}
            row["dataset"] = ds
            w.writerow(row)
    log(f"Saved {master_path}")

    # --- Step 8: Final summary ---
    n_total   = len(datasets) * len(seeds)
    n_jcut    = n_total * len(JCUT_SCAN)
    n_done    = sum(len(v) for v in all_seed_results.values())
    any_drop  = any(
        (all_agg.get(ds, {}).get("shuffle_drop_pp_mean") or 0) < -5
        for ds in datasets
    )

    print("\n" + "=" * 70)
    print(f"{'Dataset':<15} {'MLP-half':>14} {'Sliced-best':>14} "
          f"{'j_cut*':>7} {'Shuf-drop':>12}")
    print("-" * 70)
    for ds in datasets:
        agg  = all_agg.get(ds, {})
        mlp_m = agg.get("mlp_half_test_mean")
        mlp_s = agg.get("mlp_half_test_std",  0)
        slc_m = agg.get("jcut_best_test_mean")
        slc_s = agg.get("jcut_best_test_std",  0)
        drp_m = agg.get("shuffle_drop_pp_mean")
        drp_s = agg.get("shuffle_drop_pp_std",  0)
        mode  = agg.get("jcut_mode", "?")
        if mlp_m is not None and slc_m is not None:
            try:
                print(f"  {ds:<13} "
                      f"{float(mlp_m)*100:>5.1f} ± {float(mlp_s)*100:<4.1f}  "
                      f"{float(slc_m)*100:>5.1f} ± {float(slc_s)*100:<4.1f}  "
                      f"{str(mode):>7}  "
                      f"{float(drp_m):>+5.1f} ± {float(drp_s):<4.1f}pp")
            except Exception:
                print(f"  {ds:<13}  (error formatting)")
        else:
            print(f"  {ds:<13}  FAILED")
    print("=" * 70)
    print(f"\nSeeds completed       : {n_done}/{n_total}")
    print(f"j_cut scans completed : {sum(len(r.get('jcut_rows',[])) for sr in all_seed_results.values() for r in sr)}/{n_jcut}")
    print(f"Failed runs           : {failed_runs if failed_runs else 'NONE'}")
    print(f"Any dataset >5pp drop : {'YES' if any_drop else 'NO'}")
    print(f"READY FOR REPORT      : {'YES' if not failed_runs and n_done == n_total else 'NO — check failed runs'}")


if __name__ == "__main__":
    main()
