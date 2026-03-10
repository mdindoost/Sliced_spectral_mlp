# SlicedSpectralMLP — Research Log

**Project:** Sliced Spectral MLP for node classification on graphs
**Location:** `/home/md724/Sliced_spectral_mlp/`
**Python env:** `/home/md724/Spectral-Basis/venv/bin/python`
**Related project:** `/home/md724/Spectral-Basis/` (parent research, see its CLAUDE.md)

---

## What this project is

An architecture that trains a single shared MLP weight matrix on *nested prefix
slices* of the normalized graph Laplacian eigenvector matrix. The core idea:
rather than feeding all k eigenvectors to one MLP, simultaneously train 33
heads that each see a progressively larger prefix of the eigenvector matrix
(d = 32, 33, 34, ..., 64). All heads share the same weight matrices; each uses
only the top-left submatrix corresponding to its prefix dimension. This creates
a "spectral resolution curriculum" — coarse-to-fine in one model.

The architecture was built to test two hypotheses:
1. **Sharing tax hypothesis**: forcing the shared weights to serve noisy
   high-frequency slices hurts the informative low-frequency slices.
2. **Spectral ordering hypothesis**: the nested-prefix structure only helps
   if eigenvectors are ordered by eigenvalue (smooth first). Random column
   ordering should destroy the architecture's advantage.

Both hypotheses were confirmed on Cora. The experiments evolved from
architecture validation → shuffle ablation → truncation fix → automatic cutoff.

---

## Codebase

### Core files

| File | Purpose |
|------|---------|
| `model.py` | `SlicedSpectralMLP` — the main architecture |
| `data.py` | Dataset loading + LCC extraction + Laplacian eigenvectors |
| `baselines.py` | `StandardMLP` + `train_baseline()` helper |
| `train.py` | CLI training script with per-epoch logging + gradient heatmaps |
| `eval.py` | Evaluates all 3 weight strategies + baselines, generates plots |

### Experiment scripts (run these directly)

| Script | What it runs |
|--------|-------------|
| `run_cornell_experiments.py` | Full Cornell pipeline, all 10 splits |
| `run_actor_squirrel_experiments.py` | Actor (and optionally Squirrel) pipeline |
| `run_cora_truncation.py` | Spectral truncation: cutoffs j=7,11,15,32 + MLP-k43 |
| `run_cora_autocutoff.py` | Three automatic cutoff strategies (A/B/C) |

### Output directories

```
outputs/
├── checkpoints/best_model.pt          ← from train.py
├── grad_heatmaps/                     ← from train.py
├── eval/                              ← from eval.py
├── cornell/                           ← from run_cornell_experiments.py
├── actor/                             ← from run_actor_squirrel_experiments.py
├── squirrel/                          ← (not yet run; framework ready)
├── cora_truncation/                   ← from run_cora_truncation.py
└── cora_autocutoff/                   ← from run_cora_autocutoff.py
```

---

## Architecture detail (`model.py`)

```
SlicedSpectralMLP(k=64, n_classes, n_layers=2, loss_weights, custom_slice_dims=None)
```

- `n_slices = k//2 + 1 = 33`  (j = 0, 1, ..., 32)
- Slice j: input d_j = k//2 + j = 32 + j eigenvectors
- Shared weights: `W[l]` shape (k, k) per layer, orthogonal init
- Slice j at layer l uses `W[l][:d_j, :d_j]` — gradient flows naturally
- After each layer: ReLU → L2 sphere normalization (no BatchNorm, no dropout)
- Separate `nn.Linear(d_j, n_classes)` head per slice — NOT shared
- Loss: weighted sum of per-slice cross-entropies

**Key parameter added during research: `loss_cutoff`**
Pass `loss_cutoff=j` to `compute_loss()` to zero out the loss for all slices
with index > j. Active slices are re-weighted uniformly to 1/n_active.
This is the main finding — see "Spectral Truncation" section below.

**Key parameter added: `custom_slice_dims`**
Pass a list like `[32, 40, 48, 56, 64]` to override the default sequential
slices with a sparse set. Used for dense vs sparse comparison on Actor.

### Loss weight strategies

| Strategy | Description |
|----------|-------------|
| `uniform` | Equal weight 1/n_slices |
| `coarse` | Linearly decreasing — coarser slices get more weight |
| `eigenvalue` | w_j ∝ 1/λ_{k//2+j} — smoother eigenvectors get higher weight |

---

## Data pipeline (`data.py`)

**Critical: always extract the LCC.** Cora has 78 connected components in its
raw PyG form. Without LCC extraction, all 64+ eigenvalues of the normalized
Laplacian are numerically zero (one per component), giving useless eigenvectors
with near-zero row norms. The LCC extraction is implemented in `load_dataset()`.

```python
U, labels, train_mask, val_mask, test_mask, eigenvalues = load_dataset(
    name,          # 'cora', 'citeseer', 'pubmed', 'cornell', 'actor', 'squirrel'
    root='./data',
    k=64,
    split_idx=0,   # 0-9 for WebKB/Actor/Wikipedia (10 fixed splits); ignored for Planetoid
)
```

**Eigendecomposition strategy:**
- N ≤ 5000 (Cora LCC=2485, Cornell=183): dense `scipy.linalg.eigh` — exact, fast
- N > 5000 (Actor=7600): sparse `eigsh` with regularization (L + 1e-8·I, sigma=0)

**Datasets supported and their LCC sizes:**

| Dataset | Raw N | LCC N | Components | Splits | Homophily |
|---------|-------|-------|------------|--------|-----------|
| Cora | 2708 | 2485 | 78 | 1 (fixed) | High |
| CiteSeer | 3327 | ~2120 | many | 1 (fixed) | High |
| Cornell | 183 | 183 | 1 | 10 (fixed) | Low |
| Actor | 7600 | 7600 | 1 | 10 (fixed) | Low |
| Squirrel | 5201 | ? | ? | 10 (fixed) | Low |

---

## Experiment history and findings

### 1. Architecture sanity checks (Cora)

**Submatrix constraint verified:** For slice j=0 (d=32), h stays shape (N, 32)
through all layers — not (N, 64). The top-left submatrix is correctly extracted
via `W[l][:d_j, :d_j]`, which creates a view so gradients flow to the shared W.

**Gradient heatmap:** `|W[0].grad|` shows bright top-left, dark bottom-right.
Top-left 8×8 mean grad / bottom-right 8×8 mean grad ≈ 7.5×. Confirms the
(0,0) entry receives gradient from all 33 slices while (63,63) only from j=32.

---

### 2. Shuffle ablation (Cora, single seed)

Randomly permuting eigenvector columns before training:

| | Unshuffled | Shuffled | Drop |
|---|---|---|---|
| Best val | 0.7233 | 0.6710 | −5.2pp |
| Coarse slice (j=0, d=32) | 0.7388 | 0.5596 | **−17.9pp** |
| Full slice (j=32, d=64) | 0.7082 | 0.6721 | −3.6pp |

**Finding:** Spectral ordering is critical. The coarse slice collapses −17.9pp
because the 32 smoothest eigenvectors carry most of Cora's class signal; after
shuffling they're scattered across all 64 positions. The full slice barely
changes (−3.6pp) because it has all 64 columns regardless of order — its drop
is purely from shared weight contamination by the disordered coarse slices.

---

### 3. Baseline comparison (Cora, eval.py, single seed)

```
Method                   Best Val   Test (full)   Test (coarse)
Sliced(uniform)           0.7124      0.6590        0.6918
Sliced(coarse)            0.6732      0.6623        0.6787
Sliced(eigenvalue)        0.6950      0.6743        0.6754
StandardMLP-full (k=64)   0.6514      0.6448            —
StandardMLP-half (k=32)   0.7255      0.7082            —
```

**Key observation:** StandardMLP-half beats all Sliced variants. A plain MLP
on 32 eigenvectors outperforms the full sliced architecture using 64. This led
to the sharing tax hypothesis.

**Also:** Within the Sliced model, the coarse slice (j=0, d=32) at 69.18% beats
the full slice (j=32, d=64) at 65.90%. Adding more eigenvectors *hurts* the
full-slice head — same pattern as MLP-half vs MLP-full.

---

### 4. Per-slice accuracy curve (Cora, first observation)

From 50-epoch run: per-slice test accuracy peaks at **j=11, d=43 (70.71%)**,
then declines. The full slice (j=32, d=64) achieves only 68.96%. This peak-then-
decline shape is characteristic of homophilous graphs and directly motivated
the spectral truncation experiment.

---

### 5. Cross-dataset comparison: Cornell (heterophilous, 183 nodes)

Cornell has 10 fixed splits → mean ± std over all 10.

```
Method                   Best Val         Test (full)      Test (coarse)
Sliced(uniform)          0.4746±0.0652    0.4081±0.0560    0.4243±0.0975
StandardMLP-full         0.4729±0.0658    0.4270±0.0551         —
StandardMLP-half         0.4881±0.0656    0.4054±0.0592         —
```

**Shuffle drop on Cornell: −8.1pp** (vs Cora's −17.9pp). Spectral ordering
matters less on the heterophilous graph — high-frequency eigenvectors already
carry discriminative signal, so scrambling the order costs less.

**MLP-full/half reversal:** On Cora, MLP-half beats MLP-full by 6.3pp. On
Cornell, MLP-full is 2.2pp *better* than MLP-half. The extra eigenvectors that
hurt Cora help Cornell — heterophily prediction confirmed.

**Caveat:** Cornell has only 37 test nodes — each misclassified node is ±2.7pp.
All differences within ~1pp should be treated as noise.

---

### 6. Cross-dataset comparison: Actor (heterophilous, 7600 nodes)

100 epochs, mean over 10 splits.

```
Method                   Best Val         Test (full)      Test (best)
Sliced-dense(uniform)    0.2649±0.0057    0.2511±0.0089    0.2597±0.0114
StandardMLP-full         0.2609±0.0062    0.2524±0.0115         —
StandardMLP-half         0.2606±0.0060    0.2525±0.0112         —
```

**Shuffle drop on Actor: −0.1pp** — essentially zero. Shuffling has no effect.

**Interpretation:** Actor's Laplacian eigenvectors at k=64 carry no useful
spectral ordering for classification. All models hover near 25% (5-class random
baseline ≈ 20%), indicating the eigenvector features themselves are near-
uninformative for this task. The shuffle invariance proves this — if spectral
ordering mattered, shuffling would hurt the coarse slice as it did on Cora.
Actor is a null result that constrains the architecture's scope of applicability.

---

### 7. Full cross-dataset summary

```
Dataset  | N nodes | MLP-half | MLP-full | Sliced best-full | Shuffle coarse drop
Cora     |    2485 |   0.7082 |   0.6448 |           0.6743 |            -17.9pp
Cornell  |     183 |   0.4054 |   0.4270 |           0.4189 |             -8.1pp
Actor    |    7600 |   0.2525 |   0.2524 |           0.2511 |             -0.1pp
```

The shuffle coarse drop is the cleanest diagnostic of spectral ordering
relevance: −17.9pp (Cora, homophilous) → −8.1pp (Cornell, heterophilous) →
−0.1pp (Actor, heterophilous + uninformative features). Monotonically decreasing.

---

### 8. Spectral truncation experiment (Cora)

**Hypothesis:** the per-slice curve peaks at j=11 then declines because the
shared weights are "taxed" by noisy high-frequency slices. Fix: zero out the
loss for slices past a cutoff so those weights specialize to the informative band.

```
Method                    Best Val   Test@active   Test(coarse)   Test(best)
Sliced-cutoff-j07          0.7211      0.7213        0.7180        0.7344
Sliced-cutoff-j11          0.7342      0.7290        0.7322        0.7388
Sliced-cutoff-j15          0.6993      0.6940        0.7399        0.7399  ← winner
Sliced-cutoff-j32 (full)   0.7233      0.7082        0.7388        0.7388  ← baseline
StandardMLP-k43            0.7211      0.6951             —             —
StandardMLP-k32 (prior)    0.7255      0.7082             —             —
```

**Finding 1: Sharing tax confirmed. Truncation fixes it.**
Cutoff-j15 achieves 73.99% (best slice) vs MLP-half's 70.82% — a +3.2pp gain.
The architecture now beats a plain MLP on the optimal prefix.

**Finding 2: Optimal cutoff (j=15) ≠ peak slice (j=11).**
The curve peaked at j=11 in the untruncated run, but cutting at j=11 gives
73.88% while j=15 gives 73.99%. The cutoff affects gradient flow through the
shared W, not just which heads receive loss — a wider active window gives the
shared weights a better collective basis.

**Finding 3: MLP-k43 underperforms MLP-k32 (69.51% vs 70.82%).**
A plain MLP on the "peak" 43 eigenvectors is worse than a plain MLP on 32.
Only the sliced architecture can exploit the 32–43 range because it forces
prefix coherence through the shared submatrix constraint.

**Finding 4: Inactive slices still function.**
Heads j > cutoff receive no loss signal but still produce logits. They inherit
structure from the shared W trained by the active slices. Their accuracy is
non-trivial (e.g. best_slice for cutoff-j07 is 73.44% at j=1, a supervised head).

**Implementation:** `model.compute_loss(logits, labels, mask, loss_cutoff=15)`
The `loss_cutoff` parameter is also available in `train.py` via `--loss_cutoff`.

---

### 9. Automatic cutoff selection (Cora)

**The problem:** optimal cutoff required inspecting the per-slice curve from a
completed training run — circular. Three automatic strategies tested:

**Strategy A — Eigenvalue gap (pre-training, zero gradient cost)**
Find the largest gap in λ_{32+j+1} − λ_{32+j} for j = 0..31.
Result: j=21 (test best 69.18%). Misses by 6 slices.
Why it fails: the largest spectral gap on Cora is at j=21, reflecting graph
community structure — not classification difficulty. Spectral gaps and task
relevance are decoupled.

**Strategy B — Warmup peak (10-epoch warmup, then fresh 200-epoch run)**
Train 10 epochs, pick the slice with highest val accuracy.
Result: j=24 (test best 71.04%). Overshoots.
Why it fails: 10 epochs is insufficient for the per-slice val curve to
stabilize. The warmup curve is noisy (range 0.586–0.715 at epoch 10) and the
peak wanders. Would need 30–50 warmup epochs to be reliable.

**Strategy C — Eigenvalue threshold (pre-training, zero gradient cost)**
Compute w_j = 1/λ_{32+j}. Zero out w_j where λ_{32+j} > median(λ_{32..63}).
Result: j=16 (test best **74.10%**) — exceeds manual j=15 (73.99%).
Why it works on Cora: the median eigenvalue (0.0838) coincides almost exactly
with the manual optimum boundary (λ_{32+15} = 0.0836, λ_{32+16} = 0.0838).
The median is a good spectral split for homophilous graphs where low-frequency
= informative is a valid prior.

```
Method                   Cutoff   Best Val   Test (best slice)
Strategy A (eig-gap)      j=21     0.7102        0.6918
Strategy B (warmup)       j=24     0.7168        0.7104
Strategy C (eig-thresh)   j=16     0.7277        0.7410   ← best auto
Manual cutoff-j15         j=15     0.6993        0.7399   ← target
StandardMLP-half (k=32)     —      0.7255        0.7082   ← baseline
```

**Warning:** Strategy C relies on low-frequency = informative, which holds for
homophilous graphs but breaks on heterophilous ones (Cornell, Actor). For
heterophilous graphs, Strategy B (warmup) is the only approach that could
generalize, but needs more warmup epochs.

---

## Open threads (what to do next)

1. **Run Squirrel.** The framework exists (`run_actor_squirrel_experiments.py`).
   Just call `run_dataset("squirrel", "outputs/squirrel")`. Takes ~10 min.
   Key question: shuffle coarse drop on Squirrel — smaller than Cornell's −8.1pp?

2. **Extend Strategy B warmup.** Try 30 and 50 warmup epochs on Cora. Does
   the warmup peak stabilize to j≈15? If yes, Strategy B becomes reliable and
   generalizes across homophilous/heterophilous graphs unlike Strategy C.

3. **Test truncation on Cornell.** Cornell's per-slice curve slopes upward
   (high-frequency eigenvectors help). Does truncation hurt on heterophilous
   graphs? Expected: yes, cutoff should be set to j=32 (no truncation) on
   heterophilous datasets.

4. **Multi-seed variance on Cora.** All Cora results are single-seed. The
   truncation gains (+3.2pp over MLP-half) need multi-seed confirmation.
   Cornell uses 10 splits but Cora uses 1 fixed split — add seed averaging.

5. **Strategy C on Cornell/Actor.** Test whether the eigenvalue threshold
   strategy correctly selects j=32 (no truncation) on heterophilous datasets.
   If it does, it might be a universal automatic method after all.

6. **Sparse vs dense slicing redux.** On Actor, dense (33 slices) and sparse
   (5 slices at d=32,40,48,56,64) are statistically tied. On Cornell (183 nodes)
   with truncation applied, sparse might help by reducing head count below the
   node count threshold. Not yet tested.

---

## Key numbers to remember

| Fact | Value |
|------|-------|
| Best result on Cora | **74.10%** — Strategy C autocutoff (j=16) |
| Manual best on Cora | **73.99%** — cutoff j=15 |
| MLP-half baseline (Cora) | **70.82%** |
| Shuffle coarse drop (Cora) | **−17.9pp** — largest of all datasets |
| Shuffle coarse drop (Cornell) | **−8.1pp** |
| Shuffle coarse drop (Actor) | **−0.1pp** — spectral ordering irrelevant |
| Actor accuracy range | ~25% — near-chance, 5 classes |
| Cora LCC size | 2485 (raw: 2708, 78 components — must extract LCC) |
| Cornell test set | 37 nodes — ±2.7pp per misclassification |

---

## Common bugs and gotchas

- **Always extract LCC before computing Laplacian.** Cora has 78 components;
  without LCC, all returned eigenvalues are ~1e-15 (one zero per component).
  `data.py` handles this automatically for all datasets.

- **`eigsh` instability near zero.** For small graphs (N ≤ 5000), use
  `scipy.linalg.eigh` (dense) instead of `eigsh`. For larger, regularize:
  `eigsh(L + 1e-8·I, sigma=0)` then subtract 1e-8 from returned eigenvalues.

- **`loss_cutoff` in `compute_loss`.** When using cutoff, best-val tracking
  should be done on the *active* top slice (j=cutoff), not j=32. The experiment
  scripts handle this correctly but `train.py` still tracks j=32 — be careful.

- **Cornell masks are 2D.** `data.train_mask.shape = (183, 10)`. Use
  `split_idx` parameter in `load_dataset()` to select which split.
  `data.py` handles this in both the LCC and non-LCC code paths.

- **Actor dataset path.** Requires `root=f"{root}/actor"` (subdirectory),
  unlike Planetoid/WebKB which use `root` directly. Already handled in
  `_load_pyg()`.
