# CLAUDE.md — Sliced Spectral MLP

## What This Project Is

Research code for the Sliced Spectral MLP, a node classification architecture
that exploits the prefix structure of graph Laplacian eigenvectors. The core
idea: eigenvector columns are ordered by frequency (low to high), and we
enforce that the model respects this ordering at every layer via shared
submatrix weights.

The two key papers this connects to:
- Our Spectral Basis MLP Study (predecessor, see /docs if present)
- Matryoshka Representation Learning (Kusupati et al. 2022) — closest relative,
  but different in a fundamental way (see src/models/sliced_mlp.py docstring)

## Repo Structure

```
src/
  models/       — SlicedSpectralMLP, StandardMLP
  data/         — load_dataset(), compute_eigenvectors(), shuffle_eigenvectors()
  training/     — train_sliced(), compute_loss_weights(), sliced_loss()
  evaluation/   — accuracy(), per_slice_accuracy(), run_shuffle_diagnostic()
  cutoff/       — strategy_a.py, strategy_b.py, strategy_c.py
  utils/        — save_grad_heatmap(), io helpers
experiments/    — entry point scripts + yaml configs, one per dataset
scripts/        — legacy one-off experiment scripts (exact reproducibility)
tests/          — 28 unit tests covering three critical invariants (see below)
outputs/        — generated at runtime, gitignored (outputs/.gitkeep tracks dir)
notebooks/      — exploratory analysis (currently empty)
```

The two most-called functions are `load_dataset()` (src/data/loaders.py) and
`compute_eigenvectors()` (also src/data/loaders.py, re-exported from
src/data/eigenvectors.py). Do not recompute eigenvectors inline in scripts.

## Critical Invariants — Never Break These

1. **Submatrix constraint.** Slice j must use ONLY W[0:d_j, 0:d_j] at every
   layer. The coarse computation must be a strict sub-computation of the fine
   one. If you refactor the forward pass, verify with tests/test_models.py::test_submatrix_constraint.

2. **Separate output heads.** Each slice has its own nn.Linear head. They are
   NOT shared. This is load-bearing for gradient attribution.

3. **Sphere normalization per slice per layer.** F.normalize(h, p=2, dim=-1)
   after every ReLU, independently for each slice. Do not remove or move this.

4. **Eigenvector ordering.** Eigenvectors must be sorted by ascending eigenvalue
   before slicing. Never shuffle before saving — only experiments/run_shuffle.py
   shuffles, and only for the diagnostic experiment.

5. **Reproducibility.** All experiments used seed=42, k=64, n_layers=2,
   lr=0.01, weight_decay=5e-4. Do not change defaults in experiments/configs/default.yaml.

## Known Bugs and Lessons Learned

### Bug 3 — Model selection with truncated loss (FIXED, 2026-03)

**⚠ CAUTION: If you ever touch checkpoint selection logic in trainer.py, re-read this.**

When `loss_cutoff=j*` is set, slices `j*+1 … n_slices-1` receive **zero gradient**
throughout training. The full slice (j=32, `all_logits[-1]`) is therefore undertrained
and its val accuracy is essentially random noise.

**Wrong (original code):**
```python
val_acc = accuracy(all_logits[-1], labels, val_mask)  # always uses full slice
```

**Correct (fixed code in src/training/trainer.py):**
```python
track_j = loss_cutoff if loss_cutoff is not None else len(all_logits) - 1
track_val = accuracy(all_logits[track_j], labels, val_mask)
```

**Why this mattered:** The bug caused checkpoint selection to track a noisy random
signal when any cutoff was set, corrupting ALL cutoff sensitivity results and
producing sign-flipped shuffle drops on CiteSeer (+14.2pp) and PubMed (+30.7pp).

**Protection:** Regression test at `tests/test_bug3_model_selection.py`. Run it
before and after any refactor of the training loop. The test suite now has 30 tests
(was 28). Do not remove this test.

---

## Key Design Decisions (Do Not Revisit Without Good Reason)

- **Loss cutoff is the main hyperparameter.** Setting loss_cutoff=None trains
  all 33 slices (causes sharing tax on homophilous graphs). Strategy C
  (median eigenvalue threshold) is the automatic selector — see
  src/cutoff/strategy_c.py. It works on Cora, but is unreliable on PubMed
  where the spectrum is more uniform (threshold doesn't isolate informative
  low-frequency slices cleanly). This is a known limitation, not a bug.

- **Strategy B (warmup cutoff) is noisy and unreliable.** It was tested with
  warmup lengths W=10, 20, 30, 50 epochs. The selected cutoff j varies
  significantly across warmup lengths (e.g. j=24 at W=10, different values
  at W=20/30/50) because the per-slice validation curve is noisy at early
  epochs. It never reliably matched or beat Strategy C on Cora. Do not spend
  time trying to fix it — the noisiness is fundamental to early training
  dynamics, not a tuning problem.

- **Gradient heatmap is a diagnostic, not a metric.** Visualize
  |W[0].grad| as a heatmap. Top-left should be bright, bottom-right dark.
  If it's uniform, the submatrix slicing is broken.

## What the Shuffle Diagnostic Tells You

Before running a full experiment on a new dataset, run:
  python experiments/run_shuffle.py --dataset <name>

The coarse drop (shuffle vs unshuffled accuracy at j=0) tells you:
  < -10pp  : architecture will likely help, run full experiment
  -5 to -10pp : marginal, depends on graph size
  > -5pp   : eigenvectors are noise for this task, don't bother

Reference values from completed experiments:
  Cora     = -17.9pp  → helps strongly (Sliced best=0.7399 vs MLP-half=0.7082)
  PubMed   = -10.2pp  → helps (passed the gate, ran full experiment)
  Cornell  =  -8.1pp  → marginal (small graph N=183, noisy estimates)
  Actor    =  -0.1pp  → skip (no spectral signal)
  Squirrel =  +2.7pp  → skip (no spectral signal)

Pattern: larger |drop| predicts Sliced > MLP-full. Actor/Squirrel confirmed
this in the negative direction.

## Config Safety

**Before every full run, use --dry-run to verify the resolved config:**
```
python experiments/run_experiment.py --config experiments/configs/cora.yaml --dry-run
```
This prints every key with its resolved value (defaults merged, CLI overrides applied)
and runs structural checks (types, required keys). It does NOT load the dataset or train.

**validate_config() is called automatically at the start of every experiment.**
It checks:
1. All required keys present: dataset, k, n_layers, lr, weight_decay, epochs, seed
2. lr and weight_decay are floats (YAML silently reads `0.01` as str if quoted)
3. k < N_LCC — eigsh will crash if k >= number of nodes in the LCC
4. n_classes (if set) matches the actual dataset label count

validate_config() lives in src/utils/io.py. Call it at the top of run() in any new
experiment script, before loading the full eigenvector matrix.

## Coding Conventions

- All experiment entry points in experiments/ import only from src/
- Configs are yaml files in experiments/configs/ — hyperparameters never
  hardcoded in scripts
- Always call validate_config(cfg, check_data=True) before training
- Use --dry-run to verify config before a full run
- Outputs always go to outputs/<dataset>/ — never to src/ or experiments/
- Use the existing load_dataset() and compute_eigenvectors() functions —
  do not recompute eigenvectors inline in scripts
- All results tables saved as .txt, all plots as .png

## What Not To Do

- Do not add message passing / graph convolution layers. This is a pure MLP
  on eigenvector inputs. No PyG MessagePassing, no GCNConv, no attention.
- Do not add BatchNorm. Sphere normalization is the only normalization.
- Do not add dropout in the base model (ablation only, clearly flagged).
- Do not change the LCC extraction logic — Cornell especially is sensitive
  to which nodes are included (N=183 LCC vs larger raw graph).
- Do not commit outputs/, checkpoints, or the venv/.

## Multi-Seed Experiment Scripts

Two scripts run the primary multi-seed evaluation. Both use **fixed splits**
and seeds [0,1,2,3,4] for weight init only.

### scripts/run_fixed_split_multiseed.py  ← PRIMARY

The canonical evaluation script for the paper.

Split protocol:
  cora, citeseer, pubmed   — PyG Planetoid fixed splits (LCC-filtered sizes:
                             Cora 122/459/915, CiteSeer 80/328/663, PubMed 60/500/1000)
  amazon_photo, actor,     — fixed stratified 60/20/20 generated ONCE with
  cornell                    seed=100, saved to outputs/fixed_splits/,
                             reused for all 5 seeds

Usage:
  python scripts/run_fixed_split_multiseed.py
  python scripts/run_fixed_split_multiseed.py --dataset cora citeseer
  python scripts/run_fixed_split_multiseed.py --skip-pytest   # skip Bug 3 test

Outputs: outputs/fixed_split_multiseed/{per_seed,aggregated,tables,figures}/

### scripts/run_multiseed.py  ← ALIAS

Thin wrapper around run_fixed_split_multiseed.py.
Identical experiment, outputs to outputs/multiseed/ instead.
Shares the same saved split masks in outputs/fixed_splits/.

Usage:
  python scripts/run_multiseed.py   (same flags as run_fixed_split_multiseed.py)

### What both scripts produce

Per run: 3 tables (main_results, jcut_sensitivity, jcut_stability),
         6 figures (shuffle_drop, per_slice_curves, jcut_sensitivity,
                    main_results_bars, scatter_drop_vs_gain,
                    jcut_stability_heatmap),
         per-seed CSVs, aggregated CSVs, experiment_log.md.

j_cut* selected by VAL accuracy per seed — never by test.
Bug 3 fix verified by pytest before training starts (unless --skip-pytest).

---

## Running Tests

  pip install -e .
  python -m pytest tests/ -v      # runs all 28 tests

The 28 tests cover three test files, one per critical invariant group:
  tests/test_models.py  — submatrix constraint, slice shapes, weight sums
  tests/test_data.py    — eigenvector shape, sort order, shuffle correctness
  tests/test_loss.py    — weight normalization, gradient flow, cutoff behavior

All 28 must pass after any refactor before pushing.
