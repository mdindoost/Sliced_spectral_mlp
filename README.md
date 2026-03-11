# SlicedSpectralMLP

A PyTorch implementation of the SlicedSpectralMLP architecture for node
classification on graphs.

## Installation

```bash
git clone https://github.com/mdindoost/Sliced_spectral_mlp
cd Sliced_spectral_mlp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quickstart

```bash
# Run shuffle diagnostic (fastest check, ~2 min)
python experiments/run_shuffle.py --dataset cora

# Run full experiment
python experiments/run_experiment.py --config experiments/configs/cora.yaml

# Run automatic cutoff selection
python experiments/run_autocutoff.py --dataset cora --strategy c

# Evaluate all strategies + baselines
python experiments/run_baselines.py --dataset cora
```

## Results summary

| Dataset  |  N    | MLP-half | MLP-full | Sliced-best | Strategy-C | Shuffle drop |
|----------|-------|----------|----------|-------------|------------|--------------|
| Cora     | 2485  |  0.7082  |  0.6448  |   0.7399    |   0.7410   |   −17.9 pp   |
| Cornell  |  183  |  0.4054  |  0.4270  |   0.4189    |     n/a    |    −8.1 pp   |
| Actor    | 7600  |  0.2525  |  0.2524  |   0.2511    |     n/a    |    −0.1 pp   |

**Key finding:** Shuffle coarse-drop predicts whether SlicedMLP beats plain MLP.
Large negative drop (Cora −17.9 pp) → spectral ordering is informative → Sliced wins.
Near-zero drop (Actor −0.1 pp) → ordering is noise → Sliced provides no benefit.

## Architecture

Given the **k** smallest eigenvectors of the symmetric normalised Laplacian
as an (N × k) matrix **U**, the model operates on **k//2 + 1** nested prefix
slices of **U**:

| Slice j | Input width d_j | Eigenvectors used |
|---------|-----------------|-------------------|
| 0       | k//2            | 0 … k//2 − 1      |
| 1       | k//2 + 1        | 0 … k//2          |
| ⋮       | ⋮               | ⋮                 |
| k//2    | k               | 0 … k − 1         |

All slices share the same weight matrices **W[l]** of shape (k × k).
Slice j at layer l uses the top-left (d_j × d_j) sub-matrix.
After each layer: ReLU → L2 sphere normalisation.
Each slice has its own output head; heads are **not** shared.

The loss is a weighted sum of per-slice cross-entropies. Three weighting
strategies are provided:

| Strategy    | Description |
|-------------|-------------|
| `uniform`   | Equal weight 1/(k//2+1) for every slice |
| `coarse`    | Linearly decreasing — coarser slices receive more weight |
| `eigenvalue`| w_j ∝ 1/λ_{k//2+j} — smoother eigenvectors get higher weight |

## Project structure

```
sliced_spectral_mlp/
├── src/                        — importable package (pip install -e .)
│   ├── models/                 — SlicedSpectralMLP, StandardMLP
│   ├── data/                   — dataset loaders, eigenvector utilities
│   ├── training/               — trainer, loss functions
│   ├── evaluation/             — metrics, shuffle diagnostic
│   ├── cutoff/                 — strategies A, B, C for auto cutoff
│   └── utils/                  — visualization, I/O
├── experiments/
│   ├── configs/                — YAML configs per dataset
│   ├── run_experiment.py       — main entry point
│   ├── run_shuffle.py          — shuffle diagnostic
│   ├── run_autocutoff.py       — automatic cutoff selection
│   └── run_baselines.py        — evaluate all strategies + baselines
├── scripts/                    — one-off experiment scripts (legacy)
├── tests/                      — unit tests (pytest)
├── outputs/                    — generated at runtime (gitignored)
├── setup.py
├── requirements.txt
└── README.md
```

## Running tests

```bash
python -m pytest tests/ -v
```

## Legacy scripts

Dataset-specific experiment scripts are preserved in `scripts/` for
exact reproducibility of published results:

```bash
python scripts/run_cora_autocutoff.py      # Cora truncation/autocutoff
python scripts/run_cornell_experiments.py  # Cornell 10-split experiments
python scripts/run_pubmed.py               # PubMed with shuffle gate
python scripts/run_actor_squirrel_experiments.py  # Actor + Squirrel
```

## Gradient heatmap interpretation

The `|W[0].grad|` heatmap should exhibit a **bright top-left, dark
bottom-right** pattern:

- The (0,0) entry is the weight connecting eigenvector-0 to
  eigenvector-0 in every slice → receives gradient from all k//2+1 slices.
- The (k-1, k-1) entry only participates in the full slice (j=k//2) →
  receives gradient from one slice only.
- This gradient magnitude gradient is a diagnostic confirming that the
  shared-submatrix architecture is working as intended.

## Key design decisions

- **No message passing** — model is a pure MLP on eigenvector inputs.
- **No shared output heads** — each slice has its own `nn.Linear`.
- **No BatchNorm** — sphere projection is the only normalisation.
- **No dropout** in this version.
- Orthogonal initialisation of W matrices for stable training with sphere
  normalisation.
