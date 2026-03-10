# SlicedSpectralMLP

A PyTorch implementation of the SlicedSpectralMLP architecture for node
classification on graphs.

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
Sliced_spectral_mlp/
├── model.py        — SlicedSpectralMLP + compute_loss
├── data.py         — dataset loading + Laplacian eigenvector computation
├── baselines.py    — StandardMLP trained on full / half eigenvectors
├── train.py        — training script
├── eval.py         — evaluation + plots
├── requirements.txt
└── README.md
```

Output directories (created automatically):

```
outputs/
├── checkpoints/best_model.pt
├── grad_heatmaps/epoch_XXXX.png   ← |W[0].grad| heatmap every 10 epochs
└── eval/
    ├── spectral_resolution_curve.png
    └── training_curves_coarse_vs_full.png
```

## Setup

```bash
# Using the existing Spectral-Basis venv (already has all dependencies):
/home/md724/Spectral-Basis/venv/bin/python train.py

# Or install dependencies into a fresh environment:
pip install -r requirements.txt
python train.py
```

## Usage

### Train on Cora (default)

```bash
python train.py
```

Prints per-slice validation accuracy every epoch; saves gradient heatmaps
to `outputs/grad_heatmaps/` every 10 epochs; saves best checkpoint to
`outputs/checkpoints/best_model.pt`.

### Train on Cornell (heterophilous)

```bash
python train.py --dataset cornell
```

### Use a different loss weighting strategy

```bash
python train.py --loss_weights coarse
python train.py --loss_weights eigenvalue
```

### Full option list

```
--dataset      {cora, citeseer, cornell}   (default: cora)
--k            int   number of eigenvectors (default: 64)
--n_layers     int   number of shared hidden layers (default: 2)
--lr           float Adam learning rate (default: 0.01)
--wd           float weight decay (default: 5e-4)
--epochs       int   training epochs (default: 200)
--loss_weights {uniform, coarse, eigenvalue} (default: uniform)
```

### Evaluate and compare against baselines

```bash
python eval.py
```

Trains SlicedSpectralMLP with all three weight strategies and both
StandardMLP baselines from scratch, then prints a comparison table and
saves two plots to `outputs/eval/`.

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
