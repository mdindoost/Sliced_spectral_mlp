"""
Baseline models for comparison with SlicedSpectralMLP.

StandardMLP
-----------
A standard 3-layer MLP (2 hidden layers + output head).
Hidden dimension matches k for a fair comparison with SlicedSpectralMLP,
which uses k-dimensional hidden representations in its largest slice.
No sphere normalisation, no shared weights.

Two variants are evaluated:
  - 'full'  : trained on all k eigenvectors  (matches the full slice j=k//2)
  - 'half'  : trained on only k//2 eigenvectors (matches the coarse slice j=0)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardMLP(nn.Module):
    """Two-hidden-layer MLP.  Architecture:

        Linear(n_in, hidden) → ReLU
        Linear(hidden, hidden) → ReLU
        Linear(hidden, n_classes)

    n_layers controls the number of hidden layers (each of size hidden_dim).
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class RowNormMLP(nn.Module):
    """MLP with row-wise L2 normalisation at input and after each hidden layer.

    No bias terms — sphere normalisation already removes the mean-field
    component, so biases are redundant and can destabilise training.

    Two standard variants:
        RowNormMLP-half : input_dim = k // 2
        RowNormMLP-full : input_dim = k

    Note: RowNormMLP normalizes the full input vector, not per-slice.
    Per-slice normalization only applies to SlicedSpectralMLP with
    use_row_norm_input=True. These are different operations.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=1)
        x = F.relu(self.fc1(x))
        x = F.normalize(x, p=2, dim=1)
        x = F.relu(self.fc2(x))
        x = F.normalize(x, p=2, dim=1)
        return self.fc3(x)


# ---------------------------------------------------------------------------
# Training helper shared by baselines (and reused in eval.py)
# ---------------------------------------------------------------------------

def train_baseline(
    model: nn.Module,
    X: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    lr: float = 0.01,
    wd: float = 5e-4,
    epochs: int = 200,
) -> tuple[float, float]:
    """
    Train a StandardMLP and return (best_val_acc, test_acc_at_best_val).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val_acc = 0.0
    best_state: dict | None = None

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X).argmax(dim=-1)
            val_acc = (preds[val_mask] == labels[val_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best checkpoint and evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=-1)
        test_acc = (preds[test_mask] == labels[test_mask]).float().mean().item()

    return best_val_acc, test_acc
