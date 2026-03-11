"""
Shuffle diagnostic: compare SlicedSpectralMLP with and without
eigenvector ordering.

A large coarse-slice accuracy drop under shuffling indicates that the
model relies on the spectral ordering (low-frequency eigenvectors first),
which is the intended behaviour for homophilous graphs.
"""

from __future__ import annotations

import torch

from src.models.sliced_mlp import SlicedSpectralMLP
from src.evaluation.metrics import accuracy


def run_shuffle_diagnostic(
    U: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    k: int,
    n_classes: int,
    n_layers: int = 2,
    seed: int = 42,
    epochs: int = 200,
    lr: float = 0.01,
    wd: float = 5e-4,
) -> dict:
    """
    Run the shuffle diagnostic on a dataset.

    Trains SlicedSpectralMLP (uniform weights) on the original and shuffled
    eigenvectors; returns accuracy statistics and the coarse-slice drop.

    Args:
        U:          (N, k) eigenvector matrix.
        labels:     (N,) class labels.
        train_mask: (N,) boolean training mask.
        val_mask:   (N,) boolean validation mask.
        test_mask:  (N,) boolean test mask.
        k, n_classes, n_layers, seed, epochs, lr, wd: hyperparameters.

    Returns:
        dict with keys:
            unshuffled_coarse, unshuffled_full, unshuffled_val
            shuffled_coarse, shuffled_full
            coarse_drop_pp, full_drop_pp
            slice_test_unshuffled, slice_test_shuffled
    """
    def _train_and_eval(U_in, seed_val):
        torch.manual_seed(seed_val)
        model = SlicedSpectralMLP(k=k, n_classes=n_classes, n_layers=n_layers,
                                   loss_weights="uniform")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        best_val, best_state = 0.0, None

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            lg = model(U_in)
            model.compute_loss(lg, labels, train_mask).backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                lg = model(U_in)
            v = accuracy(lg[-1], labels, val_mask)
            if v > best_val:
                best_val = v
                best_state = {kk: vv.clone() for kk, vv in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            lg = model(U_in)
        slice_test = [accuracy(l, labels, test_mask) for l in lg]
        return best_val, slice_test

    # Unshuffled
    best_val_u, st_u = _train_and_eval(U, seed)

    # Shuffled
    perm = torch.randperm(k, generator=torch.Generator().manual_seed(seed))
    U_shuf = U[:, perm]
    _, st_s = _train_and_eval(U_shuf, seed)

    coarse_drop_pp = (st_s[0] - st_u[0]) * 100
    full_drop_pp   = (st_s[-1] - st_u[-1]) * 100

    return {
        "unshuffled_coarse":   st_u[0],
        "unshuffled_full":     st_u[-1],
        "unshuffled_val":      best_val_u,
        "shuffled_coarse":     st_s[0],
        "shuffled_full":       st_s[-1],
        "coarse_drop_pp":      coarse_drop_pp,
        "full_drop_pp":        full_drop_pp,
        "slice_test_unshuffled": st_u,
        "slice_test_shuffled":   st_s,
    }
