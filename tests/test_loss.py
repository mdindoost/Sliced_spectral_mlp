"""
Unit tests for loss functions and weight strategies.

Tests:
  - compute_loss_weights: sums to 1.0 for all strategies
  - sliced_loss: correct shape, requires grad
  - loss_cutoff: only active slices contribute
"""

import numpy as np
import torch
import pytest

from src.training.loss import compute_loss_weights, sliced_loss
from src.models.sliced_mlp import SlicedSpectralMLP


K = 64
N_CLASSES = 7
N = 10


def _make_dummy_logits(n_slices):
    """Generate dummy logits for n_slices slices."""
    return [torch.randn(N, N_CLASSES, requires_grad=True) for _ in range(n_slices)]


def test_uniform_weights_sum_to_one():
    weights = compute_loss_weights("uniform", k=K)
    assert abs(weights.sum() - 1.0) < 1e-5, f"Uniform weights sum to {weights.sum()}"


def test_coarse_weights_sum_to_one():
    weights = compute_loss_weights("coarse", k=K)
    assert abs(weights.sum() - 1.0) < 1e-5, f"Coarse weights sum to {weights.sum()}"


def test_eigenvalue_weights_sum_to_one():
    eigs = np.linspace(0.01, 2.0, K)
    weights = compute_loss_weights("eigenvalue", k=K, eigenvalues=eigs)
    assert abs(weights.sum() - 1.0) < 1e-5, f"Eigenvalue weights sum to {weights.sum()}"


def test_all_strategies_sum_to_one():
    """Parametrized check for all strategies."""
    eigs = np.linspace(0.01, 2.0, K)
    for strategy in ["uniform", "coarse", "eigenvalue"]:
        weights = compute_loss_weights(strategy, k=K, eigenvalues=eigs)
        assert abs(weights.sum() - 1.0) < 1e-5, (
            f"Strategy '{strategy}' weights sum to {weights.sum():.6f}, expected 1.0"
        )


def test_uniform_weights_all_equal():
    """Uniform strategy: all weights equal."""
    weights = compute_loss_weights("uniform", k=K)
    n = K // 2 + 1
    expected = 1.0 / n
    assert np.allclose(weights, expected), "Uniform weights should all be equal"


def test_coarse_weights_decreasing():
    """Coarse strategy: weights strictly decreasing (more weight on coarser slices)."""
    weights = compute_loss_weights("coarse", k=K)
    assert np.all(weights[:-1] >= weights[1:]), "Coarse weights should be non-increasing"


def test_weights_correct_length():
    """Weight vector has length n_slices = k//2 + 1."""
    for k in [32, 64, 128]:
        weights = compute_loss_weights("uniform", k=k)
        assert len(weights) == k // 2 + 1, (
            f"Expected {k//2+1} weights for k={k}, got {len(weights)}"
        )


def test_eigenvalue_strategy_requires_eigenvalues():
    """eigenvalue strategy raises ValueError if eigenvalues=None."""
    with pytest.raises(ValueError, match="eigenvalues must be supplied"):
        compute_loss_weights("eigenvalue", k=K, eigenvalues=None)


def test_unknown_strategy_raises():
    """Unknown strategy name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        compute_loss_weights("bogus", k=K)


def test_sliced_loss_scalar():
    """sliced_loss returns a scalar tensor."""
    n_slices = K // 2 + 1
    all_logits = _make_dummy_logits(n_slices)
    labels = torch.randint(0, N_CLASSES, (N,))
    mask = torch.ones(N, dtype=torch.bool)
    weights = torch.ones(n_slices) / n_slices

    loss = sliced_loss(all_logits, labels, mask, weights)
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"


def test_sliced_loss_requires_grad():
    """sliced_loss output requires grad (can backprop)."""
    n_slices = K // 2 + 1
    all_logits = _make_dummy_logits(n_slices)
    labels = torch.randint(0, N_CLASSES, (N,))
    mask = torch.ones(N, dtype=torch.bool)
    weights = torch.ones(n_slices) / n_slices

    loss = sliced_loss(all_logits, labels, mask, weights)
    loss.backward()
    assert all_logits[0].grad is not None


def test_sliced_loss_cutoff():
    """
    With loss_cutoff=0, only slice j=0 contributes.
    The loss should equal CE(logits[0], labels[mask]) * 1/1 = CE(logits[0]).
    """
    import torch.nn.functional as F
    n_slices = K // 2 + 1
    all_logits = [torch.randn(N, N_CLASSES) for _ in range(n_slices)]
    labels = torch.randint(0, N_CLASSES, (N,))
    mask = torch.ones(N, dtype=torch.bool)
    weights = torch.ones(n_slices) / n_slices

    loss_cutoff = sliced_loss(all_logits, labels, mask, weights, loss_cutoff=0)
    loss_direct = F.cross_entropy(all_logits[0][mask], labels[mask])

    assert abs(loss_cutoff.item() - loss_direct.item()) < 1e-5


def test_model_compute_loss_matches():
    """
    model.compute_loss() should match sliced_loss() with the same weights.
    """
    torch.manual_seed(0)
    model = SlicedSpectralMLP(k=K, n_classes=N_CLASSES, n_layers=1,
                               loss_weights="uniform")
    x = torch.randn(N, K)
    labels = torch.randint(0, N_CLASSES, (N,))
    mask = torch.ones(N, dtype=torch.bool)

    with torch.no_grad():
        all_logits = model(x)

    loss_model = model.compute_loss(all_logits, labels, mask)
    loss_fn    = sliced_loss(all_logits, labels, mask, model.loss_w)

    assert abs(loss_model.item() - loss_fn.item()) < 1e-5
