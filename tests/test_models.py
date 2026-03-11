"""
Unit tests for SlicedSpectralMLP.

Tests:
  - Correct number of slices (k//2 + 1 = 33 for k=64)
  - Correct output shapes per slice
  - Submatrix constraint: slice j only uses top-left d_j × d_j of W[l]
"""

import torch
import pytest

from src.models.sliced_mlp import SlicedSpectralMLP


def test_slice_count():
    """n_slices = k//2 + 1 = 33 for k=64."""
    model = SlicedSpectralMLP(k=64, n_layers=2, n_classes=7)
    assert model.n_slices == 33


def test_slice_output_shapes():
    """Each slice head outputs (N, n_classes) tensors."""
    model = SlicedSpectralMLP(k=64, n_layers=2, n_classes=7)
    x = torch.randn(10, 64)
    logits = model(x)
    assert len(logits) == 33
    assert logits[0].shape == (10, 7),  f"Expected (10,7), got {logits[0].shape}"
    assert logits[32].shape == (10, 7), f"Expected (10,7), got {logits[32].shape}"


def test_coarse_slice_input_width():
    """Coarse slice j=0 uses d_0 = k//2 = 32 eigenvectors."""
    k = 64
    model = SlicedSpectralMLP(k=k, n_layers=2, n_classes=7)
    assert model.slice_dims[0] == k // 2


def test_full_slice_input_width():
    """Full slice j=k//2 uses d_{k//2} = k = 64 eigenvectors."""
    k = 64
    model = SlicedSpectralMLP(k=k, n_layers=2, n_classes=7)
    assert model.slice_dims[-1] == k


def test_submatrix_constraint():
    """
    Slice j=0 only touches the top-left (k//2 × k//2) submatrix of W[l].
    We verify this by zeroing out the bottom-right of W[0] and checking
    that the coarse slice output is unchanged, while the full slice changes.
    """
    torch.manual_seed(0)
    k = 64
    half = k // 2
    model = SlicedSpectralMLP(k=k, n_layers=1, n_classes=7)
    x = torch.randn(10, k)

    with torch.no_grad():
        logits_before = model(x)
        coarse_before = logits_before[0].clone()
        full_before   = logits_before[-1].clone()

        # Zero out the bottom-right of W[0] (only affects slices that use it)
        model.W[0].data[half:, :] = 0.0
        model.W[0].data[:, half:] = 0.0

        logits_after = model(x)
        coarse_after = logits_after[0]
        full_after   = logits_after[-1]

    # Coarse slice j=0 only uses top-left half×half — unaffected by zeroing bottom-right
    assert torch.allclose(coarse_before, coarse_after), (
        "Coarse slice should be unaffected by zeroing bottom-right of W[0]"
    )
    # Full slice j=k//2 uses the entire W[0] — it should change
    assert not torch.allclose(full_before, full_after), (
        "Full slice should be affected by zeroing bottom-right of W[0]"
    )


def test_loss_weights_sum_to_one():
    """Loss weights pre-computed in model must sum to 1."""
    import numpy as np
    for strategy in ["uniform", "coarse"]:
        model = SlicedSpectralMLP(k=64, n_layers=2, n_classes=7,
                                   loss_weights=strategy)
        w_sum = model.loss_w.sum().item()
        assert abs(w_sum - 1.0) < 1e-5, (
            f"Strategy '{strategy}' weights sum to {w_sum}, expected 1.0"
        )

    eigs = np.linspace(0.1, 2.0, 64)
    model = SlicedSpectralMLP(k=64, n_layers=2, n_classes=7,
                               loss_weights="eigenvalue", eigenvalues=eigs)
    w_sum = model.loss_w.sum().item()
    assert abs(w_sum - 1.0) < 1e-5, f"Eigenvalue weights sum to {w_sum}"


def test_custom_slice_dims():
    """Custom slice dims override the default sequential dims."""
    custom = [32, 40, 48, 56, 64]
    model = SlicedSpectralMLP(k=64, n_layers=2, n_classes=7,
                               custom_slice_dims=custom)
    assert model.n_slices == 5
    x = torch.randn(10, 64)
    logits = model(x)
    assert len(logits) == 5
    assert all(lg.shape == (10, 7) for lg in logits)


def test_forward_no_grad():
    """Forward pass runs without errors in eval mode."""
    model = SlicedSpectralMLP(k=64, n_layers=2, n_classes=7)
    model.eval()
    x = torch.randn(5, 64)
    with torch.no_grad():
        logits = model(x)
    assert len(logits) == 33
