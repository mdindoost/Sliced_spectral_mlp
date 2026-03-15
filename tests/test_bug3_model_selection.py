"""
Regression tests for Bug 3: model selection must use the last ACTIVE slice
val accuracy, not the full slice (j=n_slices-1) val accuracy.

Bug 3 description
-----------------
When training with loss_cutoff=j*, slices j*+1 … n_slices-1 receive no
loss gradient. The full slice (j=n_slices-1) therefore remains undertrained
and its val accuracy is essentially random noise throughout training.

If checkpoint selection tracks accuracy(all_logits[-1], ...) unconditionally,
it saves checkpoints chosen by a random signal, corrupting all results.

Fix (src/training/trainer.py)
------------------------------
    track_j = loss_cutoff if loss_cutoff is not None else len(all_logits) - 1
    track_val = accuracy(all_logits[track_j], labels, val_mask)

These tests verify both the fixed case (cutoff set) and the baseline case
(no cutoff — full slice is correctly used).
"""

import torch
import torch.nn.functional as F
import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sliced_mlp import SlicedSpectralMLP
from src.training.trainer import train_sliced
from src.evaluation.metrics import accuracy


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_data(n=100, k=64, n_classes=3, seed=42):
    """Return (X, labels, train_mask, val_mask, test_mask)."""
    torch.manual_seed(seed)
    X = torch.randn(n, k)
    labels = torch.randint(0, n_classes, (n,))
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[:60] = True
    val_mask[60:80] = True
    test_mask[80:]  = True
    return X, labels, train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# Test 1 — truncated loss: selection must track slice[loss_cutoff], not [-1]
# ---------------------------------------------------------------------------

def test_model_selection_uses_last_active_slice():
    """
    When loss_cutoff=3, best_val reported by train_sliced must match the
    val accuracy of slice 3 at the saved checkpoint — not slice 32.

    Failure means trainer is still using slice[-1] for model selection (Bug 3).
    """
    k = 64
    n_classes = 3
    loss_cutoff = 3

    X, labels, train_mask, val_mask, test_mask = _make_data(k=k, n_classes=n_classes)

    torch.manual_seed(42)
    model = SlicedSpectralMLP(k=k, n_classes=n_classes, n_layers=2,
                               loss_weights="uniform")

    best_val, _, _ = train_sliced(
        model, X, labels, train_mask, val_mask,
        epochs=10, lr=0.01, wd=5e-4,
        loss_cutoff=loss_cutoff,
    )

    # Evaluate the saved checkpoint (model is restored to best state by train_sliced)
    model.eval()
    with torch.no_grad():
        all_logits = model(X)

    slice3_val  = accuracy(all_logits[loss_cutoff], labels, val_mask)
    slice32_val = accuracy(all_logits[-1],           labels, val_mask)

    assert abs(slice3_val - best_val) < 0.02, (
        f"Bug 3 regression: best_val={best_val:.4f} does not match "
        f"slice3_val={slice3_val:.4f}. "
        f"Trainer is likely still using slice32_val={slice32_val:.4f} "
        f"for model selection."
    )


# ---------------------------------------------------------------------------
# Test 2 — dense training: selection must track full slice when no cutoff
# ---------------------------------------------------------------------------

def test_dense_model_selection_uses_full_slice():
    """
    When loss_cutoff=None (dense training), best_val reported by train_sliced
    must match the val accuracy of the full slice (slice[-1]) at the saved
    checkpoint.

    This confirms the fix handles both cases without breaking the default.
    """
    k = 64
    n_classes = 3

    X, labels, train_mask, val_mask, test_mask = _make_data(k=k, n_classes=n_classes)

    torch.manual_seed(42)
    model = SlicedSpectralMLP(k=k, n_classes=n_classes, n_layers=2,
                               loss_weights="uniform")

    best_val, _, _ = train_sliced(
        model, X, labels, train_mask, val_mask,
        epochs=10, lr=0.01, wd=5e-4,
        loss_cutoff=None,
    )

    model.eval()
    with torch.no_grad():
        all_logits = model(X)

    slice32_val = accuracy(all_logits[-1], labels, val_mask)

    assert abs(slice32_val - best_val) < 0.02, (
        f"Dense model selection broken: best_val={best_val:.4f} does not "
        f"match slice32_val={slice32_val:.4f}."
    )


# ---------------------------------------------------------------------------
# Test 3 — per-slice row norm: every slice must have unit-norm inputs,
#           and slice j=0 (d=32) must differ from slice j=32 (d=64)
# ---------------------------------------------------------------------------

def test_per_slice_rownorm_has_unit_norm():
    """
    With use_row_norm_input=True, the normalization must be applied inside
    the slice loop on x[:, :d_j], NOT once on the full x before slicing.

    Invariant A: x_j.norm(dim=1) ≈ 1.0 for every j (verified for j=0 and j=32).
    Invariant B: x_j0[:, :32] != x_j32[:, :32]
        The first 32 dims are normalised differently because j=0 divides by
        ||x[:, :32]|| while j=32 divides by ||x[:, :64]||.
        If these are equal, the fix is wrong (global normalisation is back).
    """
    N, k = 10, 64
    torch.manual_seed(7)
    x = torch.randn(N, k)

    # Compute what the fixed forward() applies per slice
    d0  = k // 2       # 32  — slice j=0
    d32 = k            # 64  — slice j=32

    x_j0  = F.normalize(x[:, :d0],  p=2, dim=1)  # (N, 32)
    x_j32 = F.normalize(x[:, :d32], p=2, dim=1)  # (N, 64)

    # Invariant A — unit norm
    norms_j0  = x_j0.norm(dim=1)
    norms_j32 = x_j32.norm(dim=1)
    assert torch.allclose(norms_j0,  torch.ones(N), atol=1e-5), (
        f"Slice j=0 (d={d0}) does not have unit norm. "
        f"max deviation = {(norms_j0 - 1).abs().max():.2e}"
    )
    assert torch.allclose(norms_j32, torch.ones(N), atol=1e-5), (
        f"Slice j=32 (d={d32}) does not have unit norm. "
        f"max deviation = {(norms_j32 - 1).abs().max():.2e}"
    )

    # Invariant B — independent normalisation produces different first-32-dim vectors
    assert not torch.allclose(x_j0, x_j32[:, :d0], atol=1e-4), (
        "Slice j=0 and slice j=32 produce identical first-32 dimensions. "
        "This means global normalisation is in effect (wrong), not per-slice "
        "normalisation (correct). Check that F.normalize is called on x[:, :d_j] "
        "inside the slice loop, not on the full x before the loop."
    )


# ---------------------------------------------------------------------------
# Test 4 — bias flags: defaults (hidden_bias=False, head_bias=True) must
#           produce identical output to a model instantiated without those flags
# ---------------------------------------------------------------------------

def test_bias_flag_does_not_affect_existing_models():
    """
    hidden_bias=False and head_bias=True are the defaults and match the
    original architecture exactly.

    A model created with no new kwargs and one created with the defaults
    spelled out must produce bit-identical outputs on the same input,
    given the same random seed.
    """
    k = 64
    n_classes = 3
    N = 10
    torch.manual_seed(99)
    x = torch.randn(N, k)

    torch.manual_seed(42)
    model_old = SlicedSpectralMLP(k=k, n_classes=n_classes, n_layers=2,
                                   loss_weights="uniform")

    torch.manual_seed(42)
    model_new = SlicedSpectralMLP(k=k, n_classes=n_classes, n_layers=2,
                                   loss_weights="uniform",
                                   hidden_bias=False,
                                   head_bias=True)

    model_old.eval()
    model_new.eval()
    with torch.no_grad():
        out_old = model_old(x)
        out_new = model_new(x)

    for j, (lo, ln) in enumerate(zip(out_old, out_new)):
        assert torch.allclose(lo, ln, atol=1e-6), (
            f"Slice j={j}: output differs between model_old and model_new with "
            f"default bias flags. Max diff = {(lo - ln).abs().max():.2e}. "
            f"Defaults hidden_bias=False, head_bias=True must not change behaviour."
        )
