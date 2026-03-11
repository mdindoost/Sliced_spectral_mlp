"""
SlicedSpectralMLP: A shared-weight MLP on nested prefix slices of the
Laplacian eigenvector matrix.

Architecture summary for k eigenvectors and n_layers layers:
  - n_slices = k//2 + 1  (j = 0, 1, ..., k//2)
  - Slice j uses d_j = k//2 + j eigenvectors as input
  - All slices share weight matrices W[l] of shape (k, k), one per layer
  - Slice j at layer l uses the top-left d_j × d_j submatrix of W[l]
  - After each layer: ReLU → L2 sphere normalisation (no BatchNorm, no dropout)
  - Separate nn.Linear output head per slice

The top-left corner of W[0].grad accumulates signal from ALL slices; the
bottom-right corner only from the largest slice — this produces the bright
top-left pattern visible in the gradient heatmaps.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SlicedSpectralMLP(nn.Module):
    def __init__(
        self,
        k: int,
        n_classes: int,
        n_layers: int = 2,
        loss_weights: str = "uniform",
        eigenvalues: Optional[np.ndarray] = None,
        custom_slice_dims: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            k:                Number of eigenvectors (full input width).
            n_classes:        Number of output classes.
            n_layers:         Number of shared hidden layers.
            loss_weights:     One of {'uniform', 'coarse', 'eigenvalue'}.
            eigenvalues:      Array of k eigenvalues (required when
                              loss_weights='eigenvalue').
            custom_slice_dims: If provided, overrides the default sequential
                              slice dims [k//2, k//2+1, ..., k]. All values
                              must be in [1, k].
        """
        super().__init__()
        self.k = k
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.half = k // 2

        if custom_slice_dims is not None:
            self.slice_dims: List[int] = custom_slice_dims
            self.n_slices = len(custom_slice_dims)
        else:
            # Slice j ∈ {0, …, k//2}: input width d_j = k//2 + j
            self.n_slices = self.half + 1
            self.slice_dims = [self.half + j for j in range(self.n_slices)]

        # Shared weight matrices — one (k × k) matrix per layer.
        # Slice j uses the top-left (d_j × d_j) submatrix; gradient flows
        # through naturally because slicing creates a view.
        self.W = nn.ParameterList(
            [nn.Parameter(torch.empty(k, k)) for _ in range(n_layers)]
        )
        for param in self.W:
            nn.init.orthogonal_(param)

        # Separate output head per slice — NOT shared.
        self.heads = nn.ModuleList(
            [nn.Linear(d_j, n_classes) for d_j in self.slice_dims]
        )

        # Pre-compute and register loss weights as a non-trainable buffer.
        self.register_buffer(
            "loss_w", self._compute_weights(loss_weights, eigenvalues)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_weights(
        self, strategy: str, eigenvalues: Optional[np.ndarray]
    ) -> torch.Tensor:
        n = self.n_slices
        if strategy == "uniform":
            w = torch.ones(n)

        elif strategy == "coarse":
            # Linearly decreasing: coarser slices (smaller j, smaller d_j)
            # receive more weight than finer slices.
            w = torch.arange(float(n), 0.0, -1.0)

        elif strategy == "eigenvalue":
            if eigenvalues is None:
                raise ValueError(
                    "eigenvalues must be supplied when loss_weights='eigenvalue'"
                )
            eig = torch.as_tensor(eigenvalues, dtype=torch.float32)
            # w_j ∝ 1 / λ_{half+j}  (spec formula; index clamped to valid range)
            indices = [min(self.half + j, len(eig) - 1) for j in range(n)]
            lam = eig[indices].clamp(min=1e-6)
            w = 1.0 / lam

        else:
            raise ValueError(
                f"Unknown loss_weights='{strategy}'. "
                "Choose from: 'uniform', 'coarse', 'eigenvalue'."
            )

        return w / w.sum()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (N, k) — full eigenvector matrix for all N nodes.
        Returns:
            List of n_slices tensors each of shape (N, n_classes).
        """
        all_logits: List[torch.Tensor] = []

        for j, d_j in enumerate(self.slice_dims):
            h = x[:, :d_j]  # (N, d_j) — prefix slice

            for l in range(self.n_layers):
                # Top-left (d_j × d_j) submatrix of the shared (k × k) weight.
                W_j = self.W[l][:d_j, :d_j]  # gradient flows here
                h = h @ W_j                   # (N, d_j)
                h = F.relu(h)
                h = F.normalize(h, p=2, dim=-1)  # L2 sphere normalisation

            all_logits.append(self.heads[j](h))

        return all_logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        all_logits: List[torch.Tensor],
        labels: torch.Tensor,
        mask: torch.Tensor,
        loss_cutoff: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Weighted sum of per-slice cross-entropy losses.

        Args:
            all_logits:   output of forward() — list of (N, n_classes) tensors.
            labels:       (N,) integer class labels.
            mask:         (N,) boolean mask selecting nodes for this split.
            loss_cutoff:  If set, slices j > loss_cutoff contribute zero loss.
                          Active slices are re-weighted uniformly to 1/n_active.
        Returns:
            Scalar loss tensor.
        """
        loss = torch.zeros(1, device=self.loss_w.device, dtype=torch.float32)[0]
        if loss_cutoff is not None:
            n_active = loss_cutoff + 1
            for j, logits in enumerate(all_logits):
                if j > loss_cutoff:
                    continue
                ce = F.cross_entropy(logits[mask], labels[mask])
                loss = loss + ce / n_active
        else:
            for j, logits in enumerate(all_logits):
                ce = F.cross_entropy(logits[mask], labels[mask])
                loss = loss + self.loss_w[j] * ce
        return loss
