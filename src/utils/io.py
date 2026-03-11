"""
I/O utilities for saving/loading results, tables, and checkpoints.

Functions
---------
  validate_config(cfg, check_data) — validate a resolved config dict before training
  save_table(lines, path)          — write a list of strings to a text file
  save_checkpoint(model, path)     — save model state_dict
  load_checkpoint(model, path)     — load model state_dict
  save_results(results, path)      — save results dict as JSON
  load_results(path)               — load results dict from JSON
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import torch


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = ("dataset", "k", "n_layers", "lr", "weight_decay", "epochs", "seed")


def validate_config(cfg: dict, check_data: bool = True) -> None:
    """
    Validate a fully resolved config dict before training starts.

    Args:
        cfg:        Resolved config dict (defaults merged, CLI overrides applied).
        check_data: If True, load the dataset to verify n_classes and k <= N_LCC.
                    Set False for fast structural-only checks (e.g. --dry-run).

    Raises:
        KeyError   — a required key is missing from cfg.
        ValueError — n_classes mismatch, k >= N_LCC, or lr/weight_decay are strings.
    """
    # 1. Required keys present
    for key in _REQUIRED_KEYS:
        if key not in cfg:
            raise KeyError(
                f"Config missing required key: '{key}'. "
                f"Add it to your YAML or experiments/configs/default.yaml."
            )

    # 2. lr and weight_decay must be numeric (YAML silently reads '0.01' as str)
    for key in ("lr", "weight_decay"):
        val = cfg[key]
        if not isinstance(val, (int, float)):
            raise ValueError(
                f"Config '{key}' must be a float, got {type(val).__name__}: {val!r}\n"
                f"  Fix: in YAML write   {key}: 0.01   not   {key}: \"0.01\""
            )

    if not check_data:
        return

    # 3. Dataset-level checks — load with a tiny k so eigsh is instant
    from src.data.loaders import load_dataset  # lazy import to avoid circular dep

    dataset = cfg["dataset"]
    k = int(cfg["k"])

    print(f"  [validate_config] loading '{dataset}' (k=8) to verify n_classes and k …")
    U, labels, *_ = load_dataset(dataset, k=8)

    N_lcc = U.shape[0]
    actual_classes = int(labels.max().item()) + 1

    # k must be strictly less than N_lcc (eigsh requirement)
    if k >= N_lcc:
        raise ValueError(
            f"k={k} >= N_LCC={N_lcc} for '{dataset}'. "
            f"eigsh requires k < N. Set k <= {N_lcc - 2} in your config."
        )

    # n_classes must match actual data (if set explicitly)
    cfg_n_classes = cfg.get("n_classes")
    if cfg_n_classes is not None and int(cfg_n_classes) != actual_classes:
        raise ValueError(
            f"Config n_classes={cfg_n_classes} does not match "
            f"actual dataset n_classes={actual_classes} for '{dataset}'.\n"
            f"  Fix: set   n_classes: {actual_classes}   in your YAML."
        )


def save_table(lines: List[str], path: str) -> None:
    """Write table lines to a text file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    """Save model state_dict to path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Load model state_dict from path."""
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def save_results(results: Dict, path: str) -> None:
    """Save a results dict to JSON (converts non-serializable types)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _convert(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_convert(results), f, indent=2)


def load_results(path: str) -> Dict:
    """Load a results dict from JSON."""
    with open(path) as f:
        return json.load(f)
