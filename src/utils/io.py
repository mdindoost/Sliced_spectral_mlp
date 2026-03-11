"""
I/O utilities for saving/loading results, tables, and checkpoints.

Functions
---------
  save_table(lines, path)         — write a list of strings to a text file
  save_checkpoint(model, path)    — save model state_dict
  load_checkpoint(model, path)    — load model state_dict
  save_results(results, path)     — save results dict as JSON
  load_results(path)              — load results dict from JSON
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import torch


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
