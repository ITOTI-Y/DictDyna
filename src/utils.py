"""Shared utilities for DictDyna."""

import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Resolve device string to torch.device."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def build_dim_weights(
    state_dim: int,
    reward_dims: list[int],
    reward_dim_weight: float,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """Build per-dimension weight vector for WM loss.

    Returns None if reward_dim_weight <= 1.0 (uniform, backward compatible).
    """
    if reward_dim_weight <= 1.0:
        return None
    weights = torch.ones(state_dim, device=device)
    for idx in reward_dims:
        if 0 <= idx < state_dim:
            weights[idx] = reward_dim_weight
    return weights


@contextmanager
def sinergym_workdir(path: str = "output/tmp"):
    """Temporarily change CWD so Sinergym writes EnergyPlus temp files there."""
    workdir = Path(path)
    workdir.mkdir(parents=True, exist_ok=True)
    prev = os.getcwd()
    os.chdir(workdir)
    try:
        yield
    finally:
        os.chdir(prev)
