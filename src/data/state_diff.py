"""State difference computation and preprocessing."""

from pathlib import Path

import numpy as np
from loguru import logger


def compute_state_diffs(
    trajectories: list[np.ndarray],
) -> np.ndarray:
    """Compute state differences from trajectory data.

    Args:
        trajectories: List of trajectory arrays, each shape (T, d).

    Returns:
        Concatenated state diffs, shape (N_total, d).
    """
    all_diffs = []
    for traj in trajectories:
        diffs = traj[1:] - traj[:-1]  # (T-1, d)
        all_diffs.append(diffs)
    return np.concatenate(all_diffs, axis=0)


def filter_outliers(data: np.ndarray, n_std: float = 5.0) -> np.ndarray:
    """Remove outlier rows beyond n_std standard deviations.

    Args:
        data: Input data, shape (N, d).
        n_std: Number of standard deviations for outlier threshold.

    Returns:
        Filtered data.
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std = np.maximum(std, 1e-8)
    mask = np.all(np.abs(data - mean) < n_std * std, axis=1)
    n_removed = len(data) - mask.sum()
    if n_removed > 0:
        logger.info(f"Removed {n_removed}/{len(data)} outlier samples")
    return data[mask]


def save_state_diffs(
    diffs: np.ndarray,
    building_id: str,
    output_dir: str | Path = "data/processed/state_diffs",
) -> Path:
    """Save state diffs to .npy file.

    Args:
        diffs: State diffs, shape (N, d).
        building_id: Building identifier for filename.
        output_dir: Output directory.

    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{building_id}_state_diffs.npy"
    np.save(path, diffs)
    logger.info(f"Saved {len(diffs)} state diffs to {path}")
    return path


def load_state_diffs(path: str | Path) -> np.ndarray:
    """Load state diffs from .npy file."""
    return np.load(path)
