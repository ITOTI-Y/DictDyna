"""Dictionary pretraining pipeline."""

from pathlib import Path

import numpy as np
import torch
from loguru import logger

from src.dictionary.ksvd import KSVDDictionary
from src.dictionary.online_dl import OnlineDictionaryLearner


def load_state_diffs(data_dir: str | Path) -> np.ndarray:
    """Load and concatenate state diff files from a directory.

    Args:
        data_dir: Directory containing *_state_diffs.npy files.

    Returns:
        Concatenated state diffs, shape (N_total, d).
    """
    data_dir = Path(data_dir)
    all_diffs = []
    for f in sorted(data_dir.glob("*_state_diffs.npy")):
        diffs = np.load(f)
        logger.info(f"Loaded {f.name}: {diffs.shape}")
        all_diffs.append(diffs)

    if not all_diffs:
        raise FileNotFoundError(f"No *_state_diffs.npy files in {data_dir}")

    return np.concatenate(all_diffs, axis=0)


def normalize_data(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize data.

    Returns:
        (normalized_data, mean, std)
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std = np.maximum(std, 1e-8)
    return (data - mean) / std, mean, std


def pretrain_dictionary(
    data_dir: str | Path,
    n_atoms: int = 128,
    method: str = "ksvd",
    n_nonzero: int = 10,
    max_iter: int = 50,
    output_path: str | Path = "output/pretrained/dict.pt",
    transitions_dir: str | Path = "data/offline_rollouts",
) -> torch.Tensor:
    """Run the full pretraining pipeline.

    Args:
        data_dir: Directory with state diff .npy files.
        n_atoms: Number of dictionary atoms K.
        method: "ksvd" or "online".
        n_nonzero: Max nonzero coefficients (for K-SVD).
        max_iter: Training iterations.
        output_path: Where to save the dictionary.

    Returns:
        Dictionary tensor, shape (d, K).
    """
    logger.info(f"Loading state diffs from {data_dir}")
    data = load_state_diffs(data_dir)
    logger.info(f"Total data: {data.shape}")

    data_norm, mean, std = normalize_data(data)
    logger.info(f"Normalized data: mean~{mean.mean():.4f}, std~{std.mean():.4f}")

    if method == "ksvd":
        learner = KSVDDictionary(
            n_atoms=n_atoms, n_nonzero=n_nonzero, max_iter=max_iter
        )
        learner.fit(data_norm)
        dict_tensor = learner.to_torch()
    elif method == "online":
        learner = OnlineDictionaryLearner(n_atoms=n_atoms, n_iter=max_iter)
        learner.fit(data_norm)
        dict_tensor = learner.to_torch()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute observation stats from raw transitions (for obs normalization)
    transitions_dir = Path(transitions_dir)
    obs_mean_t = torch.zeros(data.shape[1], dtype=torch.float32)
    obs_std_t = torch.ones(data.shape[1], dtype=torch.float32)
    if transitions_dir.exists():
        all_states = []
        for f in sorted(transitions_dir.glob("*_transitions.npz")):
            t = np.load(f)
            if "states" in t and t["states"].shape[1] == data.shape[1]:
                all_states.append(t["states"])
        if all_states:
            states_all = np.concatenate(all_states, axis=0)
            obs_mean_t = torch.tensor(states_all.mean(axis=0), dtype=torch.float32)
            obs_std_t = torch.tensor(
                np.maximum(states_all.std(axis=0), 1e-8), dtype=torch.float32
            )
            logger.info(
                f"Computed obs stats from {len(all_states)} buildings, {len(states_all)} samples"
            )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dictionary": dict_tensor,
            "mean": torch.tensor(mean, dtype=torch.float32),  # diff stats
            "std": torch.tensor(std, dtype=torch.float32),  # diff stats
            "obs_mean": obs_mean_t,  # observation stats
            "obs_std": obs_std_t,  # observation stats
            "n_atoms": n_atoms,
            "method": method,
        },
        output_path,
    )
    logger.info(f"Saved dictionary to {output_path}")

    return dict_tensor
