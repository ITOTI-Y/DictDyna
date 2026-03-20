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


def compute_obs_stats(
    transitions_dir: str | Path, state_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute observation mean/std from raw transitions.

    Returns:
        (obs_mean, obs_std) each shape (d,).
    """
    transitions_dir = Path(transitions_dir)
    all_states = []
    for f in sorted(transitions_dir.glob("*_transitions.npz")):
        t = np.load(f)
        if "states" in t and t["states"].shape[1] == state_dim:
            all_states.append(t["states"])

    if not all_states:
        logger.warning("No transitions found, using zero mean / unit std")
        return np.zeros(state_dim), np.ones(state_dim)

    states_all = np.concatenate(all_states, axis=0)
    obs_mean = states_all.mean(axis=0)
    obs_std = np.maximum(states_all.std(axis=0), 1e-8)
    logger.info(
        f"Computed obs stats from {len(all_states)} buildings, "
        f"{len(states_all)} samples"
    )
    return obs_mean, obs_std


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

    Dictionary is trained in obs-normalized diff space: Δs_raw / obs_std.
    This ensures D*alpha produces values in the same space as normalized obs,
    so s_norm + D*alpha is mathematically correct.

    Args:
        data_dir: Directory with state diff .npy files.
        n_atoms: Number of dictionary atoms K.
        method: "ksvd" or "online".
        n_nonzero: Max nonzero coefficients (for K-SVD).
        max_iter: Training iterations.
        output_path: Where to save the dictionary.
        transitions_dir: Directory with raw transitions (for obs stats).

    Returns:
        Dictionary tensor, shape (d, K).
    """
    logger.info(f"Loading state diffs from {data_dir}")
    raw_diffs = load_state_diffs(data_dir)
    state_dim = raw_diffs.shape[1]
    logger.info(f"Total data: {raw_diffs.shape}")

    # Compute obs stats from raw transitions
    obs_mean, obs_std = compute_obs_stats(transitions_dir, state_dim)

    # Normalize diffs into obs-normalized space: Δs_norm = Δs_raw / obs_std
    # This way D*alpha lives in the same space as (s - obs_mean) / obs_std
    diffs_norm = raw_diffs / obs_std
    logger.info(
        f"Normalized diffs (Δs/obs_std): "
        f"mean~{diffs_norm.mean():.6f}, std~{diffs_norm.std():.4f}"
    )

    if method == "ksvd":
        learner = KSVDDictionary(
            n_atoms=n_atoms, n_nonzero=n_nonzero, max_iter=max_iter
        )
        learner.fit(diffs_norm)
        dict_tensor = learner.to_torch()
    elif method == "online":
        learner = OnlineDictionaryLearner(n_atoms=n_atoms, n_iter=max_iter)
        learner.fit(diffs_norm)
        dict_tensor = learner.to_torch()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dictionary": dict_tensor,
            "obs_mean": torch.tensor(obs_mean, dtype=torch.float32),
            "obs_std": torch.tensor(obs_std, dtype=torch.float32),
            "n_atoms": n_atoms,
            "method": method,
        },
        output_path,
    )
    logger.info(f"Saved dictionary to {output_path}")

    return dict_tensor
