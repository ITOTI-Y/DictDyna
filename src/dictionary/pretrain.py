"""Dictionary pretraining pipeline."""

from pathlib import Path

import numpy as np
import torch
from loguru import logger

from src.dictionary.ksvd import KSVDDictionary
from src.dictionary.online_dl import OnlineDictionaryLearner


def load_state_diffs(data_dir: str | Path) -> np.ndarray:
    """Load and concatenate state diff files from a directory."""
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
    """Compute observation mean/std from raw transitions."""
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
        f"Computed obs stats from {len(all_states)} buildings, {len(states_all)} samples"
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

    Dictionary is trained on z-score normalized diffs: (Δs - diff_mean) / diff_std.
    Obs stats are also saved for policy-side normalization.
    The world model handles space conversion internally.
    """
    logger.info(f"Loading state diffs from {data_dir}")
    raw_diffs = load_state_diffs(data_dir)
    state_dim = raw_diffs.shape[1]
    logger.info(f"Total data: {raw_diffs.shape}")

    # Z-score normalize diffs
    diff_mean = raw_diffs.mean(axis=0)
    diff_std = np.maximum(raw_diffs.std(axis=0), 1e-8)
    diffs_norm = (raw_diffs - diff_mean) / diff_std
    logger.info(
        f"Normalized diffs: mean~{diff_mean.mean():.6f}, std~{diff_std.mean():.4f}"
    )

    # Compute obs stats from raw transitions
    obs_mean, obs_std = compute_obs_stats(transitions_dir, state_dim)

    # Train dictionary
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

    # Save all stats needed for space conversion
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dictionary": dict_tensor,
            "diff_mean": torch.tensor(diff_mean, dtype=torch.float32),
            "diff_std": torch.tensor(diff_std, dtype=torch.float32),
            "obs_mean": torch.tensor(obs_mean, dtype=torch.float32),
            "obs_std": torch.tensor(obs_std, dtype=torch.float32),
            "n_atoms": n_atoms,
            "method": method,
        },
        output_path,
    )
    logger.info(f"Saved dictionary to {output_path}")
    return dict_tensor
