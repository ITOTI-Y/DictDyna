"""Dictionary pretraining pipeline."""

from pathlib import Path

import numpy as np
import torch
from loguru import logger

from src.dictionary.ksvd import KSVDDictionary
from src.dictionary.online_dl import OnlineDictionaryLearner


def load_state_diffs(
    data_dir: str | Path, buildings: list[str] | None = None
) -> np.ndarray:
    """Load and concatenate state diff files from a directory.

    Args:
        data_dir: Directory containing *_state_diffs.npy files.
        buildings: If provided, only load files matching these building IDs
                   (e.g. ["office_hot", "office_mixed"]).
    """
    data_dir = Path(data_dir)
    all_diffs = []
    for f in sorted(data_dir.glob("*_state_diffs.npy")):
        if buildings and not any(bid in f.stem for bid in buildings):
            logger.info(f"Skipping {f.name} (not in source buildings)")
            continue
        diffs = np.load(f)
        logger.info(f"Loaded {f.name}: {diffs.shape}")
        all_diffs.append(diffs)
    if not all_diffs:
        raise FileNotFoundError(f"No *_state_diffs.npy files in {data_dir}")
    return np.concatenate(all_diffs, axis=0)


def compute_obs_stats(
    transitions_dir: str | Path,
    state_dim: int,
    buildings: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute observation mean/std from raw transitions."""
    transitions_dir = Path(transitions_dir)
    all_states = []
    for f in sorted(transitions_dir.glob("*_transitions.npz")):
        if buildings and not any(bid in f.stem for bid in buildings):
            logger.info(f"Skipping {f.name} (not in source buildings)")
            continue
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
    buildings: list[str] | None = None,
) -> torch.Tensor:
    """Run the full pretraining pipeline.

    Dictionary is trained on z-score normalized diffs: (Δs - diff_mean) / diff_std.
    Obs stats are also saved for policy-side normalization.
    The world model handles space conversion internally.
    """
    if buildings:
        logger.info(f"Source-only pretraining: buildings={buildings}")
    logger.info(f"Loading state diffs from {data_dir}")
    raw_diffs = load_state_diffs(data_dir, buildings=buildings)
    state_dim = raw_diffs.shape[1]
    logger.info(f"Total data: {raw_diffs.shape}")

    # Compute obs stats from raw transitions
    obs_mean, obs_std = compute_obs_stats(
        transitions_dir, state_dim, buildings=buildings
    )

    # Normalize diffs into obs-normalized space: Δs / obs_std
    # This way D*alpha lives in the same space as (s - obs_mean) / obs_std
    # and s_norm + D*alpha = s'_norm without any space conversion
    diffs_norm = raw_diffs / obs_std
    logger.info(
        f"Normalized diffs (raw/obs_std): mean~{diffs_norm.mean():.6f}, "
        f"std~{diffs_norm.std():.4f}"
    )

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

    # Save dictionary and obs stats (no space conversion needed)
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
