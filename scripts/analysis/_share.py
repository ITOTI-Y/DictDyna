"""Shared utilities for zero-shot mechanism analysis scripts."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src.xxx` works
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import ultraplot as up

from src.agent._share import STABILITY_EPS, normalize_obs
from src.obs_config import OBS_CONFIG  # noqa: F401

# === Paths ===
PROJECT_ROOT = _PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
OFFLINE_DIR = DATA_DIR / "offline_rollouts"
DIFFS_DIR = DATA_DIR / "processed" / "state_diffs"
PRETRAINED_DIR = PROJECT_ROOT / "output" / "pretrained"
OUTPUT_DIR = PROJECT_ROOT / "output" / "analysis" / "zero_shot_why"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# === Constants ===
BUILDINGS = ["office_hot", "office_mixed", "office_cool"]
BUILDING_LABELS = {"office_hot": "Hot", "office_mixed": "Mixed", "office_cool": "Cool"}
BUILDING_COLORS = {"office_hot": "#d62728", "office_mixed": "#ff7f0e", "office_cool": "#1f77b4"}
SOURCE_BUILDINGS = ["office_hot", "office_mixed"]
TARGET_BUILDING = "office_cool"

DEFAULT_DICT_PATH = PRETRAINED_DIR / "dict_k128_source_only.pt"

OBS_DIM_NAMES = [
    "Month", "Day", "Hour", "Outdoor T", "Outdoor RH",
    "Wind Spd", "Wind Dir", "Diffuse Sol", "Direct Sol",
    "Indoor T", "Indoor RH", "Occupant", "Heat SP", "Cool SP",
    "CO2", "HVAC Pwr", "HVAC Energy",
]


# === Data Loading ===
def load_building_transitions(bid: str) -> dict[str, np.ndarray]:
    path = OFFLINE_DIR / f"{bid}_transitions.npz"
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_state_diffs(bid: str) -> np.ndarray:
    return np.load(DIFFS_DIR / f"{bid}_state_diffs.npy")


def load_pretrained_dict(
    path: Path | str = DEFAULT_DICT_PATH,
) -> dict[str, np.ndarray | torch.Tensor]:
    data = torch.load(path, weights_only=False)
    return {
        "dictionary": data["dictionary"],  # (d, K) tensor
        "obs_mean": data["obs_mean"].numpy(),
        "obs_std": data["obs_std"].numpy(),
    }


def normalize_states(
    states: np.ndarray, obs_mean: np.ndarray, obs_std: np.ndarray
) -> np.ndarray:
    return np.stack([normalize_obs(s, obs_mean, obs_std) for s in states])


def normalize_states_fast(
    states: np.ndarray, obs_mean: np.ndarray, obs_std: np.ndarray
) -> np.ndarray:
    """Vectorized normalization (no per-row loop)."""
    safe_std = np.maximum(obs_std, STABILITY_EPS)
    return np.clip((states - obs_mean) / safe_std, -10.0, 10.0).astype(np.float32)


# === Sparse Coding ===
def compute_omp_codes(
    diffs_norm: np.ndarray,
    dictionary: np.ndarray,
    n_nonzero: int = 10,
) -> np.ndarray:
    """Compute OMP sparse codes for normalized diffs against dictionary.

    Args:
        diffs_norm: Normalized state diffs, shape (N, d).
        dictionary: Dictionary atoms, shape (d, K).
        n_nonzero: Number of nonzero coefficients per sample.

    Returns:
        Sparse codes, shape (N, K).
    """
    from sklearn.linear_model import OrthogonalMatchingPursuit

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero, fit_intercept=False)
    omp.fit(dictionary, np.eye(dictionary.shape[0]))  # dummy fit for API

    K = dictionary.shape[1]
    N = len(diffs_norm)
    codes = np.zeros((N, K), dtype=np.float32)

    # OMP per sample (sklearn doesn't batch well for this)
    for i in range(N):
        omp_single = OrthogonalMatchingPursuit(
            n_nonzero_coefs=n_nonzero, fit_intercept=False
        )
        omp_single.fit(dictionary, diffs_norm[i])
        codes[i] = omp_single.coef_

    return codes


# === Plotting ===
def setup_figure():
    """Configure ultraplot for paper-quality figures."""
    up.rc.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
