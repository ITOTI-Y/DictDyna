"""Shared utilities for agent modules."""

import numpy as np
import torch

# Observation clipping range for normalized observations
OBS_CLIP_RANGE = (-10.0, 10.0)

# Default number of random steps before training starts
DEFAULT_LEARNING_STARTS = 1000

# Numerical stability epsilon
STABILITY_EPS = 1e-8


def normalize_obs(
    raw_obs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Normalize observation: clip((s - mean) / std, clip_lo, clip_hi)."""
    lo, hi = OBS_CLIP_RANGE
    return np.clip((raw_obs - mean) / std, lo, hi).astype(np.float32)


def compute_td_error_weights(
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    critic_target: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    gamma: float,
) -> torch.Tensor:
    """Compute TD-error based sample weights for reward-weighted WM training.

    Returns weights in [1, 1+beta] range, shape (batch,).
    """
    with torch.no_grad():
        next_actions, _ = actor(batch["next_states"])
        q1_next, q2_next = critic_target(batch["next_states"], next_actions)
        q_next = torch.min(q1_next, q2_next)
        target_q = batch["rewards"] + (1.0 - batch["dones"]) * gamma * q_next
        q1, q2 = critic(batch["states"], batch["actions"])
        td_error = (torch.min(q1, q2) - target_q).abs().squeeze(-1)
        return 1.0 + td_error / (td_error.mean() + STABILITY_EPS)
