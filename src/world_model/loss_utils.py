"""Shared loss utilities for world model training.

Provides per-dimension adaptive weighting and identity guard penalty
to prevent the world model from performing worse than identity mapping.
"""

import torch


def compute_dim_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    state: torch.Tensor,
    dim_ema: torch.Tensor,
    ema_decay: float = 0.99,
    identity_penalty_lambda: float = 0.5,
    sample_weights: torch.Tensor | None = None,
    training: bool = True,
    reward_dim_indices: list[int] | None = None,
    reward_dim_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute per-dimension weighted MSE with identity guard.

    Harder dimensions (higher EMA MSE) receive proportionally more gradient.
    Identity guard penalizes any dimension where model prediction is worse
    than simply predicting no change (s' = s).

    Args:
        pred: Predicted next states, shape (batch, d).
        target: True next states, shape (batch, d).
        state: Current states for identity guard, shape (batch, d).
        dim_ema: Per-dimension MSE EMA buffer, shape (d,). Updated in-place.
        ema_decay: EMA decay factor for dimension weights.
        identity_penalty_lambda: Weight for identity guard penalty.
        sample_weights: Per-sample importance weights, shape (batch,).
        training: Whether in training mode (updates EMA in-place).
        reward_dim_indices: Indices of reward-relevant dims for explicit boost.
        reward_dim_weight: Extra multiplier for reward-relevant dims (>1 = boost).

    Returns:
        (weighted_mse_loss, extra_metrics_dict)
    """
    per_dim_sq_err = (target - pred) ** 2  # (batch, d)

    # Update EMA and compute adaptive dimension weights
    with torch.no_grad():
        batch_dim_mse = per_dim_sq_err.mean(dim=0)  # (d,)
        if training:
            dim_ema.lerp_(batch_dim_mse, 1.0 - ema_decay)
        # Weight: harder dims get more gradient, normalized to mean=1
        dim_weights = dim_ema / (dim_ema.mean() + 1e-8)

        # Explicit boost for reward-relevant dimensions
        if reward_dim_indices and reward_dim_weight > 1.0:
            for idx in reward_dim_indices:
                if 0 <= idx < dim_weights.shape[0]:
                    dim_weights[idx] = dim_weights[idx] * reward_dim_weight
            # Re-normalize to keep overall scale stable
            dim_weights = dim_weights / (dim_weights.mean() + 1e-8)

    # Apply per-dimension weights
    weighted_sq_err = per_dim_sq_err * dim_weights  # (batch, d)
    per_sample_mse = weighted_sq_err.mean(dim=-1)  # (batch,)

    if sample_weights is not None:
        mse_loss = (per_sample_mse * sample_weights).mean()
    else:
        mse_loss = per_sample_mse.mean()

    # Identity guard: penalize dimensions significantly worse than identity
    # Uses relative threshold (1.5x) + absolute floor (1e-4) to avoid
    # penalizing near-constant dimensions where identity MSE ≈ 0
    identity_penalty = torch.tensor(0.0, device=pred.device)
    if identity_penalty_lambda > 0:
        identity_sq_err = (target - state) ** 2  # (batch, d)
        threshold = torch.maximum(
            1.5 * identity_sq_err,
            torch.tensor(1e-4, device=pred.device),
        )
        excess = torch.relu(per_dim_sq_err - threshold)  # (batch, d)
        identity_penalty = excess.mean()
        mse_loss = mse_loss + identity_penalty_lambda * identity_penalty

    extra_metrics = {
        "identity_penalty": identity_penalty.item(),
        "dim_weight_max": dim_weights.max().item(),
        "dim_weight_min": dim_weights.min().item(),
    }

    return mse_loss, extra_metrics
