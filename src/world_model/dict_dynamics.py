"""Dictionary dynamics model with space conversion.

When policy operates in obs-normalized space and dictionary in diff-normalized space:
  D*alpha is in diff-norm space -> convert to obs-norm: delta_obs = D*alpha * diff_std / obs_std
  s'_norm = s_norm + delta_obs + diff_mean / obs_std
"""

import torch
import torch.nn as nn

from src.world_model.loss_utils import compute_dim_weighted_mse
from src.world_model.sparse_encoder import SparseEncoder


class DictDynamicsModel(nn.Module):
    """Dictionary-based transition dynamics model with space conversion.

    Supports two modes:
    1. Raw mode (no conversion): s' = s + D @ alpha
    2. Normalized mode: policy in obs-norm, dict in diff-norm, with conversion

    Args:
        dictionary: Initial dictionary tensor, shape (d, K).
        sparse_encoder: SparseEncoder instance.
        learnable_dict: Whether D is a learnable parameter.
        diff_mean: Mean of state diffs (for space conversion).
        diff_std: Std of state diffs (for space conversion).
        obs_std: Std of observations (for space conversion).
    """

    def __init__(
        self,
        dictionary: torch.Tensor,
        sparse_encoder: SparseEncoder,
        learnable_dict: bool = True,
        diff_mean: torch.Tensor | None = None,
        diff_std: torch.Tensor | None = None,
        obs_std: torch.Tensor | None = None,
        dim_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.encoder = sparse_encoder

        if learnable_dict:
            self.dictionary = nn.Parameter(dictionary.clone())
        else:
            self.register_buffer("dictionary", dictionary.clone())

        # Per-dimension MSE EMA for adaptive loss weighting
        self.register_buffer("_dim_ema", torch.ones(dictionary.shape[0]))

        # Space conversion buffers (None = raw mode, no conversion)
        self._has_conversion = diff_std is not None and obs_std is not None
        if self._has_conversion:
            assert diff_std is not None and obs_std is not None
            # scale converts diff-norm space to obs-norm space: diff_std / obs_std
            scale = diff_std / obs_std
            self.register_buffer("_scale", scale)
            # bias accounts for diff_mean in obs-norm space: diff_mean / obs_std
            bias = (
                diff_mean / obs_std
                if diff_mean is not None
                else torch.zeros_like(scale)
            )
            self.register_buffer("_bias", bias)

        # Dimension-weighted loss
        if dim_weights is not None:
            self.register_buffer("dim_weights", dim_weights)
        else:
            self.dim_weights: torch.Tensor | None = None

    @property
    def n_atoms(self) -> int:
        return self.dictionary.shape[1]

    @property
    def state_dim(self) -> int:
        return self.dictionary.shape[0]

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        building_id: str = "0",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next state.

        If space conversion is enabled:
          delta_diff_norm = D @ alpha          (in diff-norm space)
          delta_obs_norm = delta * scale + bias (convert to obs-norm space)
          s'_obs_norm = s_obs_norm + delta_obs_norm

        Args:
            state: Current states (obs-normalized if conversion enabled).
            action: Actions (always in raw action space).
            building_id: Building identifier.

        Returns:
            (predicted_next_state, alpha)
        """
        alpha = self.encoder(state, action, building_id)
        delta = alpha @ self.dictionary.T
        if self._has_conversion:
            # Convert from diff-norm to obs-norm space
            delta = delta * self._scale + self._bias
        next_state = state + delta
        return next_state, alpha

    def predict(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        building_id: str = "0",
    ) -> torch.Tensor:
        """Predict next state (without returning alpha)."""
        next_state, _ = self.forward(state, action, building_id)
        return next_state

    def normalize_atoms(self) -> None:
        """Normalize dictionary columns to unit L2 norm."""
        with torch.no_grad():
            norms = torch.norm(self.dictionary, dim=0, keepdim=True)
            norms = torch.clamp(norms, min=1e-10)
            self.dictionary.div_(norms)

    def compute_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        building_id: str = "0",
        sparsity_lambda: float = 0.1,
        sample_weights: torch.Tensor | None = None,
        identity_penalty_lambda: float = 0.0,
        dim_weight_ema_decay: float = 0.99,
        use_dim_weighting: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute world model loss: weighted MSE + lambda * L1.

        Args:
            sample_weights: Per-sample importance weights, shape (batch,).
            identity_penalty_lambda: Penalty for being worse than identity.
            dim_weight_ema_decay: EMA decay for per-dimension weights.
            use_dim_weighting: Enable per-dimension adaptive weighting.
        """
        pred_next, alpha = self.forward(state, action, building_id)

        if use_dim_weighting or identity_penalty_lambda > 0:
            mse_loss, extra = compute_dim_weighted_mse(
                pred=pred_next,
                target=next_state,
                state=state,
                dim_ema=self._dim_ema,  # ty: ignore[invalid-argument-type]
                ema_decay=dim_weight_ema_decay,
                identity_penalty_lambda=identity_penalty_lambda,
                sample_weights=sample_weights,
                training=self.training,
            )
        else:
            per_dim_sq_err = (next_state - pred_next) ** 2
            if self.dim_weights is not None:
                per_dim_sq_err = per_dim_sq_err * self.dim_weights
            per_sample_mse = per_dim_sq_err.mean(dim=-1)
            if sample_weights is not None:
                mse_loss = (per_sample_mse * sample_weights).mean()
            else:
                mse_loss = per_sample_mse.mean()
            extra = {}

        l1_loss = torch.mean(torch.abs(alpha))
        total_loss = mse_loss + sparsity_lambda * l1_loss

        sparsity = (alpha.abs() < 1e-6).float().mean().item()

        metrics: dict[str, float] = {
            "mse_loss": mse_loss.item(),
            "l1_loss": l1_loss.item(),
            "total_loss": total_loss.item(),
            "sparsity": sparsity,
        }
        metrics.update(extra)
        if self.dim_weights is not None:
            raw_sq_err = (next_state - pred_next) ** 2
            reward_mask = self.dim_weights > 1.0
            metrics["mse_reward_dims"] = raw_sq_err[:, reward_mask].mean().item()
            metrics["mse_other_dims"] = raw_sq_err[:, ~reward_mask].mean().item()

        return total_loss, metrics
