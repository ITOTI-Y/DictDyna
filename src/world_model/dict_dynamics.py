"""Dictionary dynamics model with space conversion.

When policy operates in obs-normalized space and dictionary in diff-normalized space:
  D*alpha is in diff-norm space -> convert to obs-norm: delta_obs = D*alpha * diff_std / obs_std
  s'_norm = s_norm + delta_obs + diff_mean / obs_std
"""

import torch
import torch.nn as nn

from src.world_model._share import (
    apply_space_conversion,
    build_residual_head,
)
from src.world_model._share import (
    normalize_atoms as _normalize_atoms,
)
from src.world_model.loss_utils import compute_dim_weighted_mse
from src.world_model.sparse_encoder import SparseEncoder


class DictDynamicsModel(nn.Module):
    """Dictionary-based transition dynamics model with space conversion.

    Supports two modes:
    1. Raw mode (no conversion): s' = s + D @ alpha + residual(h)
    2. Normalized mode: policy in obs-norm, dict in diff-norm, with conversion

    Args:
        dictionary: Initial dictionary tensor, shape (d, K).
        sparse_encoder: SparseEncoder instance.
        learnable_dict: Whether D is a learnable parameter.
        diff_mean: Mean of state diffs (for space conversion).
        diff_std: Std of state diffs (for space conversion).
        obs_std: Std of observations (for space conversion).
        dim_weights: Static per-dimension loss weights.
        residual_hidden_dim: Residual correction head hidden dim (0=disabled).
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
        residual_hidden_dim: int = 0,
        controllable_dims: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = sparse_encoder
        self.controllable_dims = controllable_dims

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
            scale = diff_std / obs_std
            bias = (
                diff_mean / obs_std
                if diff_mean is not None
                else torch.zeros_like(scale)
            )
        else:
            scale = None
            bias = None
        self.register_buffer("_scale", scale)
        self.register_buffer("_bias", bias)

        # Dimension-weighted loss
        if dim_weights is not None:
            self.register_buffer("dim_weights", dim_weights)
        else:
            self.dim_weights: torch.Tensor | None = None

        # Nonlinear residual correction head (zero-initialized output layer)
        if residual_hidden_dim > 0:
            self.residual_head: nn.Module | None = build_residual_head(
                sparse_encoder.shared_output_dim,
                residual_hidden_dim,
                dictionary.shape[0],
            )
        else:
            self.residual_head = None
        self._last_residual: torch.Tensor | None = None

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

        Args:
            state: Current states (obs-normalized if conversion enabled).
            action: Actions (always in raw action space).
            building_id: Building identifier.

        Returns:
            (predicted_next_state, alpha)
        """
        # Direct trunk access (same pattern as ProbabilisticDictDynamics)
        x = torch.cat([state, action], dim=-1)
        h = self.encoder.shared_trunk(x)
        alpha = self.encoder.adapters[building_id](h)
        if self.encoder.sparsity_method == "topk":
            alpha = self.encoder._topk_sparsify(alpha)

        delta_raw = alpha @ self.dictionary.T

        # Nonlinear residual correction
        if self.residual_head is not None:
            residual = self.residual_head(h)
            delta_raw = delta_raw + residual
            self._last_residual = residual
        else:
            self._last_residual = None

        # Space conversion: diff-norm → obs-norm + controllable-only embedding
        delta = apply_space_conversion(
            delta_raw,
            state,
            self.controllable_dims,
            self._scale,
            self._bias,
            self._has_conversion,
        )

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
        _normalize_atoms(self.dictionary)

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
        residual_lambda: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute world model loss: weighted MSE + lambda * L1 + residual L2.

        Args:
            sample_weights: Per-sample importance weights, shape (batch,).
            identity_penalty_lambda: Penalty for being worse than identity.
            dim_weight_ema_decay: EMA decay for per-dimension weights.
            use_dim_weighting: Enable per-dimension adaptive weighting.
            residual_lambda: L2 regularization weight on residual output.
        """
        pred_next, alpha = self.forward(state, action, building_id)

        # In controllable-only mode, compute loss only on predicted dims
        if self.controllable_dims is not None:
            ctrl = list(self.controllable_dims)
            loss_pred = pred_next[:, ctrl]
            loss_target = next_state[:, ctrl]
            loss_state = state[:, ctrl]
        else:
            loss_pred = pred_next
            loss_target = next_state
            loss_state = state

        if use_dim_weighting or identity_penalty_lambda > 0:
            mse_loss, extra = compute_dim_weighted_mse(
                pred=loss_pred,
                target=loss_target,
                state=loss_state,
                dim_ema=self._dim_ema,  # ty: ignore[invalid-argument-type]
                ema_decay=dim_weight_ema_decay,
                identity_penalty_lambda=identity_penalty_lambda,
                sample_weights=sample_weights,
                training=self.training,
            )
        else:
            per_dim_sq_err = (loss_target - loss_pred) ** 2
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

        # Residual L2 regularization
        if self._last_residual is not None and residual_lambda > 0:
            total_loss = total_loss + residual_lambda * (self._last_residual**2).mean()

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
        if self._last_residual is not None:
            metrics["residual_norm"] = self._last_residual.norm(dim=-1).mean().item()

        return total_loss, metrics
