"""Context-conditioned dictionary dynamics model.

Replaces DictDynamicsModel's adapter-based routing with continuous context
conditioning. The context vector z is inferred by ContextEncoder and fed
to ContextConditionedEncoder alongside (s, a).

Core equation: s_hat' = s + D * alpha(s, a, z)
"""

import torch
import torch.nn as nn

from src.world_model.context_encoder import ContextConditionedEncoder, ContextEncoder
from src.world_model.loss_utils import compute_dim_weighted_mse


class ContextDynamicsModel(nn.Module):
    """Dictionary dynamics model conditioned on context vector.

    Args:
        dictionary: Pretrained dictionary tensor, shape (d, K).
        context_encoder: ContextEncoder that infers z from transitions.
        conditioned_encoder: ContextConditionedEncoder that produces alpha.
        learnable_dict: If True, dictionary is an nn.Parameter.
        diff_std: Diff-space std for space conversion (optional).
        obs_std: Obs-space std for space conversion (optional).
        diff_mean: Diff-space mean for space conversion (optional).
    """

    def __init__(
        self,
        dictionary: torch.Tensor,
        context_encoder: ContextEncoder,
        conditioned_encoder: ContextConditionedEncoder,
        learnable_dict: bool = True,
        diff_mean: torch.Tensor | None = None,
        diff_std: torch.Tensor | None = None,
        obs_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.context_encoder = context_encoder
        self.encoder = conditioned_encoder

        if learnable_dict:
            self.dictionary = nn.Parameter(dictionary.clone())
        else:
            self.register_buffer("dictionary", dictionary.clone())

        # Per-dimension MSE EMA for adaptive loss weighting
        self.register_buffer("_dim_ema", torch.ones(dictionary.shape[0]))

        # Space conversion buffers
        self._has_conversion = diff_std is not None and obs_std is not None
        if self._has_conversion:
            assert diff_std is not None and obs_std is not None
            scale = diff_std / obs_std
            self.register_buffer("_scale", scale)
            bias = (
                diff_mean / obs_std
                if diff_mean is not None
                else torch.zeros_like(scale)
            )
            self.register_buffer("_bias", bias)

    @property
    def n_atoms(self) -> int:
        return self.dictionary.shape[1]

    @property
    def state_dim(self) -> int:
        return self.dictionary.shape[0]

    def infer_context(self, transitions: torch.Tensor) -> torch.Tensor:
        """Infer context vector from transition window.

        Args:
            transitions: Shape (batch, K, 2*d + m).

        Returns:
            Context vector z, shape (batch, context_dim).
        """
        return self.context_encoder(transitions)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next state given pre-computed context.

        Args:
            state: Current states, shape (batch, d).
            action: Actions, shape (batch, m).
            context: Context vector z, shape (batch, context_dim).

        Returns:
            (predicted_next_state, alpha)
        """
        alpha = self.encoder(state, action, context)
        delta = alpha @ self.dictionary.T

        if self._has_conversion:
            delta = delta * self._scale + self._bias

        next_state = state + delta
        return next_state, alpha

    def predict(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next state (without returning alpha)."""
        next_state, _ = self.forward(state, action, context)
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
        context: torch.Tensor,
        sparsity_lambda: float = 0.1,
        sample_weights: torch.Tensor | None = None,
        identity_penalty_lambda: float = 0.0,
        dim_weight_ema_decay: float = 0.99,
        use_dim_weighting: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute prediction loss.

        Args:
            state: Current states, shape (batch, d).
            action: Actions, shape (batch, m).
            next_state: True next states, shape (batch, d).
            context: Context vector z, shape (batch, context_dim).
            sparsity_lambda: L1 sparsity weight.
            sample_weights: Per-sample loss weights (reward-weighted WM).
            identity_penalty_lambda: Penalty for being worse than identity.
            dim_weight_ema_decay: EMA decay for per-dimension weights.
            use_dim_weighting: Enable per-dimension adaptive weighting.

        Returns:
            (total_loss, metrics_dict)
        """
        pred_next, alpha = self.forward(state, action, context)

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
            per_sample_mse = ((next_state - pred_next) ** 2).mean(dim=-1)
            if sample_weights is not None:
                mse_loss = (per_sample_mse * sample_weights).mean()
            else:
                mse_loss = per_sample_mse.mean()
            extra = {}

        l1_loss = torch.mean(torch.abs(alpha))
        total_loss = mse_loss + sparsity_lambda * l1_loss

        sparsity = (alpha.abs() < 1e-6).float().mean().item()

        metrics = {
            "wm_loss": total_loss.item(),
            "wm_mse": mse_loss.item(),
            "wm_l1": l1_loss.item(),
            "wm_sparsity": sparsity,
        }
        metrics.update(extra)

        return total_loss, metrics
