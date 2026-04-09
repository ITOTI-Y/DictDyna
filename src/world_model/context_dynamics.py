"""Context-conditioned dictionary dynamics model.

Replaces adapter-based routing with continuous context conditioning.
The context vector z is inferred by ContextEncoder from recent transitions
and fed to ContextConditionedEncoder alongside (s, a).

Core equation: s' = s + D @ alpha(s, a, z) + residual(h)
"""

import torch

from src.world_model._share import BaseDictDynamics
from src.world_model.context_encoder import ContextConditionedEncoder, ContextEncoder


class ContextDynamicsModel(BaseDictDynamics):
    """Context-conditioned dictionary dynamics model.

    Uses a continuous context vector z (inferred from recent transitions)
    instead of discrete per-building adapters.

    Args:
        dictionary: Pretrained dictionary tensor, shape (d, K).
        context_encoder: ContextEncoder that infers z from transitions.
        conditioned_encoder: ContextConditionedEncoder that produces alpha.
        learnable_dict: If True, dictionary is an nn.Parameter.
        diff_mean: Diff-space mean for space conversion (optional).
        diff_std: Diff-space std for space conversion (optional).
        obs_std: Obs-space std for space conversion (optional).
        dim_weights: Static per-dimension loss weights.
        residual_hidden_dim: Residual correction head hidden dim (0=disabled).
        controllable_dims: If set, only predict these dimensions.
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
        dim_weights: torch.Tensor | None = None,
        residual_hidden_dim: int = 0,
        controllable_dims: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(
            dictionary=dictionary,
            learnable_dict=learnable_dict,
            diff_mean=diff_mean,
            diff_std=diff_std,
            obs_std=obs_std,
            dim_weights=dim_weights,
            residual_hidden_dim=residual_hidden_dim,
            controllable_dims=controllable_dims,
            trunk_output_dim=conditioned_encoder.shared_output_dim,
        )
        self.context_encoder = context_encoder
        self.encoder = conditioned_encoder

    def infer_context(self, transitions: torch.Tensor) -> torch.Tensor:
        """Infer context vector from transition window.

        Args:
            transitions: Shape (batch, K, 2*d + m).

        Returns:
            Context vector z, shape (batch, context_dim).
        """
        return self.context_encoder(transitions)

    def _encode(  # ty: ignore[invalid-method-override]
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        context: torch.Tensor,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action, context], dim=-1)
        h = self.encoder.shared_trunk(x)
        alpha = self.encoder.head(h)
        # Context gating: modulate before sparsification
        alpha = self.encoder.apply_gating(alpha, context)
        if self.encoder.sparsity_method == "topk":
            alpha = self.encoder._topk_sparsify(alpha)
        return alpha, h
