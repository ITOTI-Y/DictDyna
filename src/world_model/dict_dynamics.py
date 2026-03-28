"""Dictionary dynamics model with adapter-based building routing.

Uses SparseEncoder with per-building adapters to produce sparse codes.
Falls back mode for single-building experiments; see ContextDynamicsModel
for the recommended context-conditioned approach.
"""

import torch

from src.world_model._share import BaseDictDynamics
from src.world_model.sparse_encoder import SparseEncoder


class DictDynamicsModel(BaseDictDynamics):
    """Adapter-based dictionary dynamics model.

    Core equation: s' = s + D @ adapter[building_id](trunk(s, a)) + residual(h)

    Args:
        dictionary: Initial dictionary tensor, shape (d, K).
        sparse_encoder: SparseEncoder with per-building adapters.
        learnable_dict: Whether D is a learnable parameter.
        diff_mean: Mean of state diffs (for space conversion).
        diff_std: Std of state diffs (for space conversion).
        obs_std: Std of observations (for space conversion).
        dim_weights: Static per-dimension loss weights.
        residual_hidden_dim: Residual correction head hidden dim (0=disabled).
        controllable_dims: If set, only predict these dimensions.
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
        super().__init__(
            dictionary=dictionary,
            learnable_dict=learnable_dict,
            diff_mean=diff_mean,
            diff_std=diff_std,
            obs_std=obs_std,
            dim_weights=dim_weights,
            residual_hidden_dim=residual_hidden_dim,
            controllable_dims=controllable_dims,
            trunk_output_dim=sparse_encoder.shared_output_dim,
        )
        self.encoder = sparse_encoder

    def _encode(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        building_id: str = "0",
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        h = self.encoder.shared_trunk(x)
        alpha = self.encoder.adapters[building_id](h)
        if self.encoder.sparsity_method == "topk":
            alpha = self.encoder._topk_sparsify(alpha)
        return alpha, h
