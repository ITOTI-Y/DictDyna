"""Dictionary dynamics model with space conversion.

When policy operates in obs-normalized space and dictionary in diff-normalized space:
  D*alpha is in diff-norm space -> convert to obs-norm: delta_obs = D*alpha * diff_std / obs_std
  s'_norm = s_norm + delta_obs + diff_mean / obs_std
"""

import torch
import torch.nn as nn

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
    ) -> None:
        super().__init__()
        self.encoder = sparse_encoder

        if learnable_dict:
            self.dictionary = nn.Parameter(dictionary.clone())
        else:
            self.register_buffer("dictionary", dictionary.clone())

        # Space conversion buffers (None = raw mode, no conversion)
        self._has_conversion = diff_std is not None and obs_std is not None
        if self._has_conversion:
            # scale converts diff-norm space to obs-norm space: diff_std / obs_std
            scale = diff_std / obs_std  # ty: ignore[unsupported-operator]
            self.register_buffer("_scale", scale)
            # bias accounts for diff_mean in obs-norm space: diff_mean / obs_std
            bias = (
                diff_mean / obs_std
                if diff_mean is not None
                else torch.zeros_like(scale)
            )  # ty: ignore[unsupported-operator]
            self.register_buffer("_bias", bias)

    @property
    def n_atoms(self) -> int:
        return self.dictionary.shape[1]  # ty: ignore[not-subscriptable]

    @property
    def state_dim(self) -> int:
        return self.dictionary.shape[0]  # ty: ignore[not-subscriptable]

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
        delta = alpha @ self.dictionary.T  # ty: ignore[unsupported-operator]

        if self._has_conversion:
            # Convert from diff-norm to obs-norm space
            delta = delta * self._scale + self._bias  # ty: ignore[unsupported-operator]

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
            self.dictionary.div_(norms)  # ty: ignore[unresolved-attribute]

    def compute_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        building_id: str = "0",
        sparsity_lambda: float = 0.1,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute world model loss: weighted MSE + lambda * L1.

        Args:
            sample_weights: Per-sample importance weights, shape (batch,).
                If provided, MSE is weighted: transitions with higher
                TD-error or reward magnitude are reconstructed more accurately.
        """
        pred_next, alpha = self.forward(state, action, building_id)

        per_sample_mse = ((next_state - pred_next) ** 2).mean(dim=-1)  # (batch,)
        if sample_weights is not None:
            mse_loss = (per_sample_mse * sample_weights).mean()
        else:
            mse_loss = per_sample_mse.mean()
        l1_loss = torch.mean(torch.abs(alpha))
        total_loss = mse_loss + sparsity_lambda * l1_loss

        sparsity = (alpha.abs() < 1e-6).float().mean().item()

        return total_loss, {
            "mse_loss": mse_loss.item(),
            "l1_loss": l1_loss.item(),
            "total_loss": total_loss.item(),
            "sparsity": sparsity,
        }
