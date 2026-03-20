"""Dictionary dynamics model: s' = s + D @ alpha."""

import torch
import torch.nn as nn

from src.world_model.sparse_encoder import SparseEncoder


class DictDynamicsModel(nn.Module):
    """Dictionary-based transition dynamics model.

    Core equation: s_{t+1} = s_t + D @ g_theta(s_t, a_t; phi_i)

    Where D is a (d, K) dictionary matrix and g_theta is the sparse encoder.

    Args:
        dictionary: Initial dictionary tensor, shape (d, K).
        sparse_encoder: SparseEncoder instance.
        learnable_dict: Whether D is a learnable parameter.
    """

    def __init__(
        self,
        dictionary: torch.Tensor,
        sparse_encoder: SparseEncoder,
        learnable_dict: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = sparse_encoder

        if learnable_dict:
            self.dictionary = nn.Parameter(dictionary.clone())
        else:
            self.register_buffer("dictionary", dictionary.clone())

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
            state: Current states, shape (batch, d).
            action: Actions, shape (batch, m).
            building_id: Building identifier.

        Returns:
            (predicted_next_state, alpha) where:
              - predicted_next_state: shape (batch, d)
              - alpha: sparse codes, shape (batch, K)
        """
        alpha = self.encoder(state, action, building_id)
        delta = alpha @ self.dictionary.T  # (batch, d)
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
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute world model loss.

        Loss = MSE(s', s_hat') + lambda * ||alpha||_1

        Args:
            state: Current states, shape (batch, d).
            action: Actions, shape (batch, m).
            next_state: True next states, shape (batch, d).
            building_id: Building identifier.
            sparsity_lambda: L1 sparsity weight.

        Returns:
            (total_loss, metrics_dict)
        """
        pred_next, alpha = self.forward(state, action, building_id)

        mse_loss = torch.mean((next_state - pred_next) ** 2)
        l1_loss = torch.mean(torch.abs(alpha))
        total_loss = mse_loss + sparsity_lambda * l1_loss

        sparsity = (alpha.abs() < 1e-6).float().mean().item()

        return total_loss, {
            "mse_loss": mse_loss.item(),
            "l1_loss": l1_loss.item(),
            "total_loss": total_loss.item(),
            "sparsity": sparsity,
        }
