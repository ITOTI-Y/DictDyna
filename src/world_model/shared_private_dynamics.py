"""Shared-Private Dictionary Dynamics Model.

Splits dictionary atoms into:
- Shared atoms (frozen, from cross-building pretrain): universal thermal patterns
- Private atoms (trainable, per-building): building-specific adaptation

Prediction: delta = alpha_shared @ D_shared.T + alpha_private @ D_private.T
"""

import torch
import torch.nn as nn


class SharedPrivateDynamics(nn.Module):
    """Shared-Private dictionary dynamics model.

    Args:
        shared_dict: Shared dictionary from pretrain, shape (d, K_shared). Frozen.
        n_private_atoms: Number of private atoms per building (trainable).
        sparse_encoder: SparseEncoder that outputs (K_shared + K_private) codes.
        topk_shared: Top-k for shared atoms.
        topk_private: Top-k for private atoms.
    """

    def __init__(
        self,
        shared_dict: torch.Tensor,
        n_private_atoms: int = 64,
        topk_shared: int = 8,
        topk_private: int = 8,
    ) -> None:
        super().__init__()
        self.state_dim = shared_dict.shape[0]
        self.n_shared = shared_dict.shape[1]
        self.n_private = n_private_atoms
        self.topk_shared = topk_shared
        self.topk_private = topk_private

        # Shared dictionary: frozen (cross-building knowledge)
        self.register_buffer("shared_dict", shared_dict.clone())

        # Private dictionary: trainable (initialized randomly, unit norm)
        private_init = torch.randn(self.state_dim, n_private_atoms)
        private_init = private_init / private_init.norm(dim=0, keepdim=True)
        self.private_dict = nn.Parameter(private_init)

    @property
    def n_atoms(self) -> int:
        return self.n_shared + self.n_private

    def forward(
        self,
        alpha_shared: torch.Tensor,
        alpha_private: torch.Tensor,
    ) -> torch.Tensor:
        """Compute state delta from shared + private codes.

        Args:
            alpha_shared: Sparse codes for shared atoms, shape (batch, K_shared).
            alpha_private: Sparse codes for private atoms, shape (batch, K_private).

        Returns:
            State delta, shape (batch, d).
        """
        delta_shared = alpha_shared @ self.shared_dict.T  # ty: ignore[unsupported-operator]
        delta_private = alpha_private @ self.private_dict.T
        return delta_shared + delta_private

    def normalize_private_atoms(self) -> None:
        """Normalize private dictionary columns to unit L2 norm."""
        with torch.no_grad():
            norms = torch.norm(self.private_dict, dim=0, keepdim=True)
            norms = torch.clamp(norms, min=1e-10)
            self.private_dict.div_(norms)


class SharedPrivateEncoder(nn.Module):
    """Encoder that outputs separate shared and private sparse codes.

    Architecture: shared MLP trunk → two heads:
      - shared_head → alpha_shared (topk_shared sparsity)
      - adapter[building_id] → alpha_private (topk_private sparsity)

    Args:
        state_dim: State dimension.
        action_dim: Action dimension.
        n_shared_atoms: Number of shared dictionary atoms.
        n_private_atoms: Number of private dictionary atoms.
        hidden_dims: Shared trunk hidden layers.
        adapter_dim: Per-building adapter hidden dim.
        n_buildings: Number of buildings.
        topk_shared: Top-k for shared codes.
        topk_private: Top-k for private codes.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_shared_atoms: int = 64,
        n_private_atoms: int = 64,
        hidden_dims: list[int] | None = None,
        adapter_dim: int = 64,
        n_buildings: int = 1,
        topk_shared: int = 8,
        topk_private: int = 8,
    ) -> None:
        super().__init__()
        self.topk_shared = topk_shared
        self.topk_private = topk_private

        hidden_dims = hidden_dims or [256, 256]
        layers: list[nn.Module] = []
        in_dim = state_dim + action_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        # Shared head (same for all buildings)
        self.shared_head = nn.Linear(in_dim, n_shared_atoms)

        # Private adapters (per-building)
        self.adapters = nn.ModuleDict()
        for i in range(n_buildings):
            self.adapters[str(i)] = nn.Sequential(
                nn.Linear(in_dim, adapter_dim),
                nn.ReLU(),
                nn.Linear(adapter_dim, n_private_atoms),
            )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        building_id: str = "0",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode to shared + private sparse codes.

        Returns:
            (alpha_shared, alpha_private)
        """
        x = torch.cat([state, action], dim=-1)
        h = self.trunk(x)

        alpha_shared = self._topk(self.shared_head(h), self.topk_shared)
        alpha_private = self._topk(self.adapters[building_id](h), self.topk_private)

        return alpha_shared, alpha_private

    def add_adapter(
        self, building_id: str, adapter_dim: int = 64, n_private: int = 64
    ) -> None:
        """Add adapter for a new building."""
        in_dim: int = self.trunk[-2].out_features  # type: ignore[assignment]
        self.adapters[building_id] = nn.Sequential(
            nn.Linear(in_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, n_private),
        )

    @staticmethod
    def _topk(x: torch.Tensor, k: int) -> torch.Tensor:
        """Keep only top-k absolute values."""
        _, indices = torch.topk(x.abs(), k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, indices, 1.0)
        return x * mask

    def get_shared_params(self) -> list[nn.Parameter]:
        return list(self.trunk.parameters()) + list(self.shared_head.parameters())

    def get_adapter_params(self, building_id: str) -> list[nn.Parameter]:
        return list(self.adapters[building_id].parameters())


class SharedPrivateWorldModel(nn.Module):
    """Complete shared-private world model.

    s' = s + D_shared @ alpha_shared + D_private @ alpha_private
    """

    def __init__(
        self,
        shared_dict: torch.Tensor,
        n_private_atoms: int = 64,
        state_dim: int = 17,
        action_dim: int = 2,
        hidden_dims: list[int] | None = None,
        adapter_dim: int = 64,
        n_buildings: int = 1,
        topk_shared: int = 8,
        topk_private: int = 8,
        dim_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        n_shared = shared_dict.shape[1]

        # Dimension-weighted loss
        if dim_weights is not None:
            self.register_buffer("dim_weights", dim_weights)
        else:
            self.dim_weights: torch.Tensor | None = None

        self.encoder = SharedPrivateEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            n_shared_atoms=n_shared,
            n_private_atoms=n_private_atoms,
            hidden_dims=hidden_dims,
            adapter_dim=adapter_dim,
            n_buildings=n_buildings,
            topk_shared=topk_shared,
            topk_private=topk_private,
        )

        self.dynamics = SharedPrivateDynamics(
            shared_dict=shared_dict,
            n_private_atoms=n_private_atoms,
            topk_shared=topk_shared,
            topk_private=topk_private,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        building_id: str = "0",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next state. Compatible with DictDynamicsModel interface.

        Returns:
            (next_state, alpha_combined) where alpha_combined = cat(shared, private)
        """
        alpha_s, alpha_p = self.encoder(state, action, building_id)
        delta = self.dynamics(alpha_s, alpha_p)
        next_state = state + delta
        # Concat for compatibility with rollout.py (expects single alpha)
        alpha = torch.cat([alpha_s, alpha_p], dim=-1)
        return next_state, alpha

    def predict(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        building_id: str = "0",
    ) -> torch.Tensor:
        next_state, _ = self.forward(state, action, building_id)
        return next_state

    def compute_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        building_id: str = "0",
        sparsity_lambda: float = 0.1,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss with separate sparsity for shared/private."""
        # Get separate alpha_s and alpha_p from encoder directly
        alpha_s, alpha_p = self.encoder(state, action, building_id)
        delta = self.dynamics(alpha_s, alpha_p)
        pred = state + delta

        per_dim_sq_err = (next_state - pred) ** 2
        if self.dim_weights is not None:
            per_dim_sq_err = per_dim_sq_err * self.dim_weights
        per_sample_mse = per_dim_sq_err.mean(dim=-1)
        if sample_weights is not None:
            mse_loss = (per_sample_mse * sample_weights).mean()
        else:
            mse_loss = per_sample_mse.mean()

        l1_shared = torch.mean(torch.abs(alpha_s))
        l1_private = torch.mean(torch.abs(alpha_p))
        total_loss = mse_loss + sparsity_lambda * (l1_shared + l1_private)

        sparsity_s = (alpha_s.abs() < 1e-6).float().mean().item()
        sparsity_p = (alpha_p.abs() < 1e-6).float().mean().item()

        metrics: dict[str, float] = {
            "mse_loss": mse_loss.item(),
            "l1_shared": l1_shared.item(),
            "l1_private": l1_private.item(),
            "total_loss": total_loss.item(),
            "sparsity_shared": sparsity_s,
            "sparsity_private": sparsity_p,
        }
        if self.dim_weights is not None:
            raw_sq_err = (next_state - pred) ** 2
            reward_mask = self.dim_weights > 1.0
            metrics["mse_reward_dims"] = raw_sq_err[:, reward_mask].mean().item()
            metrics["mse_other_dims"] = raw_sq_err[:, ~reward_mask].mean().item()

        return total_loss, metrics

    @property
    def dictionary(self) -> nn.Parameter:
        """Private dict as 'dictionary' for WorldModelTrainer compatibility."""
        return self.dynamics.private_dict

    def normalize_atoms(self) -> None:
        self.dynamics.normalize_private_atoms()
