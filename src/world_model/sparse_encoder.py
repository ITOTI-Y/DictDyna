"""Sparse encoder g_theta(s,a;phi_i) for DictDyna world model."""

import torch
import torch.nn as nn


class BuildingAdapter(nn.Module):
    """Per-building adapter layer (phi_i).

    Small MLP that modulates the shared encoder output
    for a specific building.

    Args:
        input_dim: Input dimension from shared encoder.
        hidden_dim: Adapter hidden dimension.
        output_dim: Output dimension (n_atoms K).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseEncoder(nn.Module):
    """Sparse encoder: shared MLP + per-building adapter -> sparse alpha.

    Architecture: shared_trunk(s, a) -> adapter_i -> alpha
    Sparsity enforced via L1 penalty, top-k, or proximal gradient.

    Args:
        state_dim: State dimension d.
        action_dim: Action dimension m.
        n_atoms: Number of dictionary atoms K.
        shared_hidden_dims: Hidden layer sizes for shared trunk.
        adapter_dim: Per-building adapter hidden dimension.
        n_buildings: Number of buildings (adapters to create).
        activation: Activation function name.
        sparsity_method: "l1_penalty", "topk", or "proximal".
        topk_k: If sparsity_method="topk", keep top-k activations.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_atoms: int,
        shared_hidden_dims: list[int] | None = None,
        adapter_dim: int = 64,
        n_buildings: int = 1,
        activation: str = "relu",
        sparsity_method: str = "l1_penalty",
        topk_k: int = 16,
    ) -> None:
        super().__init__()
        self.n_atoms = n_atoms
        self.sparsity_method = sparsity_method
        self.topk_k = topk_k

        shared_hidden_dims = shared_hidden_dims or [256, 256]
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]

        # Shared trunk
        layers: list[nn.Module] = []
        in_dim = state_dim + action_dim
        for h in shared_hidden_dims:
            layers.extend([nn.Linear(in_dim, h), act_fn()])
            in_dim = h
        self.shared_trunk = nn.Sequential(*layers)
        self.shared_output_dim = in_dim

        # Per-building adapters
        self.adapters = nn.ModuleDict()
        for i in range(n_buildings):
            self.adapters[str(i)] = BuildingAdapter(in_dim, adapter_dim, n_atoms)

    def add_adapter(self, building_id: str, adapter_dim: int | None = None) -> None:
        """Add a new building adapter (for transfer learning)."""
        if adapter_dim is None:
            # Infer from existing adapter
            existing = next(iter(self.adapters.values()))
            adapter_dim = existing.net[0].out_features
        self.adapters[building_id] = BuildingAdapter(
            self.shared_output_dim, adapter_dim, self.n_atoms
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        building_id: str = "0",
    ) -> torch.Tensor:
        """Encode (s, a) to sparse coefficient vector alpha.

        Args:
            state: States, shape (batch, d).
            action: Actions, shape (batch, m).
            building_id: Building identifier for adapter selection.

        Returns:
            Sparse codes alpha, shape (batch, K).
        """
        x = torch.cat([state, action], dim=-1)
        h = self.shared_trunk(x)
        alpha = self.adapters[building_id](h)

        # Apply sparsity
        if self.sparsity_method == "topk":
            alpha = self._topk_sparsify(alpha)

        return alpha

    def _topk_sparsify(self, alpha: torch.Tensor) -> torch.Tensor:
        """Keep only top-k absolute values, zero out the rest."""
        _, indices = torch.topk(alpha.abs(), self.topk_k, dim=-1)
        mask = torch.zeros_like(alpha)
        mask.scatter_(-1, indices, 1.0)
        return alpha * mask

    def get_shared_params(self) -> list[nn.Parameter]:
        """Get shared trunk parameters (theta)."""
        return list(self.shared_trunk.parameters())

    def get_adapter_params(self, building_id: str) -> list[nn.Parameter]:
        """Get adapter parameters for a specific building (phi_i)."""
        return list(self.adapters[building_id].parameters())

    def get_all_adapter_params(self) -> list[nn.Parameter]:
        """Get all adapter parameters."""
        params = []
        for adapter in self.adapters.values():
            params.extend(adapter.parameters())
        return params
