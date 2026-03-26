"""Context-conditioned encoder for DictDyna world model.

Replaces discrete adapter routing (nn.ModuleDict + building_id) with a
continuous context vector z inferred from recent transitions. Inspired by
CaDM (Lee et al., ICML 2020) and DALI (NeurIPS 2025).

Architecture:
    z = ContextEncoder(recent K transitions)
    alpha = ContextConditionedEncoder(s, a, z)  → topk sparse codes
"""

import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """Infer building thermal fingerprint from recent transitions.

    Takes a window of K recent (s, a, Δs) transitions and produces a
    continuous context vector z ∈ R^context_dim. Uses a per-transition
    MLP with mean-pooling (permutation-invariant).

    Args:
        state_dim: State dimension d.
        action_dim: Action dimension m.
        context_dim: Output context vector dimension.
        hidden_dims: Hidden layer sizes for per-transition MLP.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        context_dim: int = 16,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        # Each transition: (s, a, Δs) → dim = 2*state_dim + action_dim
        transition_dim = 2 * state_dim + action_dim

        hidden_dims = hidden_dims or [128, 128]

        # Per-transition MLP (shared weights across window)
        layers: list[nn.Module] = []
        in_dim = transition_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.transition_net = nn.Sequential(*layers)

        # Final projection after mean-pooling
        self.projection = nn.Linear(in_dim, context_dim)

    def forward(self, transitions: torch.Tensor) -> torch.Tensor:
        """Infer context from transition window.

        Args:
            transitions: Shape (batch, K, 2*d + m) where each row is (s, a, Δs).

        Returns:
            Context vector z, shape (batch, context_dim).
        """
        # Per-transition encoding: (batch, K, hidden)
        h = self.transition_net(transitions)
        # Mean-pool across window: (batch, hidden)
        h_pooled = h.mean(dim=1)
        # Project to context space: (batch, context_dim)
        return self.projection(h_pooled)


class ContextConditionedEncoder(nn.Module):
    """Sparse encoder conditioned on context vector.

    Replaces SparseEncoder's per-building adapter routing with continuous
    context conditioning via concatenation: trunk input = (s, a, z).

    Args:
        state_dim: State dimension d.
        action_dim: Action dimension m.
        context_dim: Context vector dimension.
        n_atoms: Number of dictionary atoms K.
        shared_hidden_dims: Hidden layer sizes for trunk MLP.
        activation: Activation function name.
        sparsity_method: "topk" or "l1_penalty".
        topk_k: If sparsity_method="topk", keep top-k activations.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        context_dim: int,
        n_atoms: int,
        shared_hidden_dims: list[int] | None = None,
        activation: str = "relu",
        sparsity_method: str = "topk",
        topk_k: int = 16,
        use_layernorm: bool = False,
        soft_topk_temperature: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_atoms = n_atoms
        self.sparsity_method = sparsity_method
        self.topk_k = topk_k
        self.soft_topk_temperature = soft_topk_temperature

        shared_hidden_dims = shared_hidden_dims or [256, 256]
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]

        # Trunk: (s, a, z) → hidden → n_atoms
        layers: list[nn.Module] = []
        in_dim = state_dim + action_dim + context_dim
        for h in shared_hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(act_fn())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_atoms))
        self.trunk = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Encode (s, a, z) to sparse coefficient vector alpha.

        Args:
            state: States, shape (batch, d).
            action: Actions, shape (batch, m).
            context: Context vector, shape (batch, context_dim).

        Returns:
            Sparse codes alpha, shape (batch, K).
        """
        x = torch.cat([state, action, context], dim=-1)
        alpha = self.trunk(x)

        if self.sparsity_method == "topk":
            alpha = self._topk_sparsify(alpha)

        return alpha

    def _topk_sparsify(self, alpha: torch.Tensor) -> torch.Tensor:
        """Apply top-k sparsity with optional soft relaxation."""
        if self.soft_topk_temperature > 0 and self.training:
            abs_alpha = alpha.abs()
            kth_val = abs_alpha.topk(self.topk_k, dim=-1).values[:, -1:]
            mask = torch.sigmoid((abs_alpha - kth_val) / self.soft_topk_temperature)
            return alpha * mask
        _, indices = torch.topk(alpha.abs(), self.topk_k, dim=-1)
        mask = torch.zeros_like(alpha)
        mask.scatter_(-1, indices, 1.0)
        return alpha * mask
