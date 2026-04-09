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

from src.world_model._share import topk_sparsify


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

    Optionally applies context-to-sparse gating: context z generates
    atom-level gates that modulate sparse codes before top-k selection.
    This allows different buildings to activate different sparse subspaces
    on the shared dictionary.

    Args:
        state_dim: State dimension d.
        action_dim: Action dimension m.
        context_dim: Context vector dimension.
        n_atoms: Number of dictionary atoms K.
        shared_hidden_dims: Hidden layer sizes for trunk MLP.
        activation: Activation function name.
        sparsity_method: "topk" or "l1_penalty".
        topk_k: If sparsity_method="topk", keep top-k activations.
        use_context_gating: If True, apply context-conditioned atom gating.
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
        use_context_gating: bool = False,
    ) -> None:
        super().__init__()
        self.n_atoms = n_atoms
        self.sparsity_method = sparsity_method
        self.topk_k = topk_k
        self.soft_topk_temperature = soft_topk_temperature
        self.use_context_gating = use_context_gating

        shared_hidden_dims = shared_hidden_dims or [256, 256]
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]

        # Shared trunk: (s, a, z) → hidden features
        layers: list[nn.Module] = []
        in_dim = state_dim + action_dim + context_dim
        for h in shared_hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(act_fn())
            in_dim = h
        self.shared_trunk = nn.Sequential(*layers)
        self.shared_output_dim = in_dim

        # Output head: hidden → n_atoms
        self.head = nn.Linear(in_dim, n_atoms)

        # Context-to-sparse gating: z → atom-level gate ∈ (0, 1)^K
        # Gate modulates alpha BEFORE top-k, so context influences which
        # atoms are selected. Small-init bias to start near-uniform.
        if use_context_gating:
            self.gate_net = nn.Linear(context_dim, n_atoms)
            nn.init.normal_(self.gate_net.weight, std=0.01)
            nn.init.zeros_(self.gate_net.bias)

    def apply_gating(self, alpha: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply context-conditioned gating to sparse codes.

        Args:
            alpha: Raw sparse codes, shape (batch, K).
            context: Context vector, shape (batch, context_dim).

        Returns:
            Gated sparse codes, shape (batch, K).
        """
        if not self.use_context_gating:
            return alpha
        gate = torch.sigmoid(self.gate_net(context))  # (batch, K)
        return alpha * gate

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
        h = self.shared_trunk(x)
        alpha = self.head(h)

        # Context gating: modulate before sparsification
        alpha = self.apply_gating(alpha, context)

        if self.sparsity_method == "topk":
            alpha = self._topk_sparsify(alpha)

        return alpha

    def _topk_sparsify(self, alpha: torch.Tensor) -> torch.Tensor:
        """Apply top-k sparsity with optional soft relaxation."""
        return topk_sparsify(
            alpha, self.topk_k, self.soft_topk_temperature, self.training
        )
