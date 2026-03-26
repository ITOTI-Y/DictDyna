"""Probabilistic dictionary dynamics with per-atom heteroscedastic variance.

Extends DictDynamicsModel with learned uncertainty estimation. Each sparse
code alpha_k has an associated variance, which propagates through the dictionary
to give per-dimension prediction uncertainty:

    pred_var[d] = sum_k D[d,k]^2 * alpha_var[k]

This enables:
- NLL training loss for better calibration than MSE
- Uncertainty-aware rollouts with pessimistic reward penalization
- Automatic down-weighting of unreliable dimensions (e.g., HVAC setpoints)

Novel contribution: heteroscedastic uncertainty in sparse code space,
unique to the dictionary learning + MBRL combination.
"""

import torch
import torch.nn as nn

from src.world_model.loss_utils import compute_dim_weighted_mse
from src.world_model.sparse_encoder import BuildingAdapter, SparseEncoder


class ProbabilisticDictDynamics(nn.Module):
    """Dictionary dynamics model with per-atom variance estimation.

    Architecture:
        shared_trunk(s,a) -> adapter -> alpha_mean  (sparse, topk)
        shared_trunk(s,a) -> var_adapter -> alpha_log_var (same mask)
        alpha_sampled = alpha_mean + eps * exp(0.5 * log_var)  [training]
        s' = s + D @ alpha
        pred_var = alpha_var @ (D^2).T  [for rollout penalization]

    Args:
        dictionary: Initial dictionary tensor, shape (d, K).
        sparse_encoder: SparseEncoder instance (provides shared trunk + mean adapters).
        learnable_dict: Whether D is a learnable parameter.
        min_log_var: Minimum log variance for numerical stability.
        max_log_var: Maximum log variance to prevent explosion.
    """

    def __init__(
        self,
        dictionary: torch.Tensor,
        sparse_encoder: SparseEncoder,
        learnable_dict: bool = True,
        min_log_var: float = -10.0,
        max_log_var: float = 2.0,
    ) -> None:
        super().__init__()
        self.encoder = sparse_encoder
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        if learnable_dict:
            self.dictionary = nn.Parameter(dictionary.clone())
        else:
            self.register_buffer("dictionary", dictionary.clone())

        # Per-dimension MSE EMA for adaptive loss weighting
        self.register_buffer("_dim_ema", torch.ones(dictionary.shape[0]))

        # Variance adapters (parallel to encoder's mean adapters)
        self.var_adapters = nn.ModuleDict()
        for bid, adapter in sparse_encoder.adapters.items():
            hidden_dim = adapter.net[0].out_features  # ty: ignore[not-subscriptable, unresolved-attribute]
            self.var_adapters[bid] = BuildingAdapter(
                sparse_encoder.shared_output_dim,
                hidden_dim,
                sparse_encoder.n_atoms,
            )

        # Cache for rollout variance access
        self._last_pred_var: torch.Tensor | None = None

    @property
    def n_atoms(self) -> int:
        return self.dictionary.shape[1]

    @property
    def state_dim(self) -> int:
        return self.dictionary.shape[0]

    def add_var_adapter(self, building_id: str) -> None:
        """Add variance adapter for a new building (transfer learning)."""
        if building_id in self.encoder.adapters:
            adapter = self.encoder.adapters[building_id]
            hidden_dim = adapter.net[0].out_features  # ty: ignore[not-subscriptable, unresolved-attribute]
            self.var_adapters[building_id] = BuildingAdapter(
                self.encoder.shared_output_dim,
                hidden_dim,
                self.n_atoms,
            )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        building_id: str = "0",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next state with uncertainty estimation.

        During training: samples alpha via reparameterization trick.
        During eval: uses alpha_mean (deterministic).

        Returns:
            (predicted_next_state, alpha) where alpha is sampled or mean.
        """
        # Get shared features from encoder trunk
        x = torch.cat([state, action], dim=-1)
        h = self.encoder.shared_trunk(x)

        # Mean alpha from regular adapter
        alpha_mean = self.encoder.adapters[building_id](h)

        # Log-variance from variance adapter
        alpha_log_var = self.var_adapters[building_id](h)
        alpha_log_var = alpha_log_var.clamp(self.min_log_var, self.max_log_var)

        # Apply topk sparsity to mean (determines active atoms)
        if self.encoder.sparsity_method == "topk":
            alpha_mean = self.encoder._topk_sparsify(alpha_mean)
            # Apply same sparsity mask to log_var
            active_mask = (alpha_mean.abs() > 1e-6).float()
            alpha_log_var = alpha_log_var * active_mask + self.min_log_var * (
                1 - active_mask
            )

        # Reparameterization sampling (training only)
        if self.training:
            eps = torch.randn_like(alpha_mean)
            alpha = alpha_mean + eps * torch.exp(0.5 * alpha_log_var)
        else:
            alpha = alpha_mean

        # Predict next state
        delta = alpha @ self.dictionary.T
        next_state = state + delta

        # Compute and cache per-dimension prediction variance
        alpha_var = torch.exp(alpha_log_var)  # (batch, K)
        dict_sq = self.dictionary**2  # (d, K)
        self._last_pred_var = alpha_var @ dict_sq.T  # (batch, d)

        return next_state, alpha

    def predict(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        building_id: str = "0",
    ) -> torch.Tensor:
        next_state, _ = self.forward(state, action, building_id)
        return next_state

    def get_prediction_std(self) -> torch.Tensor | None:
        """Get per-dimension prediction std from last forward pass."""
        if self._last_pred_var is None:
            return None
        return torch.sqrt(self._last_pred_var + 1e-8)

    def normalize_atoms(self) -> None:
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
        sample_weights: torch.Tensor | None = None,
        identity_penalty_lambda: float = 0.0,
        dim_weight_ema_decay: float = 0.99,
        use_dim_weighting: bool = False,
        nll_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute NLL-based loss with uncertainty estimation.

        Loss = NLL + lambda_sparsity * L1 + lambda_id * identity_guard
        NLL  = 0.5 * mean(log(pred_var) + (target - pred)^2 / pred_var)

        Args:
            nll_weight: Weight for NLL loss (vs fallback to dim-weighted MSE).
        """
        pred_next, alpha = self.forward(state, action, building_id)
        pred_var = self._last_pred_var  # (batch, d), set by forward()
        assert pred_var is not None

        # Clamp variance for numerical stability
        pred_var = pred_var.clamp(min=1e-6)

        # Gaussian NLL: 0.5 * (log(var) + (y - mu)^2 / var)
        sq_err = (next_state - pred_next) ** 2
        nll_per_dim = 0.5 * (torch.log(pred_var) + sq_err / pred_var)  # (batch, d)
        nll_per_sample = nll_per_dim.mean(dim=-1)  # (batch,)

        if sample_weights is not None:
            nll_loss = (nll_per_sample * sample_weights).mean()
        else:
            nll_loss = nll_per_sample.mean()

        # Identity guard (still based on MSE, not NLL)
        identity_penalty = torch.tensor(0.0, device=pred_next.device)
        if identity_penalty_lambda > 0:
            identity_sq_err = (next_state - state) ** 2
            excess = torch.relu(sq_err - identity_sq_err)
            identity_penalty = excess.mean()

        # Optional dim-weighted MSE as auxiliary loss
        if use_dim_weighting:
            mse_loss, dim_extra = compute_dim_weighted_mse(
                pred=pred_next,
                target=next_state,
                state=state,
                dim_ema=self._dim_ema,  # ty: ignore[invalid-argument-type]
                ema_decay=dim_weight_ema_decay,
                identity_penalty_lambda=0.0,
                sample_weights=sample_weights,
                training=self.training,
            )
        else:
            mse_loss = sq_err.mean()
            dim_extra = {}

        # Sparsity penalty
        l1_loss = torch.mean(torch.abs(alpha))

        # Total loss
        total_loss = (
            nll_weight * nll_loss
            + identity_penalty_lambda * identity_penalty
            + sparsity_lambda * l1_loss
        )

        sparsity = (alpha.abs() < 1e-6).float().mean().item()

        metrics = {
            "mse_loss": mse_loss.item(),
            "nll_loss": nll_loss.item(),
            "l1_loss": l1_loss.item(),
            "total_loss": total_loss.item(),
            "identity_penalty": identity_penalty.item(),
            "sparsity": sparsity,
            "pred_var_mean": pred_var.mean().item(),
            "pred_var_max": pred_var.max().item(),
        }
        metrics.update(dim_extra)

        return total_loss, metrics
