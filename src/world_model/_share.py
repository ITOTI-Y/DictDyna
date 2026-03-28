"""Shared utilities for world model modules."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.world_model.loss_utils import compute_dim_weighted_mse

# Numerical stability constants
NORM_EPS = 1e-10
STABILITY_EPS = 1e-8


def normalize_atoms(dictionary: nn.Parameter | torch.Tensor) -> None:
    """Normalize dictionary columns to unit L2 norm in-place."""
    with torch.no_grad():
        norms = torch.norm(dictionary, dim=0, keepdim=True)
        norms = torch.clamp(norms, min=NORM_EPS)
        dictionary.div_(norms)


def topk_sparsify(
    alpha: torch.Tensor,
    k: int,
    soft_temperature: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Apply top-k sparsity with optional soft relaxation.

    Args:
        alpha: Activation tensor, shape (batch, K).
        k: Number of top activations to keep.
        soft_temperature: If > 0 and training, use differentiable soft mask.
        training: Whether in training mode.
    """
    if soft_temperature > 0 and training:
        abs_alpha = alpha.abs()
        kth_val = abs_alpha.topk(k, dim=-1).values[:, -1:]
        mask = torch.sigmoid((abs_alpha - kth_val) / soft_temperature)
        return alpha * mask
    # Hard top-k (inference or temperature=0)
    _, indices = torch.topk(alpha.abs(), k, dim=-1)
    mask = torch.zeros_like(alpha)
    mask.scatter_(-1, indices, 1.0)
    return alpha * mask


def build_residual_head(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> nn.Sequential:
    """Build a residual correction MLP with small-init output layer."""
    out_layer = nn.Linear(hidden_dim, output_dim)
    nn.init.normal_(out_layer.weight, std=0.01)
    nn.init.zeros_(out_layer.bias)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        out_layer,
    )


def apply_space_conversion(
    delta_raw: torch.Tensor,
    state: torch.Tensor,
    controllable_dims: tuple[int, ...] | None,
    scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    has_conversion: bool,
) -> torch.Tensor:
    """Convert delta from diff-norm space to obs-norm space.

    Handles both full-obs and controllable-only prediction modes.

    Args:
        delta_raw: Raw prediction delta (batch, d_pred).
        state: Current state for zero-padding (batch, d_full).
        controllable_dims: If set, embed delta into full obs space.
        scale: diff_std / obs_std conversion factor.
        bias: diff_mean / obs_std conversion offset.
        has_conversion: Whether space conversion is enabled.
    """
    if controllable_dims is not None:
        delta = torch.zeros_like(state)
        if has_conversion:
            delta_raw = delta_raw * scale + bias  # ty: ignore[unsupported-operator]
        delta[:, controllable_dims] = delta_raw
    else:
        delta = delta_raw
        if has_conversion:
            delta = delta * scale + bias  # ty: ignore[unsupported-operator]
    return delta


class BaseDictDynamics(nn.Module, ABC):
    """Abstract base for dictionary-based dynamics models.

    Subclasses implement ``_encode(state, action, **kwargs) -> (alpha, h)``
    where *alpha* are sparse codes and *h* is the trunk hidden features
    (used by the optional residual correction head).

    Shared logic: dictionary management, space conversion, residual head,
    dimension-weighted loss, controllable-dims masking.
    """

    def __init__(
        self,
        dictionary: torch.Tensor,
        learnable_dict: bool = True,
        diff_mean: torch.Tensor | None = None,
        diff_std: torch.Tensor | None = None,
        obs_std: torch.Tensor | None = None,
        dim_weights: torch.Tensor | None = None,
        residual_hidden_dim: int = 0,
        controllable_dims: tuple[int, ...] | None = None,
        trunk_output_dim: int = 0,
    ) -> None:
        super().__init__()
        self.controllable_dims = controllable_dims

        if learnable_dict:
            self.dictionary = nn.Parameter(dictionary.clone())
        else:
            self.register_buffer("dictionary", dictionary.clone())

        # Per-dimension MSE EMA for adaptive loss weighting
        self.register_buffer("_dim_ema", torch.ones(dictionary.shape[0]))

        # Space conversion buffers (None = raw mode, no conversion)
        self._has_conversion = diff_std is not None and obs_std is not None
        if self._has_conversion:
            assert diff_std is not None and obs_std is not None
            scale = diff_std / obs_std
            bias = (
                diff_mean / obs_std
                if diff_mean is not None
                else torch.zeros_like(scale)
            )
        else:
            scale = None
            bias = None
        self.register_buffer("_scale", scale)
        self.register_buffer("_bias", bias)

        # Dimension-weighted loss
        if dim_weights is not None:
            self.register_buffer("dim_weights", dim_weights)
        else:
            self.dim_weights: torch.Tensor | None = None

        # Nonlinear residual correction head
        if residual_hidden_dim > 0 and trunk_output_dim > 0:
            self.residual_head: nn.Module | None = build_residual_head(
                trunk_output_dim,
                residual_hidden_dim,
                dictionary.shape[0],
            )
        else:
            self.residual_head = None
        self._last_residual: torch.Tensor | None = None

    @property
    def n_atoms(self) -> int:
        return self.dictionary.shape[1]

    @property
    def state_dim(self) -> int:
        return self.dictionary.shape[0]

    def normalize_atoms(self) -> None:
        """Normalize dictionary columns to unit L2 norm."""
        normalize_atoms(self.dictionary)

    @abstractmethod
    def _encode(
        self, state: torch.Tensor, action: torch.Tensor, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce sparse codes and trunk features.

        Returns:
            (alpha, h) where alpha has shape (batch, K) and h has shape
            (batch, trunk_dim).
        """
        ...

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next state.

        Keyword arguments are forwarded to ``_encode`` (e.g. ``building_id``
        or ``context``).

        Returns:
            (predicted_next_state, alpha)
        """
        alpha, h = self._encode(state, action, **kwargs)

        delta_raw = alpha @ self.dictionary.T

        # Nonlinear residual correction
        if self.residual_head is not None:
            residual = self.residual_head(h)
            delta_raw = delta_raw + residual
            self._last_residual = residual
        else:
            self._last_residual = None

        # Space conversion: diff-norm → obs-norm + controllable-only embedding
        delta = apply_space_conversion(
            delta_raw,
            state,
            self.controllable_dims,
            self._scale,  # type: ignore[arg-type]
            self._bias,  # type: ignore[arg-type]
            self._has_conversion,
        )

        next_state = state + delta
        return next_state, alpha

    def predict(
        self, state: torch.Tensor, action: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        """Predict next state (without returning alpha)."""
        next_state, _ = self.forward(state, action, **kwargs)
        return next_state

    def compute_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        *,
        sparsity_lambda: float = 0.1,
        sample_weights: torch.Tensor | None = None,
        identity_penalty_lambda: float = 0.0,
        dim_weight_ema_decay: float = 0.99,
        use_dim_weighting: bool = False,
        residual_lambda: float = 0.0,
        **forward_kwargs: object,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute world model loss: weighted MSE + L1 sparsity + residual L2.

        Forward keyword arguments (``building_id`` or ``context``) are passed
        through ``**forward_kwargs``.
        """
        pred_next, alpha = self.forward(state, action, **forward_kwargs)

        # In controllable-only mode, compute loss only on predicted dims
        if self.controllable_dims is not None:
            ctrl = list(self.controllable_dims)
            loss_pred = pred_next[:, ctrl]
            loss_target = next_state[:, ctrl]
            loss_state = state[:, ctrl]
        else:
            loss_pred = pred_next
            loss_target = next_state
            loss_state = state

        if use_dim_weighting or identity_penalty_lambda > 0:
            mse_loss, extra = compute_dim_weighted_mse(
                pred=loss_pred,
                target=loss_target,
                state=loss_state,
                dim_ema=self._dim_ema,  # ty: ignore[invalid-argument-type]
                ema_decay=dim_weight_ema_decay,
                identity_penalty_lambda=identity_penalty_lambda,
                sample_weights=sample_weights,
                training=self.training,
            )
        else:
            per_dim_sq_err = (loss_target - loss_pred) ** 2
            if self.dim_weights is not None:
                per_dim_sq_err = per_dim_sq_err * self.dim_weights
            per_sample_mse = per_dim_sq_err.mean(dim=-1)
            if sample_weights is not None:
                mse_loss = (per_sample_mse * sample_weights).mean()
            else:
                mse_loss = per_sample_mse.mean()
            extra = {}

        l1_loss = torch.mean(torch.abs(alpha))
        total_loss = mse_loss + sparsity_lambda * l1_loss

        # Residual L2 regularization
        if self._last_residual is not None and residual_lambda > 0:
            total_loss = total_loss + residual_lambda * (self._last_residual**2).mean()

        sparsity = (alpha.abs() < 1e-6).float().mean().item()

        metrics: dict[str, float] = {
            "mse": mse_loss.item(),
            "l1": l1_loss.item(),
            "loss": total_loss.item(),
            "sparsity": sparsity,
        }
        metrics.update(extra)
        if self.dim_weights is not None:
            raw_sq_err = (next_state - pred_next) ** 2
            reward_mask = self.dim_weights > 1.0
            metrics["mse_reward_dims"] = raw_sq_err[:, reward_mask].mean().item()
            metrics["mse_other_dims"] = raw_sq_err[:, ~reward_mask].mean().item()
        if self._last_residual is not None:
            metrics["residual_norm"] = self._last_residual.norm(dim=-1).mean().item()

        return total_loss, metrics
