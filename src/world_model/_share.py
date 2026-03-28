"""Shared utilities for world model modules."""

import torch
import torch.nn as nn

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
