"""World model factory functions.

Centralizes construction of dynamics models and trainers to avoid
duplicated setup code across DynaSAC, MultiBuildingDynaSAC,
FewShotTransferExperiment, and evaluate.py.
"""

from typing import Literal

import torch

from src.obs_config import OBS_CONFIG
from src.schemas import TrainSchema
from src.utils import build_dim_weights
from src.world_model._share import BaseDictDynamics
from src.world_model.dict_dynamics import DictDynamicsModel
from src.world_model.model_trainer import WorldModelTrainer


def build_world_model(
    dictionary: torch.Tensor,
    state_dim: int,
    action_dim: int,
    config: TrainSchema,
    device: torch.device,
    n_buildings: int = 1,
    diff_mean: torch.Tensor | None = None,
    diff_std: torch.Tensor | None = None,
    obs_std: torch.Tensor | None = None,
    mode: Literal["context", "dict"] | None = None,
) -> BaseDictDynamics:
    """Build a world model from configuration.

    Args:
        dictionary: Pretrained dictionary tensor, shape (d, K).
        state_dim: Observation dimension.
        action_dim: Action dimension.
        config: Training configuration (config.mode used if *mode* is None).
        device: Target device.
        n_buildings: Number of buildings (only for dict mode adapters).
        diff_mean: Diff-space mean for space conversion (controllable-only).
        diff_std: Diff-space std for space conversion (controllable-only).
        obs_std: Obs-space std for space conversion (controllable-only).
        mode: Override config.mode. Default None uses config.mode.

    Returns:
        Constructed world model on *device*.
    """
    mode = mode or config.mode

    dim_weights = build_dim_weights(
        state_dim,
        config.dictionary.reward_dims,
        config.dictionary.reward_dim_weight,
        device,
    )
    ctrl_dims = OBS_CONFIG.CONTROLLABLE if config.dictionary.controllable_only else None

    # Space conversion for controllable-only mode
    wm_diff_mean = None
    wm_diff_std = None
    wm_obs_std = None
    if ctrl_dims is not None and diff_std is not None and obs_std is not None:
        wm_diff_mean = diff_mean.to(device) if diff_mean is not None else None
        wm_diff_std = diff_std.to(device)
        wm_obs_std = obs_std[list(ctrl_dims)].to(device)

    if mode == "context":
        from src.world_model.context_dynamics import ContextDynamicsModel
        from src.world_model.context_encoder import (
            ContextConditionedEncoder,
            ContextEncoder,
        )

        ctx_cfg = config.context
        context_encoder = ContextEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            context_dim=ctx_cfg.context_dim,
            hidden_dims=ctx_cfg.hidden_dims,
        )
        conditioned_encoder = ContextConditionedEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            context_dim=ctx_cfg.context_dim,
            n_atoms=config.dictionary.n_atoms,
            shared_hidden_dims=config.encoder.shared_hidden_dims,
            sparsity_method=config.encoder.sparsity_method,
            topk_k=config.encoder.topk_k,
            use_layernorm=config.encoder.use_layernorm,
        )
        return ContextDynamicsModel(
            dictionary=dictionary.to(device),
            context_encoder=context_encoder,
            conditioned_encoder=conditioned_encoder,
            learnable_dict=config.dictionary.slow_update_lr > 0,
            dim_weights=dim_weights,
            residual_hidden_dim=config.wm_loss.residual_hidden_dim,
            controllable_dims=ctrl_dims,
            diff_mean=wm_diff_mean,
            diff_std=wm_diff_std,
            obs_std=wm_obs_std,
        ).to(device)

    # mode == "dict"
    from src.world_model.sparse_encoder import SparseEncoder  # lazy: avoid circular

    encoder = SparseEncoder(
        state_dim=state_dim,
        action_dim=action_dim,
        n_atoms=config.dictionary.n_atoms,
        shared_hidden_dims=config.encoder.shared_hidden_dims,
        adapter_dim=config.encoder.adapter_dim,
        n_buildings=n_buildings,
        activation=config.encoder.activation,
        sparsity_method=config.encoder.sparsity_method,
        topk_k=config.encoder.topk_k,
        use_layernorm=config.encoder.use_layernorm,
        soft_topk_temperature=config.encoder.soft_topk_temperature,
    )
    return DictDynamicsModel(
        dictionary=dictionary.to(device),
        sparse_encoder=encoder,
        learnable_dict=config.dictionary.slow_update_lr > 0,
        dim_weights=dim_weights,
        residual_hidden_dim=config.wm_loss.residual_hidden_dim,
        controllable_dims=ctrl_dims,
        diff_mean=wm_diff_mean,
        diff_std=wm_diff_std,
        obs_std=wm_obs_std,
    ).to(device)


def build_trainer(
    model: BaseDictDynamics,
    config: TrainSchema,
) -> WorldModelTrainer:
    """Build a WorldModelTrainer with correct parameter groups.

    Auto-detects context encoder and sets separate learning rates.
    """
    return WorldModelTrainer(
        model=model,
        encoder_lr=config.dictionary.pretrain_lr,
        context_lr=config.context.context_lr,
        dict_lr=config.dictionary.slow_update_lr,
        sparsity_lambda=config.dictionary.sparsity_lambda,
        grad_clip_norm=config.wm_loss.grad_clip_norm,
        grad_clip_dict_norm=config.wm_loss.grad_clip_dict_norm,
        identity_penalty_lambda=config.wm_loss.identity_penalty_lambda,
        dim_weight_ema_decay=config.wm_loss.dim_weight_ema_decay,
        use_dim_weighting=config.wm_loss.use_dim_weighting,
        residual_lambda=config.wm_loss.residual_lambda,
    )
