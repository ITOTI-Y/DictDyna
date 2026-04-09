"""Pydantic v2 configuration schemas for DictDyna."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.obs_config import OBS_CONFIG


class DictionarySchema(BaseModel):
    """Dictionary learning configuration."""

    model_config = ConfigDict(frozen=True)

    n_atoms: int = Field(128, ge=16, description="Number of dictionary atoms K")
    state_dim: int = Field(15, ge=1, description="State dimension d (Sinergym ~10-15)")
    sparsity_lambda: float = Field(0.1, gt=0, description="L1 sparsity weight lambda")
    pretrain_epochs: int = Field(100, ge=1)
    pretrain_lr: float = Field(1e-3, gt=0)
    slow_update_lr: float = Field(
        1e-5, ge=0, description="D fine-tune rate during RL, 0=fixed"
    )
    reward_dim_weight: float = Field(
        5.0, ge=1.0, description="Weight multiplier for reward-critical dimensions"
    )
    reward_dims: list[int] = Field(
        default=[OBS_CONFIG.AIR_TEMPERATURE, OBS_CONFIG.HVAC_POWER],
        description="Reward-critical state indices (air_temperature, HVAC_power)",
    )
    controllable_only: bool = Field(
        False,
        description="[EXPERIMENTAL - known shape mismatch bug, do not use] "
        "WM predicts only controllable dims (4) instead of all (17)",
    )


class SparseEncoderSchema(BaseModel):
    """Sparse encoder g_theta(s,a;phi_i) configuration."""

    model_config = ConfigDict(frozen=True)

    shared_hidden_dims: list[int] = Field(default=[256, 256])
    adapter_dim: int = Field(64, ge=1, description="Per-building adapter hidden dim")
    activation: str = Field("relu", pattern=r"^(relu|gelu|tanh)$")
    sparsity_method: str = Field("topk", pattern=r"^(l1_penalty|topk|proximal)$")
    topk_k: int = Field(
        16, ge=1, description="If sparsity_method=topk, keep top-k activations"
    )
    use_layernorm: bool = Field(
        False, description="Add LayerNorm to shared trunk hidden layers"
    )
    soft_topk_temperature: float = Field(
        0.0,
        ge=0,
        description="Soft top-k temperature (0=hard, >0=soft during training)",
    )
    soft_topk_anneal_steps: int = Field(
        50000, ge=0, description="Steps to anneal soft top-k temp to near-zero"
    )


class ContextEncoderSchema(BaseModel):
    """Context encoder configuration for context-conditioned world model."""

    model_config = ConfigDict(frozen=True)

    context_dim: int = Field(16, ge=4, le=64, description="Context vector dimension")
    context_window: int = Field(
        10, ge=1, le=50, description="Number of recent transitions for context"
    )
    hidden_dims: list[int] = Field(default=[128, 128])
    context_lr: float = Field(1e-3, gt=0, description="Context encoder learning rate")
    use_context_gating: bool = Field(
        False,
        description="Enable context-to-sparse gating: context z generates atom-level "
        "gates that modulate sparse codes before top-k selection",
    )


class WorldModelLossSchema(BaseModel):
    """World model loss and training stabilization configuration."""

    model_config = ConfigDict(frozen=True)

    use_dim_weighting: bool = Field(
        True, description="Enable per-dimension adaptive loss weighting"
    )
    dim_weight_ema_decay: float = Field(
        0.99, ge=0.9, le=1.0, description="EMA decay for dimension weights"
    )
    identity_penalty_lambda: float = Field(
        0.5, ge=0, description="Identity guard penalty weight (0=disabled)"
    )
    grad_clip_norm: float = Field(
        1.0, gt=0, description="Gradient clip norm for encoder"
    )
    grad_clip_dict_norm: float = Field(
        0.1, gt=0, description="Gradient clip norm for dictionary"
    )
    residual_hidden_dim: int = Field(
        128, ge=0, description="Residual correction head hidden dim (0=disabled)"
    )
    residual_lambda: float = Field(
        0.01, ge=0, description="L2 regularization on residual output (0=disabled)"
    )


class DynaSchema(BaseModel):
    """Dyna-style planning configuration."""

    model_config = ConfigDict(frozen=True)

    rollout_horizon: int = Field(1, ge=1, le=10, description="H: model rollout steps")
    rollouts_per_step: int = Field(
        10, ge=1, description="M: number of rollouts per real step"
    )
    model_to_real_ratio: float = Field(
        0.2, ge=0, le=1, description="Fraction of simulated data in batch"
    )
    rollout_start_step: int = Field(
        5000, ge=0, description="Start model rollouts after this many real steps"
    )
    model_update_freq: int = Field(
        1, ge=1, description="Update world model every N real steps"
    )
    multistep_horizon: int = Field(
        1, ge=1, le=10, description="Multi-step WM training horizon (1=single-step)"
    )
    multistep_discount: float = Field(
        0.95, ge=0, le=1, description="Discount for multi-step error weighting"
    )
    teacher_forcing_ratio: float = Field(
        0.5, ge=0, le=1, description="Teacher forcing ratio for multi-step training"
    )
    use_mve: bool = Field(
        False, description="Use Model-Based Value Expansion instead of Dyna rollouts"
    )
    mve_horizon: int = Field(
        3, ge=1, le=10, description="MVE horizon for value expansion"
    )


class SACSchema(BaseModel):
    """CleanRL-style SAC configuration."""

    model_config = ConfigDict(frozen=True)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    hidden_dims: list[int] = Field(default=[256, 256])
    tau: float = 0.005
    autotune_alpha: bool = True
    initial_alpha: float = 0.2


class RewardSchema(BaseModel):
    """Reward estimation configuration."""

    model_config = ConfigDict(frozen=True)

    comfort_weight: float = 0.5
    temp_target: float = 23.0
    temp_band: float = 2.0


class WandbSchema(BaseModel):
    """Weights & Biases configuration."""

    model_config = ConfigDict(frozen=True)

    project: str = "dictdyna"
    entity: str | None = None
    tags: list[str] = Field(default=[])


class TrainSchema(BaseModel):
    """Overall training configuration."""

    model_config = ConfigDict(frozen=True)

    mode: Literal["context", "dict"] = Field(
        "context",
        description="World model mode: 'context' (recommended) or 'dict' (adapter-based fallback)",
    )
    seed: int = 42
    total_timesteps: int = Field(
        35040 * 3, description="3 episodes (Sinergym: 35040 steps/year at 15-min)"
    )
    eval_freq: int = 35040
    n_eval_episodes: int = 1
    log_interval: int = 1000
    save_freq: int = 35040
    n_buildings: int = Field(3, ge=1)
    batch_size: int = 256
    buffer_size: int = 100_000
    gamma: float = 0.99
    device: str = "auto"

    dictionary: DictionarySchema = DictionarySchema()
    encoder: SparseEncoderSchema = SparseEncoderSchema()
    context: ContextEncoderSchema = ContextEncoderSchema()
    wm_loss: WorldModelLossSchema = WorldModelLossSchema()
    dyna: DynaSchema = DynaSchema()
    sac: SACSchema = SACSchema()
    reward: RewardSchema = RewardSchema()
    wandb: WandbSchema = WandbSchema()


class TransferSchema(BaseModel):
    """Few-shot transfer configuration."""

    model_config = ConfigDict(frozen=True)

    adaptation_days: int = Field(7, ge=1, description="Days of data for new building")
    freeze_dictionary: bool = True
    freeze_shared_encoder: bool = True
    adapter_lr: float = 1e-3
    adaptation_epochs: int = 50
