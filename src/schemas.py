"""Pydantic v2 configuration schemas for DictDyna."""

from pydantic import BaseModel, ConfigDict, Field


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


class ContextEncoderSchema(BaseModel):
    """Context encoder configuration for context-conditioned world model."""

    model_config = ConfigDict(frozen=True)

    context_dim: int = Field(16, ge=4, le=64, description="Context vector dimension")
    context_window: int = Field(
        10, ge=1, le=50, description="Number of recent transitions for context"
    )
    hidden_dims: list[int] = Field(default=[128, 128])
    context_lr: float = Field(1e-3, gt=0, description="Context encoder learning rate")


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
