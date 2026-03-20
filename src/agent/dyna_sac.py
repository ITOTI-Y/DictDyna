"""Dyna-SAC: SAC with dictionary world model rollouts."""

from pathlib import Path

import numpy as np
import torch
from loguru import logger

from src.agent.replay_buffer import MixedReplayBuffer
from src.agent.rollout import ModelRollout
from src.agent.sac import GaussianActor, SACTrainer, SoftQNetwork
from src.schemas import TrainSchema
from src.utils import get_device
from src.world_model.dict_dynamics import DictDynamicsModel
from src.world_model.model_trainer import WorldModelTrainer
from src.world_model.reward_estimator import SinergymRewardEstimator
from src.world_model.sparse_encoder import SparseEncoder


class DynaSAC:
    """Dyna-style SAC with dictionary world model.

    At each real step:
    1. Execute action in env, collect (s, a, r, s')
    2. Update world model on real data
    3. Generate M simulated rollouts of horizon H
    4. Update SAC policy on mixed real + simulated data

    Args:
        state_dim: Observation dimension.
        action_dim: Action dimension.
        building_ids: List of building identifiers.
        dictionary: Pretrained dictionary tensor, shape (d, K).
        config: TrainSchema configuration.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        building_ids: list[str],
        dictionary: torch.Tensor,
        config: TrainSchema,
        action_scale: float | np.ndarray = 1.0,
        action_bias: float | np.ndarray = 0.0,
        obs_mean: torch.Tensor | None = None,
        obs_std: torch.Tensor | None = None,
    ) -> None:
        self.config = config
        self.device = get_device(config.device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.building_ids = building_ids

        # Build sparse encoder
        self.encoder = SparseEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            n_atoms=config.dictionary.n_atoms,
            shared_hidden_dims=config.encoder.shared_hidden_dims,
            adapter_dim=config.encoder.adapter_dim,
            n_buildings=len(building_ids),
            activation=config.encoder.activation,
            sparsity_method=config.encoder.sparsity_method,
            topk_k=config.encoder.topk_k,
        ).to(self.device)

        # Build world model (normalized obs space, no conversion needed)
        self.world_model = DictDynamicsModel(
            dictionary=dictionary.to(self.device),
            sparse_encoder=self.encoder,
            learnable_dict=config.dictionary.slow_update_lr > 0,
        ).to(self.device)

        # Build reward estimator (denormalizes predicted states for reward calc)
        self.reward_estimator = SinergymRewardEstimator(
            obs_mean=obs_mean,
            obs_std=obs_std,
        )

        # Build SAC components (no internal ObsNormLayer - inputs already normalized)
        self.actor = GaussianActor(
            state_dim,
            action_dim,
            hidden_dims=config.sac.hidden_dims,
            action_scale=action_scale,
            action_bias=action_bias,
        ).to(self.device)

        self.critic = SoftQNetwork(
            state_dim,
            action_dim,
            hidden_dims=config.sac.hidden_dims,
        ).to(self.device)

        self.critic_target = SoftQNetwork(
            state_dim,
            action_dim,
            hidden_dims=config.sac.hidden_dims,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Build trainers
        self.sac_trainer = SACTrainer(
            actor=self.actor,
            critic=self.critic,
            critic_target=self.critic_target,
            actor_lr=config.sac.actor_lr,
            critic_lr=config.sac.critic_lr,
            alpha_lr=config.sac.alpha_lr,
            tau=config.sac.tau,
            gamma=config.gamma,
            autotune_alpha=config.sac.autotune_alpha,
            initial_alpha=config.sac.initial_alpha,
            target_entropy=-action_dim,
            device=self.device,
        )

        self.wm_trainer = WorldModelTrainer(
            model=self.world_model,
            encoder_lr=config.dictionary.pretrain_lr,
            dict_lr=config.dictionary.slow_update_lr,
            sparsity_lambda=config.dictionary.sparsity_lambda,
        )

        # Build rollout generator
        self.rollout_gen = ModelRollout(
            world_model=self.world_model,
            actor=self.actor,
            reward_estimator=self.reward_estimator,
            device=self.device,
        )

        # Build replay buffer (small model buffer for freshness, MBPO-style)
        model_capacity = min(config.buffer_size, 10_000)
        self.buffer = MixedReplayBuffer(
            real_capacity=config.buffer_size,
            model_capacity=model_capacity,
            state_dim=state_dim,
            action_dim=action_dim,
        )

        # Building id to index mapping
        self._bid_to_idx = {bid: str(i) for i, bid in enumerate(building_ids)}

    def train_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        building_id: str,
        global_step: int,
    ) -> dict[str, float]:
        """Single Dyna-SAC training step.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode ended.
            building_id: Building identifier.
            global_step: Current global timestep.

        Returns:
            Training metrics.
        """
        metrics: dict[str, float] = {}
        bid_idx = self._bid_to_idx[building_id]

        # 1. Store real transition
        self.buffer.add_real(state, action, reward, next_state, done)

        if self.buffer.real_size < self.config.batch_size:
            return metrics

        # 2. Update world model (always, using real data only)
        if global_step % self.config.dyna.model_update_freq == 0:
            batch = self.buffer.real_buffer.sample(self.config.batch_size, self.device)
            wm_metrics = self.wm_trainer.train_step(
                batch["states"],
                batch["actions"],
                batch["next_states"],
                bid_idx,
            )
            metrics.update({f"wm/{k}": v for k, v in wm_metrics.items()})

        # 3. Model rollouts (only after warmup period)
        use_model = global_step >= self.config.dyna.rollout_start_step
        if use_model:
            n_rollouts = self.config.dyna.rollouts_per_step
            horizon = self.config.dyna.rollout_horizon
            start_batch = self.buffer.real_buffer.sample(n_rollouts, self.device)
            start_states = start_batch["states"].cpu().numpy()

            rollout_data = self.rollout_gen.generate(start_states, bid_idx, horizon)
            self.buffer.add_model_batch(
                rollout_data["states"],
                rollout_data["actions"],
                rollout_data["rewards"],
                rollout_data["next_states"],
                rollout_data["dones"],
            )

            # Diagnostics: rollout data quality
            metrics["diag/model_reward_mean"] = float(rollout_data["rewards"].mean())
            metrics["diag/model_reward_std"] = float(rollout_data["rewards"].std())
            metrics["diag/model_state_std"] = float(rollout_data["next_states"].std())
            metrics["diag/real_reward"] = reward
            metrics["diag/model_buf_size"] = float(self.buffer.model_size)

        # 4. SAC update (pure real data during warmup, mixed after)
        model_ratio = self.config.dyna.model_to_real_ratio if use_model else 0.0
        mixed_batch = self.buffer.sample(
            self.config.batch_size,
            model_ratio=model_ratio,
            device=self.device,
        )
        sac_metrics = self.sac_trainer.update(
            mixed_batch["states"],
            mixed_batch["actions"],
            mixed_batch["rewards"],
            mixed_batch["next_states"],
            mixed_batch["dones"],
        )
        metrics.update({f"sac/{k}": v for k, v in sac_metrics.items()})

        # Diagnostics: SAC internals
        metrics["diag/alpha"] = sac_metrics.get("alpha", 0.0)
        metrics["diag/real_buf_size"] = float(self.buffer.real_size)

        return metrics

    def on_episode_end(self) -> None:
        """Clear model buffer at episode boundary (MBPO-style freshness)."""
        self.buffer.clear_model_buffer()

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Select action for a given state."""
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                action = self.actor.get_action(state_t)
            else:
                action, _ = self.actor(state_t)

        return action.cpu().numpy().squeeze(0)

    def save(self, path: str | Path) -> None:
        """Save all model checkpoints."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "world_model": self.world_model.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "config": self.config.model_dump(),
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")

    def load(self, path: str | Path) -> None:
        """Load model checkpoints."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        logger.info(f"Loaded checkpoint from {path}")
