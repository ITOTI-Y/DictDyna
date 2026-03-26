"""Dyna-SAC: SAC with dictionary world model rollouts."""

from pathlib import Path

import numpy as np
import torch
from loguru import logger

from src.agent.replay_buffer import MixedReplayBuffer
from src.agent.rollout import ModelRollout
from src.agent.sac import GaussianActor, SACTrainer, SoftQNetwork
from src.agent.sparse_exploration import SparseCodeExploration
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

        # Soft top-k temperature annealing config
        self._soft_topk_start = config.encoder.soft_topk_temperature
        self._soft_topk_anneal_steps = config.encoder.soft_topk_anneal_steps

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
            use_layernorm=config.encoder.use_layernorm,
            soft_topk_temperature=config.encoder.soft_topk_temperature,
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
            grad_clip_norm=config.wm_loss.grad_clip_norm,
            grad_clip_dict_norm=config.wm_loss.grad_clip_dict_norm,
            identity_penalty_lambda=config.wm_loss.identity_penalty_lambda,
            dim_weight_ema_decay=config.wm_loss.dim_weight_ema_decay,
            use_dim_weighting=config.wm_loss.use_dim_weighting,
        )

        # Build sparse-code exploration module
        self.exploration = SparseCodeExploration(eta=0.1)

        # Build rollout generator (with exploration bonus)
        self.rollout_gen = ModelRollout(
            world_model=self.world_model,
            actor=self.actor,
            reward_estimator=self.reward_estimator,
            exploration=self.exploration,
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

        # Model quality tracking for adaptive horizon
        self._model_quality_ema = 0.0
        self._model_quality_decay = 0.99

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

        # Anneal soft top-k temperature
        if self._soft_topk_start > 0 and self._soft_topk_anneal_steps > 0:
            progress = min(global_step / self._soft_topk_anneal_steps, 1.0)
            temp = self._soft_topk_start * (1.0 - progress) + 0.01 * progress
            self.encoder.soft_topk_temperature = temp
            metrics["diag/soft_topk_temp"] = temp

        # 1. Store real transition
        self.buffer.add_real(state, action, reward, next_state, done)

        if self.buffer.real_size < self.config.batch_size:
            return metrics

        # 2. Update world model (reward-weighted: high TD-error → higher weight)
        if global_step % self.config.dyna.model_update_freq == 0:
            batch = self.buffer.real_buffer.sample(self.config.batch_size, self.device)

            # Compute TD-error based sample weights
            with torch.no_grad():
                next_actions, _next_log_probs = self.actor(batch["next_states"])
                q1_next, q2_next = self.critic_target(
                    batch["next_states"], next_actions
                )
                q_next = torch.min(q1_next, q2_next)
                target_q = (
                    batch["rewards"]
                    + (1.0 - batch["dones"]) * self.config.gamma * q_next
                )
                q1, q2 = self.critic(batch["states"], batch["actions"])
                td_error = (torch.min(q1, q2) - target_q).abs().squeeze(-1)
                # Normalize to [1, 1+beta] range
                weights = 1.0 + td_error / (td_error.mean() + 1e-8)

            wm_metrics = self.wm_trainer.train_step(
                batch["states"],
                batch["actions"],
                batch["next_states"],
                bid_idx,
                sample_weights=weights,
            )
            metrics.update({f"wm/{k}": v for k, v in wm_metrics.items()})

            # Multi-step consistency training with curriculum + scheduled sampling
            dyna_cfg = self.config.dyna
            if dyna_cfg.multistep_curriculum:
                # Curriculum: horizon increases with training progress
                ms_horizon = dyna_cfg.multistep_curriculum_schedule[0]
                for h_val, step_thresh in zip(
                    dyna_cfg.multistep_curriculum_schedule,
                    dyna_cfg.multistep_curriculum_steps,
                    strict=False,
                ):
                    if global_step >= step_thresh:
                        ms_horizon = h_val
            else:
                ms_horizon = dyna_cfg.multistep_horizon

            if ms_horizon > 1 and self.buffer.real_size > ms_horizon + 1:
                # Scheduled sampling: teacher forcing decays over training
                if dyna_cfg.scheduled_sampling_steps > 0:
                    ss_progress = min(
                        global_step / dyna_cfg.scheduled_sampling_steps, 1.0
                    )
                    tf_ratio = (
                        dyna_cfg.scheduled_sampling_start * (1 - ss_progress)
                        + dyna_cfg.scheduled_sampling_end * ss_progress
                    )
                else:
                    tf_ratio = dyna_cfg.teacher_forcing_ratio

                seq_batch = self.buffer.real_buffer.sample_sequence(
                    self.config.batch_size, ms_horizon, self.device
                )
                ms_metrics = self.wm_trainer.train_multistep(
                    seq_batch["states"],
                    seq_batch["actions"],
                    seq_batch["next_states"],
                    building_id=bid_idx,
                    discount=dyna_cfg.multistep_discount,
                    teacher_forcing_ratio=tf_ratio,
                    dones_seq=seq_batch["dones"],
                )
                ms_metrics["multistep_tf_ratio"] = tf_ratio
                metrics.update({f"wm/{k}": v for k, v in ms_metrics.items()})

        # 3. Model rollouts (only after warmup period)
        use_model = global_step >= self.config.dyna.rollout_start_step
        if use_model:
            # Compute model quality: how much better than identity mapping
            if "wm/mse_loss" in metrics and "wm/identity_penalty" in metrics:
                wm_mse = metrics["wm/mse_loss"]
                id_pen = metrics["wm/identity_penalty"]
                # quality ∈ [0, 1]: 0 = worse than identity, 1 = much better
                quality = max(0.0, 1.0 - id_pen / (wm_mse + 1e-8))
                self._model_quality_ema = (
                    self._model_quality_decay * self._model_quality_ema
                    + (1 - self._model_quality_decay) * quality
                )
                metrics["diag/model_quality"] = self._model_quality_ema

            # Adaptive horizon: scale by model quality
            max_horizon = self.config.dyna.rollout_horizon
            if max_horizon > 1 and self._model_quality_ema > 0:
                horizon = max(1, int(self._model_quality_ema * max_horizon))
            else:
                horizon = max_horizon
            metrics["diag/effective_horizon"] = float(horizon)

            n_rollouts = self.config.dyna.rollouts_per_step
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
            metrics["diag/explore_n_patterns"] = float(
                self.exploration.n_unique_patterns
            )

        # 4. SAC update (with optional MVE targets)
        model_ratio = self.config.dyna.model_to_real_ratio if use_model else 0.0
        if self.config.dyna.use_mve:
            # MVE mode: use only real data, extend value with model rollouts
            mixed_batch = self.buffer.real_buffer.sample(
                self.config.batch_size, self.device
            )
        else:
            mixed_batch = self.buffer.sample(
                self.config.batch_size,
                model_ratio=model_ratio,
                device=self.device,
            )

        mve_targets = None
        if use_model and self.config.dyna.use_mve:
            mve_targets = self.rollout_gen.compute_mve_targets(
                states=mixed_batch["states"],
                actions=mixed_batch["actions"],
                rewards=mixed_batch["rewards"],
                next_states=mixed_batch["next_states"],
                critic_target=self.critic_target,
                gamma=self.config.gamma,
                alpha=self.sac_trainer.alpha.item(),
                building_id=bid_idx,
                horizon=self.config.dyna.mve_horizon,
            )

        sac_metrics = self.sac_trainer.update(
            mixed_batch["states"],
            mixed_batch["actions"],
            mixed_batch["rewards"],
            mixed_batch["next_states"],
            mixed_batch["dones"],
            mve_target_q=mve_targets,
        )
        metrics.update({f"sac/{k}": v for k, v in sac_metrics.items()})

        # Diagnostics: SAC internals
        metrics["diag/alpha"] = sac_metrics.get("alpha", 0.0)
        metrics["diag/real_buf_size"] = float(self.buffer.real_size)

        return metrics

    def on_episode_end(self) -> None:
        """Clear model buffer and decay exploration counts at episode boundary."""
        self.buffer.clear_model_buffer()
        self.exploration.apply_decay()

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
