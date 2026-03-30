"""Dyna-SAC: SAC with dictionary world model rollouts."""

from collections import deque
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from src.agent._share import compute_td_error_weights
from src.agent.replay_buffer import MixedReplayBuffer
from src.agent.rollout import ModelRollout
from src.agent.sac import GaussianActor, SACTrainer, SoftQNetwork
from src.agent.sparse_exploration import SparseCodeExploration
from src.schemas import TrainSchema
from src.utils import get_device
from src.world_model.factory import build_trainer, build_world_model
from src.world_model.reward_estimator import SinergymRewardEstimator


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
        diff_mean: torch.Tensor | None = None,
        diff_std: torch.Tensor | None = None,
    ) -> None:
        self.config = config
        self.device = get_device(config.device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.building_ids = building_ids

        # Soft top-k temperature annealing config (dict mode only)
        self._soft_topk_start = config.encoder.soft_topk_temperature
        self._soft_topk_anneal_steps = config.encoder.soft_topk_anneal_steps

        # Build world model via factory (respects config.mode)
        self.world_model = build_world_model(
            dictionary=dictionary,
            state_dim=state_dim,
            action_dim=action_dim,
            config=config,
            device=self.device,
            n_buildings=len(building_ids),
            diff_mean=diff_mean,
            diff_std=diff_std,
            obs_std=obs_std,
        )
        # Expose encoder for soft-topk annealing (dict mode only)
        self.encoder = self.world_model.encoder

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

        self.wm_trainer = build_trainer(self.world_model, config)

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

        # Context mode: maintain per-building transition window
        self._is_context_mode = config.mode == "context"
        self._context_window: deque = deque(maxlen=config.context.context_window)
        self._current_context: torch.Tensor | None = None

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

        # Anneal soft top-k temperature (dict mode only)
        if (
            not self._is_context_mode
            and self._soft_topk_start > 0
            and self._soft_topk_anneal_steps > 0
        ):
            progress = min(global_step / self._soft_topk_anneal_steps, 1.0)
            temp = self._soft_topk_start * (1.0 - progress) + 0.01 * progress
            self.encoder.soft_topk_temperature = temp  # ty: ignore[invalid-assignment]
            metrics["diag/soft_topk_temp"] = temp

        # Update context window (context mode)
        if self._is_context_mode:
            delta = next_state - state
            transition = torch.tensor(
                np.concatenate([state, action, delta]), dtype=torch.float32
            )
            self._context_window.append(transition)

        # 1. Store real transition
        self.buffer.add_real(state, action, reward, next_state, done)

        if self.buffer.real_size < self.config.batch_size:
            return metrics

        # Build WM routing kwargs (building_id or context)
        wm_kwargs = self._get_wm_kwargs(bid_idx)

        # 2. Update world model (reward-weighted: high TD-error → higher weight)
        if global_step % self.config.dyna.model_update_freq == 0:
            batch = self.buffer.real_buffer.sample(self.config.batch_size, self.device)

            # Compute TD-error based sample weights
            weights = compute_td_error_weights(
                self.actor,
                self.critic,
                self.critic_target,
                batch,
                self.config.gamma,
            )

            # Expand context to batch size if needed
            train_kwargs = dict(wm_kwargs)
            if "context" in train_kwargs:
                train_kwargs["context"] = train_kwargs["context"].expand(  # ty: ignore[unresolved-attribute]
                    self.config.batch_size, -1
                )

            wm_metrics = self.wm_trainer.train_step(
                batch["states"],
                batch["actions"],
                batch["next_states"],
                sample_weights=weights,
                **train_kwargs,
            )
            metrics.update({f"wm/{k}": v for k, v in wm_metrics.items()})

            # Multi-step consistency training (if horizon > 1 and enough data)
            ms_horizon = self.config.dyna.multistep_horizon
            if ms_horizon > 1 and self.buffer.real_size > ms_horizon + 1:
                seq_batch = self.buffer.real_buffer.sample_sequence(
                    self.config.batch_size, ms_horizon, self.device
                )
                ms_metrics = self.wm_trainer.train_multistep(
                    seq_batch["states"],
                    seq_batch["actions"],
                    seq_batch["next_states"],
                    discount=self.config.dyna.multistep_discount,
                    teacher_forcing_ratio=self.config.dyna.teacher_forcing_ratio,
                    **train_kwargs,
                )
                metrics.update({f"wm/{k}": v for k, v in ms_metrics.items()})

        # 3. Model rollouts (only after warmup period)
        use_model = global_step >= self.config.dyna.rollout_start_step
        if use_model:
            # Compute model quality: how much better than identity mapping
            if "wm/mse" in metrics and "wm/identity_penalty" in metrics:
                wm_mse = metrics["wm/mse"]
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

            rollout_data = self.rollout_gen.generate(
                start_states,
                bid_idx,
                horizon,
                context=wm_kwargs.get("context") if self._is_context_mode else None,  # ty: ignore[invalid-argument-type]
            )
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

    def _get_wm_kwargs(self, bid_idx: str) -> dict[str, object]:
        """Get world model routing kwargs (building_id or context)."""
        if not self._is_context_mode:
            return {"building_id": bid_idx}

        # Infer context from transition window
        if len(self._context_window) == 0:
            ctx_dim = self.config.context.context_dim
            self._current_context = torch.zeros(1, ctx_dim, device=self.device)
        else:
            window_size = self.config.context.context_window
            transitions = torch.stack(list(self._context_window)).unsqueeze(0)
            transitions = transitions.to(self.device)
            if transitions.shape[1] < window_size:
                pad = torch.zeros(
                    1,
                    window_size - transitions.shape[1],
                    transitions.shape[2],
                    device=self.device,
                )
                transitions = torch.cat([pad, transitions], dim=1)
            with torch.no_grad():
                self._current_context = self.world_model.infer_context(transitions)  # type: ignore[union-attr]  # ty: ignore[call-non-callable]
        return {"context": self._current_context}

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
