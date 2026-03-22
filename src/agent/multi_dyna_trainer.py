"""Multi-building Dyna-SAC with shared dictionary (Phase 3)."""

import contextlib
import json
from pathlib import Path

import gymnasium
import numpy as np
import torch
from loguru import logger

from src.agent.dyna_sac import DynaSAC
from src.agent.replay_buffer import ReplayBuffer, TaggedReplayBuffer
from src.agent.rollout import ModelRollout
from src.schemas import TrainSchema
from src.utils import get_device, seed_everything, sinergym_workdir
from src.world_model.dict_dynamics import DictDynamicsModel
from src.world_model.model_trainer import WorldModelTrainer

with contextlib.suppress(ImportError):
    import sinergym  # noqa: F401


class MultiBuildingDynaSAC:
    """Multi-building Dyna-SAC with shared dictionary.

    Round-robin steps across N buildings. Shared dictionary D captures
    universal thermal dynamics; per-building adapters encode differences.

    Key experiment: shared D vs independent D (ablation).

    Args:
        building_configs: List of {env_name, building_id} dicts.
        dict_path: Pretrained dictionary path.
        config: TrainSchema configuration.
        seed: Random seed.
        save_dir: Output directory.
        independent_dict: If True, use per-building dictionaries (ablation).
    """

    def __init__(
        self,
        building_configs: list[dict],
        dict_path: str,
        config: TrainSchema,
        seed: int = 42,
        save_dir: str = "output/results/multi_dyna",
        independent_dict: bool = False,
    ) -> None:
        self.building_configs = building_configs
        self.n_buildings = len(building_configs)
        self.seed = seed
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.device = get_device(config.device)
        self.independent_dict = independent_dict

        seed_everything(seed)

        # Only create one env at a time (EnergyPlus crashes with multiple instances)
        self.building_ids = [cfg["building_id"] for cfg in building_configs]
        self._current_env: gymnasium.Env | None = None
        self._current_bid: str | None = None

        # Probe state/action dims from first env (create and close immediately)
        with sinergym_workdir():
            probe_env = gymnasium.make(building_configs[0]["env_name"])
        state_dim = probe_env.observation_space.shape[0]  # ty: ignore[not-subscriptable]
        action_dim = probe_env.action_space.shape[0]  # ty: ignore[not-subscriptable]
        action_low = probe_env.action_space.low  # ty: ignore[unresolved-attribute]
        action_high = probe_env.action_space.high  # ty: ignore[unresolved-attribute]
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        probe_env.close()

        # Load dictionary and normalization stats
        dict_data = torch.load(dict_path, weights_only=False)
        dictionary = dict_data["dictionary"]
        self._obs_mean = dict_data["obs_mean"].numpy()
        self._obs_std = dict_data["obs_std"].numpy()

        logger.info(
            f"Multi-building: {self.n_buildings} buildings, "
            f"state_dim={state_dim}, dict={dictionary.shape}, "
            f"independent_dict={independent_dict}"
        )

        # Build shared Dyna-SAC (all buildings share D, actor, critic)
        self.dyna = DynaSAC(
            state_dim=state_dim,
            action_dim=action_dim,
            building_ids=self.building_ids,
            dictionary=dictionary,
            config=config,
            action_scale=action_scale,
            action_bias=action_bias,
            obs_mean=dict_data["obs_mean"],
            obs_std=dict_data["obs_std"],
        )

        # Independent dict mode: per-building world models
        self._per_building_wm: dict[str, DictDynamicsModel] = {}
        self._per_building_trainer: dict[str, WorldModelTrainer] = {}
        self._per_building_rollout: dict[str, ModelRollout] = {}
        if independent_dict:
            self._setup_independent(dictionary, config, dict_data)

        # Shared tagged real buffer
        self.real_buffer = TaggedReplayBuffer(config.buffer_size, state_dim, action_dim)

        # Per-building model buffers (MBPO-style, small, cleared per episode)
        self.model_buffers: dict[str, ReplayBuffer] = {}
        model_cap = min(config.buffer_size, 10_000)
        for bid in self.building_ids:
            self.model_buffers[bid] = ReplayBuffer(model_cap, state_dim, action_dim)

    def _setup_independent(
        self, base_dict: torch.Tensor, config: TrainSchema, dict_data: dict
    ) -> None:
        """Create per-building world models with independent dictionaries."""
        from src.world_model.sparse_encoder import SparseEncoder

        state_dim = base_dict.shape[0]
        action_dim = self.dyna.action_dim

        for _bid_idx, bid in enumerate(self.building_ids):
            # Independent encoder (own trunk + adapter)
            encoder = SparseEncoder(
                state_dim=state_dim,
                action_dim=action_dim,
                n_atoms=config.dictionary.n_atoms,
                shared_hidden_dims=config.encoder.shared_hidden_dims,
                adapter_dim=config.encoder.adapter_dim,
                n_buildings=1,
                activation=config.encoder.activation,
                sparsity_method=config.encoder.sparsity_method,
                topk_k=config.encoder.topk_k,
            ).to(self.device)

            # Independent dictionary (random init, no cross-building pretrain)
            random_dict = torch.randn_like(base_dict)
            random_dict = random_dict / random_dict.norm(dim=0, keepdim=True)
            wm = DictDynamicsModel(
                dictionary=random_dict.to(self.device),
                sparse_encoder=encoder,
                learnable_dict=True,  # must learn from scratch
            ).to(self.device)

            trainer = WorldModelTrainer(
                model=wm,
                encoder_lr=config.dictionary.pretrain_lr,
                dict_lr=config.dictionary.slow_update_lr,
                sparsity_lambda=config.dictionary.sparsity_lambda,
            )

            rollout = ModelRollout(
                world_model=wm,
                actor=self.dyna.actor,  # shared actor
                reward_estimator=self.dyna.reward_estimator,
                exploration=self.dyna.exploration,
                device=self.device,
            )

            self._per_building_wm[bid] = wm
            self._per_building_trainer[bid] = trainer
            self._per_building_rollout[bid] = rollout

        logger.info(
            f"Independent mode: {self.n_buildings} separate world models "
            f"(shared actor/critic)"
        )

    def _get_env(self, bid_idx: int) -> gymnasium.Env:
        """Get env for building, creating if needed (single instance only)."""
        bid = self.building_ids[bid_idx]
        if self._current_bid != bid:
            # Close current env
            if self._current_env is not None:
                with contextlib.suppress(Exception):
                    self._current_env.close()
            # Create new env
            cfg = self.building_configs[bid_idx]
            with sinergym_workdir():
                self._current_env = gymnasium.make(cfg["env_name"])
            self._current_bid = bid
            logger.info(f"Switched to building: {bid}")
        return self._current_env  # ty: ignore[return-value]

    def _normalize_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        return np.clip((raw_obs - self._obs_mean) / self._obs_std, -10.0, 10.0).astype(
            np.float32
        )

    def train(self) -> dict:
        """Run multi-building training with sequential episodes.

        Each iteration runs one full episode for one building, then
        switches to the next. Only one EnergyPlus instance at a time.
        """
        n_episodes_per_building = self.config.total_timesteps // (
            35040 * self.n_buildings
        )
        n_episodes_per_building = max(n_episodes_per_building, 1)
        learning_starts = min(1000, self.config.batch_size * 2)

        episode_counts: dict[str, int] = dict.fromkeys(self.building_ids, 0)
        all_episode_rewards: dict[str, list[float]] = {
            bid: [] for bid in self.building_ids
        }
        eval_history: list[dict] = []
        global_step = 0

        logger.info(
            f"Sequential multi-building: {n_episodes_per_building} episodes/building, "
            f"{self.n_buildings} buildings"
        )

        for ep_round in range(n_episodes_per_building):
            for bid_idx, bid in enumerate(self.building_ids):
                logger.info(
                    f"Round {ep_round + 1}/{n_episodes_per_building}, Building: {bid}"
                )

                # Create env for this building (only one at a time)
                env = self._get_env(bid_idx)
                raw_obs, _ = env.reset(seed=self.seed + ep_round)
                obs = self._normalize_obs(raw_obs)
                episode_reward = 0.0
                done = False

                while not done:
                    global_step += 1

                    # Select action
                    if global_step < learning_starts:
                        action = env.action_space.sample()  # ty: ignore[unresolved-attribute]
                    else:
                        action = self.dyna.select_action(obs)

                    # Step
                    raw_next, reward, term, trunc, _ = env.step(action)
                    done = term or trunc
                    next_obs = self._normalize_obs(raw_next)

                    # Store in shared tagged buffer
                    self.real_buffer.add(
                        obs, action, float(reward), next_obs, done, tag=bid_idx
                    )

                    # Train
                    if global_step >= learning_starts:
                        self._train_step(
                            bid,
                            bid_idx,
                            obs,
                            action,
                            float(reward),
                            next_obs,
                            done,
                            global_step,
                        )

                    episode_reward += float(reward)
                    obs = next_obs

                # Episode finished
                episode_counts[bid] += 1
                all_episode_rewards[bid].append(episode_reward)
                logger.info(
                    f"[{bid}] Episode {episode_counts[bid]} | "
                    f"Step {global_step} | Reward: {episode_reward:.1f}"
                )

                # Clear model buffer for this building
                self.model_buffers[bid].pos = 0
                self.model_buffers[bid].size = 0
                self.dyna.exploration.apply_decay()

                # Save intermediate results
                self._save_results(all_episode_rewards, eval_history)

        # Close current env
        if self._current_env is not None:
            with contextlib.suppress(Exception):
                self._current_env.close()

        return {
            "episode_rewards": all_episode_rewards,
            "eval_history": eval_history,
        }

    def _train_step(
        self,
        bid: str,
        bid_idx: int,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        global_step: int,
    ) -> None:
        """Single training step for one building."""
        bid_str = str(bid_idx)
        batch_size = self.config.batch_size

        if self.real_buffer.tag_count(bid_idx) < batch_size:
            return

        # Select world model components (shared or independent)
        if self.independent_dict and bid in self._per_building_wm:
            wm_trainer = self._per_building_trainer[bid]
            rollout_gen = self._per_building_rollout[bid]
            wm_bid = "0"
        else:
            wm_trainer = self.dyna.wm_trainer
            rollout_gen = self.dyna.rollout_gen
            wm_bid = bid_str

        # 1. Update world model
        if global_step % self.config.dyna.model_update_freq == 0:
            if self.independent_dict:
                batch = self.real_buffer.sample_tagged(batch_size, bid_idx, self.device)
                self._wm_update(wm_trainer, batch, wm_bid)
            else:
                # Shared: train on ALL buildings (anti-forgetting)
                per_bld = max(batch_size // self.n_buildings, 32)
                for other_idx, _other_bid_name in enumerate(self.building_ids):
                    if self.real_buffer.tag_count(other_idx) < per_bld:
                        continue
                    batch = self.real_buffer.sample_tagged(
                        per_bld, other_idx, self.device
                    )
                    self._wm_update(wm_trainer, batch, str(other_idx))

        # 2. Model rollouts (after warmup)
        if global_step >= self.config.dyna.rollout_start_step:
            n_rollouts = self.config.dyna.rollouts_per_step
            horizon = self.config.dyna.rollout_horizon
            start_batch = self.real_buffer.sample_tagged(
                n_rollouts, bid_idx, self.device
            )
            start_states = start_batch["states"].cpu().numpy()
            rollout_data = rollout_gen.generate(start_states, wm_bid, horizon)
            self.model_buffers[bid].add_batch(
                rollout_data["states"],
                rollout_data["actions"],
                rollout_data["rewards"],
                rollout_data["next_states"],
                rollout_data["dones"],
            )

        # 3. SAC update (real from ALL buildings + model from THIS)
        real_batch = self.real_buffer.sample(batch_size, self.device)
        model_ratio = self.config.dyna.model_to_real_ratio
        n_model = int(batch_size * model_ratio)

        if n_model > 0 and len(self.model_buffers[bid]) >= n_model:
            model_batch = self.model_buffers[bid].sample(n_model, self.device)
            n_real = batch_size - n_model
            mixed = {
                k: torch.cat([real_batch[k][:n_real], model_batch[k]], dim=0)
                for k in real_batch
            }
        else:
            mixed = real_batch

        self.dyna.sac_trainer.update(
            mixed["states"],
            mixed["actions"],
            mixed["rewards"],
            mixed["next_states"],
            mixed["dones"],
        )

    def _wm_update(
        self,
        wm_trainer: WorldModelTrainer,
        batch: dict[str, torch.Tensor],
        bid_str: str,
    ) -> None:
        """World model update with TD-error weighted loss."""
        with torch.no_grad():
            next_actions, _ = self.dyna.actor(batch["next_states"])
            q1_next, q2_next = self.dyna.critic_target(
                batch["next_states"], next_actions
            )
            q_next = torch.min(q1_next, q2_next)
            target_q = (
                batch["rewards"] + (1.0 - batch["dones"]) * self.config.gamma * q_next
            )
            q1, q2 = self.dyna.critic(batch["states"], batch["actions"])
            td_error = (torch.min(q1, q2) - target_q).abs().squeeze(-1)
            weights = 1.0 + td_error / (td_error.mean() + 1e-8)
        wm_trainer.train_step(
            batch["states"],
            batch["actions"],
            batch["next_states"],
            bid_str,
            weights,
        )

    def _save_results(
        self,
        episode_rewards: dict[str, list[float]],
        eval_history: list[dict],
    ) -> None:
        self.dyna.save(self.save_dir / "final_checkpoint.pt")
        for bid, rewards in episode_rewards.items():
            np.save(self.save_dir / f"{bid}_rewards.npy", np.array(rewards))
        with open(self.save_dir / "eval_history.json", "w") as f:
            json.dump(eval_history, f, indent=2)
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(
                {
                    "buildings": [c["building_id"] for c in self.building_configs],
                    "independent_dict": self.independent_dict,
                    "total_timesteps": self.config.total_timesteps,
                },
                f,
                indent=2,
            )
        logger.info(f"Results saved to {self.save_dir}")
