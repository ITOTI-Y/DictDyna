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
from src.schemas import TrainSchema
from src.utils import get_device, seed_everything, sinergym_workdir

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

        # Create environments
        self.envs: dict[str, gymnasium.Env] = {}
        self.building_ids: list[str] = []
        for cfg in building_configs:
            bid = cfg["building_id"]
            with sinergym_workdir():
                self.envs[bid] = gymnasium.make(cfg["env_name"])
            self.building_ids.append(bid)

        env0 = next(iter(self.envs.values()))
        state_dim = env0.observation_space.shape[0]  # ty: ignore[not-subscriptable]
        action_dim = env0.action_space.shape[0]  # ty: ignore[not-subscriptable]
        action_low = env0.action_space.low  # ty: ignore[unresolved-attribute]
        action_high = env0.action_space.high  # ty: ignore[unresolved-attribute]
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0

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

        # If independent dict, create separate dictionaries per building
        if independent_dict:
            self._setup_independent_dicts(dictionary)

        # Shared tagged real buffer
        self.real_buffer = TaggedReplayBuffer(config.buffer_size, state_dim, action_dim)

        # Per-building model buffers (MBPO-style, small, cleared per episode)
        self.model_buffers: dict[str, ReplayBuffer] = {}
        model_cap = min(config.buffer_size, 10_000)
        for bid in self.building_ids:
            self.model_buffers[bid] = ReplayBuffer(model_cap, state_dim, action_dim)

    def _setup_independent_dicts(self, base_dict: torch.Tensor) -> None:
        """Create per-building dictionary copies (ablation mode)."""
        # Store independent dicts as separate parameters
        self._independent_dicts = {}
        for bid in self.building_ids:
            d = base_dict.clone().to(self.device)
            self._independent_dicts[bid] = torch.nn.Parameter(d)
        logger.info(f"Independent dict mode: {self.n_buildings} separate dictionaries")

    def _normalize_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        return np.clip((raw_obs - self._obs_mean) / self._obs_std, -10.0, 10.0).astype(
            np.float32
        )

    def train(self) -> dict:
        """Run multi-building training with round-robin stepping."""
        total_steps = self.config.total_timesteps
        eval_freq = self.config.eval_freq
        learning_starts = min(1000, self.config.batch_size * 2)

        # Per-building state
        obs: dict[str, np.ndarray] = {}
        episode_rewards: dict[str, float] = {}
        episode_counts: dict[str, int] = {}
        all_episode_rewards: dict[str, list[float]] = {}

        for bid in self.building_ids:
            raw_obs, _ = self.envs[bid].reset(seed=self.seed)
            obs[bid] = self._normalize_obs(raw_obs)
            episode_rewards[bid] = 0.0
            episode_counts[bid] = 0
            all_episode_rewards[bid] = []

        eval_history: list[dict] = []
        global_step = 0

        logger.info(
            f"Starting multi-building training: {total_steps} steps, "
            f"{self.n_buildings} buildings"
        )

        while global_step < total_steps:
            for bid_idx, bid in enumerate(self.building_ids):
                global_step += 1
                if global_step > total_steps:
                    break

                # Select action
                if global_step < learning_starts:
                    action = self.envs[bid].action_space.sample()  # ty: ignore[unresolved-attribute]
                else:
                    action = self.dyna.select_action(obs[bid])

                # Step environment
                raw_next, reward, term, trunc, _ = self.envs[bid].step(action)
                done = term or trunc
                next_obs = self._normalize_obs(raw_next)

                # Store in shared tagged buffer
                self.real_buffer.add(
                    obs[bid], action, float(reward), next_obs, done, tag=bid_idx
                )

                # Train step
                if global_step >= learning_starts:
                    self._train_step(
                        bid,
                        bid_idx,
                        obs[bid],
                        action,
                        float(reward),
                        next_obs,
                        done,
                        global_step,
                    )

                episode_rewards[bid] += float(reward)
                obs[bid] = next_obs

                if done:
                    episode_counts[bid] += 1
                    all_episode_rewards[bid].append(episode_rewards[bid])
                    logger.info(
                        f"[{bid}] Episode {episode_counts[bid]} | "
                        f"Step {global_step}/{total_steps} | "
                        f"Reward: {episode_rewards[bid]:.1f}"
                    )
                    # Clear this building's model buffer
                    self.model_buffers[bid].pos = 0
                    self.model_buffers[bid].size = 0
                    self.dyna.exploration.apply_decay()
                    episode_rewards[bid] = 0.0
                    raw_obs, _ = self.envs[bid].reset()
                    obs[bid] = self._normalize_obs(raw_obs)

                # Evaluate
                if global_step % eval_freq == 0:
                    eval_result = {
                        bid: all_episode_rewards[bid][-1]
                        if all_episode_rewards[bid]
                        else 0.0
                        for bid in self.building_ids
                    }
                    eval_result["mean"] = np.mean(list(eval_result.values())).item()
                    eval_history.append({"step": global_step, **eval_result})
                    logger.info(f"Eval @ {global_step}: {eval_result}")

        # Save results
        self._save_results(all_episode_rewards, eval_history)

        for env in self.envs.values():
            env.close()

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

        # Need enough data for this building
        if self.real_buffer.tag_count(bid_idx) < batch_size:
            return

        # 1. Update world model on THIS building's data
        if global_step % self.config.dyna.model_update_freq == 0:
            batch = self.real_buffer.sample_tagged(batch_size, bid_idx, self.device)

            # TD-error weights (reward-aware WM training)
            with torch.no_grad():
                next_actions, _ = self.dyna.actor(batch["next_states"])
                q1_next, q2_next = self.dyna.critic_target(
                    batch["next_states"], next_actions
                )
                q_next = torch.min(q1_next, q2_next)
                target_q = (
                    batch["rewards"]
                    + (1.0 - batch["dones"]) * self.config.gamma * q_next
                )
                q1, q2 = self.dyna.critic(batch["states"], batch["actions"])
                td_error = (torch.min(q1, q2) - target_q).abs().squeeze(-1)
                weights = 1.0 + td_error / (td_error.mean() + 1e-8)

            self.dyna.wm_trainer.train_step(
                batch["states"],
                batch["actions"],
                batch["next_states"],
                bid_str,
                weights,
            )

        # 2. Model rollouts (after warmup)
        if global_step >= self.config.dyna.rollout_start_step:
            n_rollouts = self.config.dyna.rollouts_per_step
            horizon = self.config.dyna.rollout_horizon
            start_batch = self.real_buffer.sample_tagged(
                n_rollouts, bid_idx, self.device
            )
            start_states = start_batch["states"].cpu().numpy()
            rollout_data = self.dyna.rollout_gen.generate(
                start_states, bid_str, horizon
            )
            self.model_buffers[bid].add_batch(
                rollout_data["states"],
                rollout_data["actions"],
                rollout_data["rewards"],
                rollout_data["next_states"],
                rollout_data["dones"],
            )

        # 3. SAC update on mixed data (real from ALL buildings + model from THIS)
        real_batch = self.real_buffer.sample(batch_size, self.device)
        model_ratio = self.config.dyna.model_to_real_ratio
        n_model = int(batch_size * model_ratio)

        if n_model > 0 and len(self.model_buffers[bid]) >= n_model:
            model_batch = self.model_buffers[bid].sample(n_model, self.device)
            n_real = batch_size - n_model
            # Trim real batch
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
