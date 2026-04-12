"""SAC training with UniversalObsEncoder for heterogeneous buildings.

Operates in embed space (128d fixed) rather than raw obs space.
The UniversalObsEncoder maps any building's raw obs to a shared embedding,
enabling a SINGLE actor to be trained on one building and evaluated on another
with a different obs_dim.

No world model / Dyna rollouts — Phase 7 showed WM rollout contributes ~0%
to transfer performance, and reward estimation in embed space is non-trivial.
Pure SAC in embed space is sufficient for the transfer experiment.
"""

import contextlib
import json
from collections import deque
from pathlib import Path

import gymnasium
import numpy as np
import torch
from loguru import logger

from src.agent.replay_buffer import ReplayBuffer
from src.agent.sac import GaussianActor, SACTrainer, SoftQNetwork
from src.obs_config_universal import (
    TOTAL_EMBED_DIM,
    CategoryMapping,
    build_category_mapping,
)
from src.obs_encoder import UniversalObsEncoder
from src.utils import get_device, seed_everything, sinergym_workdir

with contextlib.suppress(ImportError):
    import sinergym  # noqa: F401


class UniversalSACTrainer:
    """SAC trainer with UniversalObsEncoder.

    Args:
        env_name: Sinergym environment name.
        building_id: Building identifier.
        obs_encoder: Shared UniversalObsEncoder (can be pre-trained or fresh).
        seed: Random seed.
        total_timesteps: Total training steps.
        batch_size: SAC batch size.
        buffer_size: Replay buffer capacity.
        hidden_dims: Actor/critic hidden dims.
        gamma: Discount factor.
        eval_freq: Steps between evaluations.
        save_dir: Output directory.
        device: Torch device string.
        train_encoder: If True, encoder gradients flow during SAC training.
    """

    def __init__(
        self,
        env_name: str,
        building_id: str,
        obs_encoder: UniversalObsEncoder,
        seed: int = 42,
        total_timesteps: int = 35040,
        batch_size: int = 256,
        buffer_size: int = 100_000,
        hidden_dims: list[int] | None = None,
        gamma: float = 0.99,
        learning_starts: int = 1000,
        eval_freq: int = 35040,
        save_dir: str = "output/results/universal_sac",
        device: str = "auto",
        train_encoder: bool = True,
    ) -> None:
        self.env_name = env_name
        self.building_id = building_id
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.eval_freq = eval_freq
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = get_device(device)
        self.train_encoder = train_encoder
        hidden_dims = hidden_dims or [256, 256]

        seed_everything(seed)

        # Create env and probe dims
        with sinergym_workdir():
            self.env = gymnasium.make(env_name)
        raw_obs_dim = self.env.observation_space.shape[0]  # ty: ignore[not-subscriptable]
        self.action_dim = self.env.action_space.shape[0]  # ty: ignore[not-subscriptable]

        action_low = self.env.action_space.low  # ty: ignore[unresolved-attribute]
        action_high = self.env.action_space.high  # ty: ignore[unresolved-attribute]
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

        # Build category mapping from env
        obs_vars = list(self.env.unwrapped.observation_variables)
        self.mapping = build_category_mapping(building_id, obs_vars)
        logger.info(
            f"Building {building_id}: raw_obs_dim={raw_obs_dim}, "
            f"categories={dict_summary(self.mapping)}"
        )

        # Obs encoder (shared across buildings)
        self.obs_encoder = obs_encoder.to(self.device)
        self.embed_dim = TOTAL_EMBED_DIM

        # SAC in embed space
        self.actor = GaussianActor(
            self.embed_dim,
            self.action_dim,
            hidden_dims=hidden_dims,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
        ).to(self.device)

        self.critic = SoftQNetwork(
            self.embed_dim, self.action_dim, hidden_dims=hidden_dims
        ).to(self.device)

        self.critic_target = SoftQNetwork(
            self.embed_dim, self.action_dim, hidden_dims=hidden_dims
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.sac_trainer = SACTrainer(
            actor=self.actor,
            critic=self.critic,
            critic_target=self.critic_target,
            gamma=gamma,
            autotune_alpha=True,
            target_entropy=-self.action_dim,
            device=self.device,
        )

        self.buffer = ReplayBuffer(buffer_size, self.embed_dim, self.action_dim)

    def _encode(self, raw_obs: np.ndarray) -> np.ndarray:
        """Encode raw obs to embed space."""
        raw_t = torch.tensor(
            raw_obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        if self.train_encoder:
            embed = self.obs_encoder(raw_t, self.mapping)
        else:
            with torch.no_grad():
                embed = self.obs_encoder(raw_t, self.mapping)
        return embed.squeeze(0).cpu().detach().numpy()

    def train(self) -> dict:
        """Run SAC training in embed space."""
        raw_obs, _ = self.env.reset(seed=self.seed)
        obs = self._encode(raw_obs)
        episode_reward = 0.0
        episode_count = 0
        episode_rewards: list[float] = []
        recent_rewards: deque[float] = deque(maxlen=3)
        eval_history: list[dict] = []

        logger.info(
            f"Training {self.building_id}: {self.total_timesteps} steps, "
            f"embed_dim={self.embed_dim}"
        )

        for step in range(1, self.total_timesteps + 1):
            if step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                state_t = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                with torch.no_grad():
                    action, _ = self.actor(state_t)
                action = action.cpu().numpy().squeeze(0)

            raw_next, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            next_obs = self._encode(raw_next)

            self.buffer.add(obs, action, float(reward), next_obs, done)
            episode_reward += float(reward)
            obs = next_obs

            if done:
                episode_count += 1
                episode_rewards.append(episode_reward)
                recent_rewards.append(episode_reward)
                logger.info(
                    f"Episode {episode_count} | Step {step}/{self.total_timesteps} | "
                    f"Reward: {episode_reward:.1f}"
                )
                episode_reward = 0.0
                raw_obs, _ = self.env.reset()
                obs = self._encode(raw_obs)

            if step >= self.learning_starts and len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size, self.device)
                self.sac_trainer.update(
                    batch["states"],
                    batch["actions"],
                    batch["rewards"],
                    batch["next_states"],
                    batch["dones"],
                )

            if step % self.eval_freq == 0 and recent_rewards:
                rewards = np.array(recent_rewards)
                eval_result = {
                    "mean_reward": float(rewards.mean()),
                    "std_reward": float(rewards.std()),
                }
                eval_history.append({"step": step, **eval_result})
                logger.info(
                    f"Eval @ step {step}: reward={eval_result['mean_reward']:.1f}"
                )

        self._save(episode_rewards, eval_history)
        self.env.close()
        return {"episode_rewards": episode_rewards, "eval_history": eval_history}

    def evaluate_on_env(
        self, env_name: str, building_id: str, n_episodes: int = 1
    ) -> dict:
        """Evaluate actor on a (potentially different) building.

        The same obs_encoder encodes the new building's raw obs into
        embed space, and the source-trained actor produces actions.
        Actions are clipped to the target env's valid range to handle
        cross-building action space differences.
        """
        with sinergym_workdir():
            env = gymnasium.make(env_name)
        obs_vars = list(env.unwrapped.observation_variables)
        target_mapping = build_category_mapping(building_id, obs_vars)

        # Target env action bounds (may differ from source)
        act_low = env.action_space.low  # ty: ignore[unresolved-attribute]
        act_high = env.action_space.high  # ty: ignore[unresolved-attribute]

        rewards: list[float] = []
        for ep in range(n_episodes):
            raw_obs, _ = env.reset(seed=self.seed + 1000 + ep)
            episode_reward = 0.0
            done = False
            while not done:
                raw_t = torch.tensor(
                    raw_obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                with torch.no_grad():
                    embed = self.obs_encoder(raw_t, target_mapping)
                    action = self.actor.get_action(embed)
                action = action.cpu().numpy().squeeze(0)
                # Clip to target env's valid action range
                action = np.clip(action, act_low, act_high)
                raw_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += float(reward)
            rewards.append(episode_reward)
            logger.info(
                f"Eval {building_id} ep {ep + 1}/{n_episodes}: {episode_reward:.1f}"
            )

        env.close()
        arr = np.array(rewards)
        return {
            "mean_reward": float(arr.mean()),
            "std_reward": float(arr.std()),
            "episode_rewards": rewards,
        }

    def _save(self, episode_rewards: list[float], eval_history: list[dict]) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "obs_encoder": self.obs_encoder.state_dict(),
                "building_id": self.building_id,
                "embed_dim": self.embed_dim,
            },
            self.save_dir / "checkpoint.pt",
        )
        np.save(self.save_dir / "episode_rewards.npy", np.array(episode_rewards))
        with open(self.save_dir / "eval_history.json", "w") as f:
            json.dump(eval_history, f, indent=2)
        logger.info(f"Saved to {self.save_dir}")


def dict_summary(mapping: CategoryMapping) -> str:
    """One-line summary of category counts."""
    parts = []
    for cat, indices in mapping.category_indices.items():
        if indices:
            parts.append(f"{cat}={len(indices)}")
    return ", ".join(parts)
