"""Standalone SAC baseline trainer for Sinergym environments.

Runs a pure model-free SAC (no world model) to establish reward baselines.
"""

import contextlib
from pathlib import Path

import gymnasium
import numpy as np
import torch
from loguru import logger

from src.utils import sinergym_workdir

with contextlib.suppress(ImportError):
    import sinergym  # noqa: F401 — registers Sinergym envs with gymnasium

from src.agent.obs_normalizer import RunningNormalizer
from src.agent.replay_buffer import ReplayBuffer
from src.agent.sac import GaussianActor, SACTrainer, SoftQNetwork
from src.utils import get_device, seed_everything


class SACBaselineTrainer:
    """Train model-free SAC on a single Sinergym environment.

    Args:
        env_name: Gymnasium/Sinergym environment name.
        seed: Random seed.
        total_timesteps: Total training steps.
        batch_size: SAC batch size.
        buffer_size: Replay buffer capacity.
        learning_starts: Steps before training begins.
        hidden_dims: Actor/critic hidden layer sizes.
        gamma: Discount factor.
        log_interval: Steps between log outputs.
        eval_freq: Steps between evaluations.
        n_eval_episodes: Number of evaluation episodes.
        save_dir: Directory to save checkpoints and results.
        device: Torch device string.
        wandb_project: W&B project name (None to disable).
    """

    def __init__(
        self,
        env_name: str,
        seed: int = 42,
        total_timesteps: int = 8760 * 3,
        batch_size: int = 256,
        buffer_size: int = 100_000,
        learning_starts: int = 1000,
        hidden_dims: list[int] | None = None,
        gamma: float = 0.99,
        log_interval: int = 100,
        eval_freq: int = 8760,
        n_eval_episodes: int = 3,
        save_dir: str = "output/results/baseline_sac",
        device: str = "auto",
        wandb_project: str | None = None,
    ) -> None:
        self.env_name = env_name
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.gamma = gamma
        self.log_interval = log_interval
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = get_device(device)
        self.wandb_project = wandb_project
        hidden_dims = hidden_dims or [256, 256]

        seed_everything(seed)

        # Create environment (sinergym_workdir redirects EnergyPlus temp files)
        with sinergym_workdir():
            self.env = gymnasium.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]  # ty: ignore[not-subscriptable]
        self.action_dim = self.env.action_space.shape[0]  # ty: ignore[not-subscriptable]

        # Compute per-dimension action scale/bias from env
        action_low = self.env.action_space.low  # ty: ignore[unresolved-attribute]
        action_high = self.env.action_space.high  # ty: ignore[unresolved-attribute]
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

        # Observation normalizer
        self.obs_normalizer = RunningNormalizer(shape=(self.state_dim,))

        # Build SAC
        self.actor = GaussianActor(
            self.state_dim,
            self.action_dim,
            hidden_dims=hidden_dims,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
        ).to(self.device)

        self.critic = SoftQNetwork(
            self.state_dim, self.action_dim, hidden_dims=hidden_dims
        ).to(self.device)

        self.critic_target = SoftQNetwork(
            self.state_dim, self.action_dim, hidden_dims=hidden_dims
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.trainer = SACTrainer(
            actor=self.actor,
            critic=self.critic,
            critic_target=self.critic_target,
            gamma=gamma,
            autotune_alpha=True,
            target_entropy=-self.action_dim,
            device=self.device,
        )

        self.buffer = ReplayBuffer(buffer_size, self.state_dim, self.action_dim)

    def train(self) -> dict:
        """Run the full SAC training loop.

        Returns:
            Dict with training history: episode_rewards, eval_rewards, etc.
        """
        run = None
        if self.wandb_project:
            import wandb

            run = wandb.init(
                project=self.wandb_project,
                name=f"sac_baseline_{self.env_name}_s{self.seed}",
                config={
                    "env_name": self.env_name,
                    "seed": self.seed,
                    "total_timesteps": self.total_timesteps,
                },
            )

        raw_obs, _ = self.env.reset(seed=self.seed)
        obs = self.obs_normalizer.update_and_normalize(raw_obs)
        episode_reward = 0.0
        episode_count = 0
        episode_rewards: list[float] = []
        self._recent_episode_rewards: list[float] = []
        eval_history: list[dict] = []

        for step in range(1, self.total_timesteps + 1):
            # Select action
            if step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                state_t = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                with torch.no_grad():
                    action, _ = self.actor(state_t)
                action = action.cpu().numpy().squeeze(0)

            # Step environment
            raw_next_obs, reward, terminated, truncated, _info = self.env.step(action)
            done = terminated or truncated
            next_obs = self.obs_normalizer.update_and_normalize(raw_next_obs)

            self.buffer.add(obs, action, float(reward), next_obs, done)
            episode_reward += float(reward)
            obs = next_obs

            if done:
                episode_count += 1
                episode_rewards.append(episode_reward)
                self._recent_episode_rewards.append(episode_reward)
                logger.info(
                    f"Episode {episode_count} | "
                    f"Step {step}/{self.total_timesteps} | "
                    f"Reward: {episode_reward:.1f}"
                )
                if run:
                    run.log(
                        {"episode_reward": episode_reward, "episode": episode_count},
                        step=step,
                    )
                episode_reward = 0.0
                raw_obs, _ = self.env.reset()
                obs = self.obs_normalizer.normalize(raw_obs)

            # Train SAC
            if step >= self.learning_starts and len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size, self.device)
                metrics = self.trainer.update(
                    batch["states"],
                    batch["actions"],
                    batch["rewards"],
                    batch["next_states"],
                    batch["dones"],
                )
                if step % self.log_interval == 0 and run:
                    run.log(metrics, step=step)

            # Evaluate
            if step % self.eval_freq == 0:
                eval_result = self.evaluate()
                eval_history.append({"step": step, **eval_result})
                logger.info(
                    f"Eval @ step {step}: "
                    f"reward={eval_result['mean_reward']:.1f} ± {eval_result['std_reward']:.1f}"
                )
                if run:
                    run.log({f"eval/{k}": v for k, v in eval_result.items()}, step=step)

        # Save final model and results
        self._save_results(episode_rewards, eval_history)

        self.env.close()
        if run:
            run.finish()

        return {
            "episode_rewards": episode_rewards,
            "eval_history": eval_history,
        }

    def evaluate(self) -> dict:
        """Evaluate by recording the most recent training episode reward.

        Avoids creating a new Sinergym env instance (which can cause
        EnergyPlus SIGSEGV on concurrent instances).
        """
        # Use accumulated episode rewards as eval metric
        if self._recent_episode_rewards:
            recent = self._recent_episode_rewards[-1]
            return {
                "mean_reward": recent,
                "std_reward": 0.0,
            }
        return {"mean_reward": 0.0, "std_reward": 0.0}

    def _save_results(
        self, episode_rewards: list[float], eval_history: list[dict]
    ) -> None:
        """Save checkpoint and training results."""
        # Save model
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "env_name": self.env_name,
                "seed": self.seed,
            },
            self.save_dir / "sac_checkpoint.pt",
        )
        # Save rewards
        np.save(self.save_dir / "episode_rewards.npy", np.array(episode_rewards))
        # Save eval history
        import json

        with open(self.save_dir / "eval_history.json", "w") as f:
            json.dump(eval_history, f, indent=2)

        logger.info(f"Results saved to {self.save_dir}")


class RBCBaseline:
    """Rule-Based Control baseline for comparison.

    Uses midpoint of action space as constant action.
    """

    def __init__(
        self,
        env_name: str,
        n_episodes: int = 3,
        seed: int = 42,
        save_dir: str = "output/results/baseline_rbc",
    ) -> None:
        self.env_name = env_name
        self.n_episodes = n_episodes
        self.seed = seed
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self) -> dict:
        """Run RBC and collect episode rewards."""
        with sinergym_workdir():
            env = gymnasium.make(self.env_name)
        # env is created, CWD restored; env runs EnergyPlus in its own workdir
        action_mid = (env.action_space.low + env.action_space.high) / 2.0  # ty: ignore[unresolved-attribute]

        rewards = []
        for ep in range(self.n_episodes):
            _obs, _ = env.reset(seed=self.seed + ep)
            total_reward = 0.0
            done = False
            while not done:
                _obs, reward, terminated, truncated, _ = env.step(action_mid)
                total_reward += float(reward)
                done = terminated or truncated
            rewards.append(total_reward)
            logger.info(f"RBC Episode {ep + 1}: reward={total_reward:.1f}")

        env.close()

        result = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "episode_rewards": rewards,
        }

        np.save(self.save_dir / "rbc_rewards.npy", np.array(rewards))
        logger.info(f"RBC: {result['mean_reward']:.1f} ± {result['std_reward']:.1f}")
        return result
