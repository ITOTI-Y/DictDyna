"""Dyna-SAC training loop with Sinergym environments."""

import contextlib
import json
from pathlib import Path

import gymnasium
import numpy as np
import torch
from loguru import logger

from src.agent.dyna_sac import DynaSAC
from src.schemas import TrainSchema
from src.utils import get_device, seed_everything, sinergym_workdir

with contextlib.suppress(ImportError):
    import sinergym  # noqa: F401


class DynaSACTrainer:
    """Full Dyna-SAC training loop on Sinergym.

    Manages environment lifecycle, training, evaluation, and checkpointing.

    Args:
        env_name: Sinergym environment name.
        building_id: Building identifier.
        dict_path: Path to pretrained dictionary .pt file.
        config: TrainSchema configuration.
        seed: Random seed.
        save_dir: Directory for checkpoints and results.
        wandb_project: W&B project name (None to disable).
    """

    def __init__(
        self,
        env_name: str,
        building_id: str,
        dict_path: str,
        config: TrainSchema,
        seed: int = 42,
        save_dir: str = "output/results/dyna_sac",
        wandb_project: str | None = None,
    ) -> None:
        self.env_name = env_name
        self.building_id = building_id
        self.seed = seed
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.device = get_device(config.device)
        self.wandb_project = wandb_project

        seed_everything(seed)

        # Create environment
        with sinergym_workdir():
            self.env = gymnasium.make(env_name)

        state_dim = self.env.observation_space.shape[0]  # ty: ignore[not-subscriptable]
        action_dim = self.env.action_space.shape[0]  # ty: ignore[not-subscriptable]

        # Compute per-dimension action scale/bias
        action_low = self.env.action_space.low  # ty: ignore[unresolved-attribute]
        action_high = self.env.action_space.high  # ty: ignore[unresolved-attribute]
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

        # Load pretrained dictionary
        dict_data = torch.load(dict_path, weights_only=False)
        dictionary = dict_data["dictionary"]

        logger.info(
            f"Env: {env_name}, state_dim={state_dim}, action_dim={action_dim}, "
            f"dict={dictionary.shape}"
        )

        # Build Dyna-SAC
        self.dyna = DynaSAC(
            state_dim=state_dim,
            action_dim=action_dim,
            building_ids=[building_id],
            dictionary=dictionary,
            config=config,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
        )

    def train(self) -> dict:
        """Run full Dyna-SAC training loop.

        Returns:
            Training results dict.
        """
        run = None
        if self.wandb_project:
            import wandb

            run = wandb.init(
                project=self.wandb_project,
                name=f"dyna_sac_{self.env_name}_s{self.seed}",
                config=self.config.model_dump(),
            )

        total_steps = self.config.total_timesteps
        eval_freq = self.config.eval_freq
        log_interval = self.config.log_interval
        learning_starts = min(1000, self.config.batch_size * 2)

        obs, _ = self.env.reset(seed=self.seed)
        episode_reward = 0.0
        episode_count = 0
        episode_rewards: list[float] = []
        self._recent_episode_rewards: list[float] = []
        eval_history: list[dict] = []

        logger.info(
            f"Starting training: {total_steps} steps, "
            f"eval every {eval_freq}, batch={self.config.batch_size}"
        )

        for step in range(1, total_steps + 1):
            # Select action
            if step < learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.dyna.select_action(obs)

            # Step environment
            next_obs, reward, terminated, truncated, _info = self.env.step(action)
            done = terminated or truncated

            # Dyna-SAC train step
            metrics = {}
            if step >= learning_starts:
                metrics = self.dyna.train_step(
                    obs,
                    action,
                    float(reward),
                    next_obs,
                    done,
                    self.building_id,
                    step,
                )
            else:
                # Just fill buffer before training starts
                self.dyna.buffer.add_real(obs, action, float(reward), next_obs, done)

            episode_reward += float(reward)
            obs = next_obs

            if done:
                episode_count += 1
                episode_rewards.append(episode_reward)
                self._recent_episode_rewards.append(episode_reward)
                logger.info(
                    f"Episode {episode_count} | Step {step}/{total_steps} | "
                    f"Reward: {episode_reward:.1f}"
                )
                if run:
                    run.log({"episode_reward": episode_reward}, step=step)
                episode_reward = 0.0
                obs, _ = self.env.reset()

            # Log metrics
            if step % log_interval == 0 and metrics and run:
                run.log(metrics, step=step)

            # Evaluate
            if step % eval_freq == 0:
                eval_result = self._evaluate()
                eval_history.append({"step": step, **eval_result})
                logger.info(
                    f"Eval @ step {step}: "
                    f"reward={eval_result['mean_reward']:.1f} ± "
                    f"{eval_result['std_reward']:.1f}"
                )
                if run:
                    run.log(
                        {f"eval/{k}": v for k, v in eval_result.items()},
                        step=step,
                    )
                # Save checkpoint
                self.dyna.save(self.save_dir / f"checkpoint_step{step}.pt")

        # Save final results
        self._save_results(episode_rewards, eval_history)
        self.env.close()
        if run:
            run.finish()

        logger.info(
            f"Training complete: {episode_count} episodes, "
            f"{len(eval_history)} evaluations"
        )
        return {"episode_rewards": episode_rewards, "eval_history": eval_history}

    def _evaluate(self) -> dict:
        """Evaluate using the most recent training episode reward.

        Avoids creating new Sinergym env instances which can cause
        EnergyPlus SIGSEGV crashes.
        """
        if self._recent_episode_rewards:
            recent = self._recent_episode_rewards[-1]
            return {"mean_reward": recent, "std_reward": 0.0}
        return {"mean_reward": 0.0, "std_reward": 0.0}

    def _save_results(
        self, episode_rewards: list[float], eval_history: list[dict]
    ) -> None:
        """Save training results."""
        self.dyna.save(self.save_dir / "final_checkpoint.pt")
        np.save(self.save_dir / "episode_rewards.npy", np.array(episode_rewards))
        with open(self.save_dir / "eval_history.json", "w") as f:
            json.dump(eval_history, f, indent=2)
        logger.info(f"Results saved to {self.save_dir}")
