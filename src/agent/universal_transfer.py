"""Cross-building transfer experiment using UniversalObsEncoder.

Enables transfer between buildings with DIFFERENT obs_dim (e.g., 5zone 17d → warehouse 22d)
by operating in a shared 128-dim embed space produced by UniversalObsEncoder.

Experiment workflow:
    1. Train SAC on source buildings in embed space
    2. Evaluate source actor on target building (pure zero-shot)
    3. Compare with scratch baseline trained on target
"""

import contextlib
import json
from pathlib import Path

from loguru import logger

from src.agent.universal_trainer import UniversalSACTrainer
from src.obs_encoder import UniversalObsEncoder
from src.utils import seed_everything

with contextlib.suppress(ImportError):
    import sinergym  # noqa: F401


class UniversalTransferExperiment:
    """Cross-building transfer with heterogeneous obs dimensions.

    Args:
        source_configs: Source building configs [{env_name, building_id}].
        target_configs: Target building configs [{env_name, building_id}].
        seed: Random seed.
        total_timesteps: Training steps per source building.
        save_dir: Output directory.
        device: Torch device.
    """

    def __init__(
        self,
        source_configs: list[dict],
        target_configs: list[dict],
        seed: int = 42,
        total_timesteps: int = 35040,
        hidden_dims: list[int] | None = None,
        save_dir: str = "output/results/universal_transfer",
        device: str = "auto",
    ) -> None:
        self.source_configs = source_configs
        self.target_configs = target_configs
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.hidden_dims = hidden_dims or [256, 256]
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        seed_everything(seed)
        self.obs_encoder = UniversalObsEncoder()

    def run(self) -> dict:
        """Run full transfer experiment."""
        results: dict = {}

        # Phase 1: Train on source buildings
        logger.info("=== Phase 1: Source training ===")
        source_trainer = self._train_source()

        # Phase 2: Evaluate on all targets
        for target in self.target_configs:
            tid = target["building_id"]
            tenv = target["env_name"]

            # 2a: Pure zero-shot
            logger.info(f"=== Phase 2a: Pure zero-shot on {tid} ===")
            zs_result = source_trainer.evaluate_on_env(tenv, tid)
            results[f"zero_shot_{tid}"] = zs_result["mean_reward"]
            logger.info(f"  zero_shot {tid}: {zs_result['mean_reward']:.1f}")

            # 2b: Scratch baseline
            logger.info(f"=== Phase 2b: Scratch on {tid} ===")
            scratch_result = self._train_scratch(target)
            results[f"scratch_{tid}"] = scratch_result
            logger.info(f"  scratch {tid}: {scratch_result:.1f}")

            # Advantage
            if abs(scratch_result) > 0:
                adv = (
                    (results[f"zero_shot_{tid}"] - scratch_result)
                    / abs(scratch_result)
                    * 100
                )
                results[f"advantage_{tid}"] = adv
                logger.info(f"  advantage {tid}: {adv:+.1f}%")

        # Save results
        with open(self.save_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {self.save_dir}")

        return results

    def _train_source(self) -> UniversalSACTrainer:
        """Train SAC on source buildings sequentially.

        Inherits actor/critic/buffer state across buildings so training
        progress accumulates (previously each new trainer started fresh,
        losing the first building's training).
        """
        prev_trainer: UniversalSACTrainer | None = None
        for i, cfg in enumerate(self.source_configs):
            bid = cfg["building_id"]
            env_name = cfg["env_name"]
            logger.info(f"Training on source {i + 1}/{len(self.source_configs)}: {bid}")

            # Each building gets a trainer scoped to its env, but inherits
            # learned parameters from the previous building.
            trainer = UniversalSACTrainer(
                env_name=env_name,
                building_id=bid,
                obs_encoder=self.obs_encoder,
                seed=self.seed + i,  # vary seed per building to diversify buffer
                total_timesteps=self.total_timesteps,
                hidden_dims=self.hidden_dims,
                save_dir=str(self.save_dir / f"source_{bid}"),
                device=self.device,
                train_encoder=True,
            )

            # Inherit actor/critic/buffer from previous building
            if prev_trainer is not None:
                trainer.actor.load_state_dict(prev_trainer.actor.state_dict())
                trainer.critic.load_state_dict(prev_trainer.critic.state_dict())
                trainer.critic_target.load_state_dict(
                    prev_trainer.critic_target.state_dict()
                )
                # Carry replay buffer for cross-building sample efficiency
                trainer.buffer = prev_trainer.buffer
                # Keep optimizer states from SACTrainer (inherit from prev)
                trainer.sac_trainer.actor_optimizer.load_state_dict(
                    prev_trainer.sac_trainer.actor_optimizer.state_dict()
                )
                trainer.sac_trainer.critic_optimizer.load_state_dict(
                    prev_trainer.sac_trainer.critic_optimizer.state_dict()
                )
                logger.info("  Inherited state from previous source trainer")

            trainer.train()
            prev_trainer = trainer

        assert prev_trainer is not None
        return prev_trainer

    def _train_scratch(self, target_config: dict) -> float:
        """Train from scratch on target building."""
        # Fresh encoder (no source knowledge)
        scratch_encoder = UniversalObsEncoder()

        scratch_trainer = UniversalSACTrainer(
            env_name=target_config["env_name"],
            building_id=target_config["building_id"],
            obs_encoder=scratch_encoder,
            seed=self.seed,
            total_timesteps=self.total_timesteps,
            hidden_dims=self.hidden_dims,
            save_dir=str(self.save_dir / f"scratch_{target_config['building_id']}"),
            device=self.device,
            train_encoder=True,
        )
        scratch_trainer.train()

        # Evaluate with independent eval
        eval_result = scratch_trainer.evaluate_on_env(
            target_config["env_name"], target_config["building_id"]
        )
        return eval_result["mean_reward"]
