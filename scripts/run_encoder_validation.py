"""Minimum validation experiment: does UniversalObsEncoder provide cross-climate transfer gain on 5zone?

Per 2026-04-13 review's stage-1 recommendation, this experiment tests
the encoder's value BEFORE heterogeneous buildings:
  Source: 5zone_hot + 5zone_mixed
  Target: 5zone_cool
  (same action space → no action-adapter bug interferes)

Conditions compared:
  1. random_frozen_encoder   — encoder random init, never trained (old v2 behavior baseline)
  2. trained_encoder          — encoder trained end-to-end with SAC (fixed behavior)
  3. scratch                  — target building trained from scratch (no transfer)

Evaluation: n_episodes=3 per target (fixes single-ep deterministic-eval fragility).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger

from src.agent.universal_trainer import UniversalSACTrainer
from src.obs_encoder import UniversalObsEncoder
from src.utils import seed_everything

SOURCE = [
    {"env_name": "Eplus-5zone-hot-continuous-v1", "building_id": "5zone_hot"},
    {"env_name": "Eplus-5zone-mixed-continuous-v1", "building_id": "5zone_mixed"},
]
TARGET = {"env_name": "Eplus-5zone-cool-continuous-v1", "building_id": "5zone_cool"}

DEFAULT_SEEDS = [42, 123, 7]


def train_source_sequential(
    source_configs: list[dict],
    obs_encoder: UniversalObsEncoder,
    seed: int,
    total_timesteps: int,
    save_dir: Path,
    train_encoder: bool,
    device: str,
) -> UniversalSACTrainer:
    """Train SAC on source buildings sequentially, inheriting state."""
    prev: UniversalSACTrainer | None = None
    for i, cfg in enumerate(source_configs):
        bid = cfg["building_id"]
        logger.info(f"Source {i + 1}/{len(source_configs)}: {bid}")

        trainer = UniversalSACTrainer(
            env_name=cfg["env_name"],
            building_id=bid,
            obs_encoder=obs_encoder,
            seed=seed + i,
            total_timesteps=total_timesteps,
            save_dir=str(save_dir / f"source_{bid}"),
            device=device,
            train_encoder=train_encoder,
        )
        if prev is not None:
            trainer.actor.load_state_dict(prev.actor.state_dict())
            trainer.critic.load_state_dict(prev.critic.state_dict())
            trainer.critic_target.load_state_dict(prev.critic_target.state_dict())
            trainer.buffer = prev.buffer
            logger.info("  Inherited actor/critic/buffer from previous")

        trainer.train()
        prev = trainer
    assert prev is not None
    return prev


def run_condition(
    condition: str,
    seed: int,
    total_timesteps: int,
    save_dir: Path,
    n_eval_episodes: int,
    device: str,
) -> dict:
    """Run a single condition and return its target evaluation.

    condition in {random_frozen, trained, scratch}:
      - random_frozen: source-trained actor with frozen random encoder
      - trained:       source-trained actor with encoder trained e2e
      - scratch:       target-only training with trained encoder
    """
    seed_everything(seed)
    encoder = UniversalObsEncoder()

    if condition in ("random_frozen", "trained"):
        train_encoder = condition == "trained"
        src_trainer = train_source_sequential(
            source_configs=SOURCE,
            obs_encoder=encoder,
            seed=seed,
            total_timesteps=total_timesteps,
            save_dir=save_dir / condition,
            train_encoder=train_encoder,
            device=device,
        )
        result = src_trainer.evaluate_on_env(
            TARGET["env_name"], TARGET["building_id"], n_episodes=n_eval_episodes
        )
        return {
            "mean": result["mean_reward"],
            "std": result["std_reward"],
            "episodes": result["episode_rewards"],
        }

    if condition == "scratch":
        scratch_trainer = UniversalSACTrainer(
            env_name=TARGET["env_name"],
            building_id=TARGET["building_id"],
            obs_encoder=encoder,
            seed=seed,
            total_timesteps=total_timesteps,
            save_dir=str(save_dir / "scratch"),
            device=device,
            train_encoder=True,  # always train for scratch
        )
        scratch_trainer.train()
        result = scratch_trainer.evaluate_on_env(
            TARGET["env_name"], TARGET["building_id"], n_episodes=n_eval_episodes
        )
        return {
            "mean": result["mean_reward"],
            "std": result["std_reward"],
            "episodes": result["episode_rewards"],
        }

    raise ValueError(f"Unknown condition: {condition}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate encoder value on 5zone-only transfer"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--timesteps", type=int, default=70080, help="Steps per building (2 episodes)"
    )
    parser.add_argument("--n-eval-episodes", type=int, default=3)
    parser.add_argument("--save-dir", default="output/results/encoder_validation")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["random_frozen", "trained", "scratch"],
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[int, dict] = {}

    for seed in args.seeds:
        logger.info(f"\n{'=' * 60}\nSEED {seed}\n{'=' * 60}")
        seed_results: dict[str, dict] = {}

        for cond in args.conditions:
            logger.info(f"\n--- Condition: {cond} ---")
            r = run_condition(
                condition=cond,
                seed=seed,
                total_timesteps=args.timesteps,
                save_dir=save_dir / f"s{seed}",
                n_eval_episodes=args.n_eval_episodes,
                device="auto",
            )
            seed_results[cond] = r
            logger.info(f"  {cond}: {r['mean']:.1f} ± {r['std']:.1f}")

        all_results[seed] = seed_results

    # Summary
    print(f"\n{'=' * 70}")
    print("ENCODER VALIDATION SUMMARY (5zone_hot+mixed → 5zone_cool)")
    print(f"{'=' * 70}")
    for cond in args.conditions:
        vals = [all_results[s][cond]["mean"] for s in args.seeds]
        print(f"  {cond:<20s} mean={np.mean(vals):>8.0f} std={np.std(vals):>5.0f}")

    print()
    if "random_frozen" in args.conditions and "trained" in args.conditions:
        rf = [all_results[s]["random_frozen"]["mean"] for s in args.seeds]
        tr = [all_results[s]["trained"]["mean"] for s in args.seeds]
        gain = [(t - r) / abs(r) * 100 for t, r in zip(tr, rf, strict=True)]
        print(
            f"  Encoder training gain (trained vs random_frozen): "
            f"{np.mean(gain):+.1f}% ± {np.std(gain):.1f}%"
        )

    if "scratch" in args.conditions:
        for trans in ["random_frozen", "trained"]:
            if trans not in args.conditions:
                continue
            sc = [all_results[s]["scratch"]["mean"] for s in args.seeds]
            tr = [all_results[s][trans]["mean"] for s in args.seeds]
            adv = [(t - s) / abs(s) * 100 for t, s in zip(tr, sc, strict=True)]
            print(
                f"  Transfer advantage ({trans} vs scratch): "
                f"{np.mean(adv):+.1f}% ± {np.std(adv):.1f}%"
            )

    output = {
        "seeds": args.seeds,
        "source": [s["building_id"] for s in SOURCE],
        "target": TARGET["building_id"],
        "timesteps": args.timesteps,
        "n_eval_episodes": args.n_eval_episodes,
        "per_seed": {str(k): v for k, v in all_results.items()},
    }
    with open(save_dir / "all_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {save_dir / 'all_results.json'}")


if __name__ == "__main__":
    main()
