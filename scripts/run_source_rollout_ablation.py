"""Ablation: does WM rollout help during SOURCE training?

B. Source WITH rollout → Transfer (no rollout) [from previous ablation]
D. Source WITHOUT rollout → Transfer (no rollout)

D vs B = source-stage WM rollout contribution.
If D ≈ B, WM contributes nothing — advantage is pure SAC pretraining on source data.
"""

import json

import numpy as np
from loguru import logger

from src.agent.transfer_experiment import FewShotTransferExperiment
from src.schemas import TrainSchema

SEEDS = [42, 123, 7]
DAYS = [1, 3, 7]
DICT_PATH = "output/pretrained/dict_k128_source_only.pt"
SOURCE = [
    {"env_name": "Eplus-5zone-hot-continuous-v1", "building_id": "office_hot"},
    {"env_name": "Eplus-5zone-mixed-continuous-v1", "building_id": "office_mixed"},
]
TARGET = {"env_name": "Eplus-5zone-cool-continuous-v1", "building_id": "office_cool"}

all_results: dict[int, dict] = {}

for seed in SEEDS:
    logger.info(f"\n{'=' * 60}")
    logger.info(f"SEED {seed}")
    logger.info(f"{'=' * 60}")

    config = TrainSchema(
        mode="context", seed=seed, total_timesteps=35040, device="auto"
    )
    exp = FewShotTransferExperiment(
        source_configs=SOURCE,
        target_config=TARGET,
        dict_path=DICT_PATH,
        config=config,
        adaptation_days=DAYS,
        seed=seed,
        save_dir=f"output/results/source_rollout_ablation/s{seed}",
        context_mode=True,
    )

    target_data = exp._collect_target_data()

    # B: Source WITH rollout → Transfer no-rollout
    logger.info("=== B: Source WITH rollout ===")
    source_with = exp._train_source()

    # D: Source WITHOUT rollout → Transfer no-rollout
    logger.info("=== D: Source WITHOUT rollout ===")
    source_without = exp._train_source_no_rollout()

    seed_results: dict[str, float] = {}
    for days in DAYS:
        steps = days * 96
        logger.info(f"\n--- {days}d ({steps} steps) ---")

        # B: transfer from source WITH rollout (no rollout at transfer time)
        b = exp._run_context_transfer_no_rollout(source_with, target_data, steps)
        seed_results[f"B_src_rollout_{days}d"] = b
        logger.info(f"  B (source+rollout → transfer): {b:.1f}")

        # D: transfer from source WITHOUT rollout (no rollout at transfer time)
        d = exp._run_context_transfer_no_rollout(source_without, target_data, steps)
        seed_results[f"D_src_no_rollout_{days}d"] = d
        logger.info(f"  D (source-no-rollout → transfer): {d:.1f}")

        diff = (b - d) / abs(d) * 100
        logger.info(f"  Source rollout value (B vs D): {diff:+.1f}%")

    all_results[seed] = seed_results

# Aggregate
print("\n" + "=" * 70)
print("SOURCE ROLLOUT ABLATION: 3-SEED SUMMARY")
print("=" * 70)

for d in DAYS:
    b_vals = [all_results[s][f"B_src_rollout_{d}d"] for s in SEEDS]
    d_vals = [all_results[s][f"D_src_no_rollout_{d}d"] for s in SEEDS]
    diffs = [(b - dv) / abs(dv) * 100 for b, dv in zip(b_vals, d_vals, strict=True)]

    print(f"\n  {d}d:")
    print(
        f"    B (source+rollout):    {np.mean(b_vals):>8.0f} +/- {np.std(b_vals):.0f}"
    )
    print(
        f"    D (source-no-rollout): {np.mean(d_vals):>8.0f} +/- {np.std(d_vals):.0f}"
    )
    print(
        f"    Source WM rollout value: {np.mean(diffs):+.1f} +/- {np.std(diffs):.1f}%"
    )
    for i, s in enumerate(SEEDS):
        print(
            f"      s{s}: B={b_vals[i]:.0f}  D={d_vals[i]:.0f}  diff={diffs[i]:+.1f}%"
        )

print("=" * 70)

with open("output/results/source_rollout_ablation/all_results.json", "w") as f:
    json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
