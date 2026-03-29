"""Ablation: isolate world model rollout contribution in transfer.

A. Transfer with rollouts (existing)
B. Transfer WITHOUT rollouts (SAC on real data only)
C. Scratch with rollouts (existing)

WM rollout contribution = A - B
Policy transfer contribution = B - C
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
        save_dir=f"output/results/rollout_ablation/s{seed}",
        context_mode=True,
    )

    # Phase 1+2: shared across conditions
    logger.info("=== Phase 1: Training source ===")
    source_dyna = exp._train_source()
    logger.info("=== Phase 2: Collecting target data ===")
    target_data = exp._collect_target_data()

    seed_results: dict[str, float] = {}
    for days in DAYS:
        steps = days * 96
        logger.info(f"\n--- {days}d ({steps} steps) ---")

        # A: Transfer WITH rollouts
        a = exp._run_context_transfer(source_dyna, target_data, steps)
        seed_results[f"A_transfer_{days}d"] = a
        logger.info(f"  A (transfer+rollout): {a:.1f}")

        # B: Transfer WITHOUT rollouts
        b = exp._run_context_transfer_no_rollout(source_dyna, target_data, steps)
        seed_results[f"B_no_rollout_{days}d"] = b
        logger.info(f"  B (transfer-no-rollout): {b:.1f}")

        # C: Scratch WITH rollouts
        c = exp._run_from_scratch(target_data, steps)
        seed_results[f"C_scratch_{days}d"] = c
        logger.info(f"  C (scratch+rollout): {c:.1f}")

        # Decomposition
        rollout_contrib = (a - b) / abs(b) * 100
        policy_contrib = (b - c) / abs(c) * 100
        total_adv = (a - c) / abs(c) * 100
        logger.info(f"  Total advantage (A-C): {total_adv:+.1f}%")
        logger.info(f"  WM rollout contribution (A-B): {rollout_contrib:+.1f}%")
        logger.info(f"  Policy transfer contribution (B-C): {policy_contrib:+.1f}%")

    all_results[seed] = seed_results

# Aggregate
print("\n" + "=" * 70)
print("ROLLOUT ABLATION: 3-SEED SUMMARY")
print("=" * 70)

for d in DAYS:
    a_vals = [all_results[s][f"A_transfer_{d}d"] for s in SEEDS]
    b_vals = [all_results[s][f"B_no_rollout_{d}d"] for s in SEEDS]
    c_vals = [all_results[s][f"C_scratch_{d}d"] for s in SEEDS]

    total_advs = [(a - c) / abs(c) * 100 for a, c in zip(a_vals, c_vals, strict=True)]
    rollout_contribs = [a - b for a, b in zip(a_vals, b_vals, strict=True)]
    policy_contribs = [b - c for b, c in zip(b_vals, c_vals, strict=True)]

    print(f"\n  {d}d:")
    print(
        f"    A (transfer+rollout):    {np.mean(a_vals):>8.0f} +/- {np.std(a_vals):.0f}"
    )
    print(
        f"    B (transfer-no-rollout): {np.mean(b_vals):>8.0f} +/- {np.std(b_vals):.0f}"
    )
    print(
        f"    C (scratch+rollout):     {np.mean(c_vals):>8.0f} +/- {np.std(c_vals):.0f}"
    )
    print(
        f"    Total advantage (A vs C):  {np.mean(total_advs):+.1f} +/- {np.std(total_advs):.1f}%"
    )
    print(
        f"    WM rollout value (A - B):  {np.mean(rollout_contribs):+.0f} +/- {np.std(rollout_contribs):.0f} reward"
    )
    print(
        f"    Policy transfer (B - C):   {np.mean(policy_contribs):+.0f} +/- {np.std(policy_contribs):.0f} reward"
    )

    # Percentage decomposition
    total_gap = np.mean(a_vals) - np.mean(c_vals)
    rollout_pct = (
        (np.mean(a_vals) - np.mean(b_vals)) / abs(total_gap) * 100
        if total_gap != 0
        else 0
    )
    policy_pct = (
        (np.mean(b_vals) - np.mean(c_vals)) / abs(total_gap) * 100
        if total_gap != 0
        else 0
    )
    print(
        f"    Decomposition: WM rollout={rollout_pct:.0f}%, policy transfer={policy_pct:.0f}%"
    )

print("=" * 70)

with open("output/results/rollout_ablation/all_results.json", "w") as f:
    json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
