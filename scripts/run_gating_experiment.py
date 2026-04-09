"""Context-to-Sparse Gating experiment: compare gating vs no-gating.

Runs context transfer with and without context gating across 3 seeds x 3 budgets.
Also includes zero-shot variants to test gating's effect on inference-only adaptation.

Usage:
    uv run python scripts/run_gating_experiment.py
    uv run python scripts/run_gating_experiment.py --seeds 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger

from src.agent.transfer_experiment import FewShotTransferExperiment
from src.schemas import ContextEncoderSchema, TrainSchema

DICT_PATH = "output/pretrained/dict_k128_source_only.pt"
SOURCE = [
    {"env_name": "Eplus-5zone-hot-continuous-v1", "building_id": "office_hot"},
    {"env_name": "Eplus-5zone-mixed-continuous-v1", "building_id": "office_mixed"},
]
TARGET = {"env_name": "Eplus-5zone-cool-continuous-v1", "building_id": "office_cool"}
DAYS = [1, 3, 7]
DEFAULT_SEEDS = [42, 123, 7]

# Experiment conditions: (label, use_context_gating, fine_tune)
CONDITIONS = [
    ("scratch", None, None),  # baseline
    ("context", False, True),  # current default
    ("context_gated", True, True),  # with gating
    ("zeroshot", False, False),  # zero-shot, no gating
    ("zeroshot_gated", True, False),  # zero-shot, with gating
]


def main():
    parser = argparse.ArgumentParser(description="Context gating experiment")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds"
    )
    parser.add_argument(
        "--save-dir",
        default="output/results/gating_experiment",
        help="Output directory",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[int, dict] = {}

    for seed in args.seeds:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"SEED {seed}")
        logger.info(f"{'=' * 60}")

        seed_results: dict[str, float] = {}

        for label, use_gating, fine_tune in CONDITIONS:
            logger.info(f"\n--- Condition: {label} ---")

            if label == "scratch":
                # Run scratch baseline (no gating relevant)
                config = TrainSchema(
                    mode="context", seed=seed, total_timesteps=35040, device="auto"
                )
                experiment = FewShotTransferExperiment(
                    source_configs=SOURCE,
                    target_config=TARGET,
                    dict_path=DICT_PATH,
                    config=config,
                    adaptation_days=DAYS,
                    seed=seed,
                    save_dir=str(save_dir / f"s{seed}" / label),
                    context_mode=True,
                )
                results = experiment.run_ablation(conditions=["scratch"])
                for k, v in results.items():
                    seed_results[k] = v
                continue

            # Context / zero-shot conditions with or without gating
            ctx_schema = ContextEncoderSchema(use_context_gating=use_gating)
            config = TrainSchema(
                mode="context",
                seed=seed,
                total_timesteps=35040,
                device="auto",
                context=ctx_schema,
            )
            experiment = FewShotTransferExperiment(
                source_configs=SOURCE,
                target_config=TARGET,
                dict_path=DICT_PATH,
                config=config,
                adaptation_days=DAYS,
                seed=seed,
                save_dir=str(save_dir / f"s{seed}" / label),
                context_mode=True,
            )

            # Use run_ablation with appropriate condition
            cond = "context" if fine_tune else "zero_shot"
            results = experiment.run_ablation(conditions=[cond])

            # Rename keys to include label
            for k, v in results.items():
                # "context_1d" -> "context_gated_1d" etc.
                new_key = k.replace(cond, label)
                seed_results[new_key] = v

        all_results[seed] = seed_results

    # === Summary ===
    labels = [label for label, _, _ in CONDITIONS]
    print(f"\n{'=' * 80}")
    print("GATING EXPERIMENT SUMMARY")
    print(f"{'=' * 80}")

    for d in DAYS:
        print(f"\n  {d}-day:")
        scratch_vals = [
            all_results[s].get(f"scratch_{d}d", float("nan")) for s in args.seeds
        ]
        sm = np.nanmean(scratch_vals)
        print(f"    scratch:         {sm:>8.0f}")

        for label in labels:
            if label == "scratch":
                continue
            vals = [
                all_results[s].get(f"{label}_{d}d", float("nan")) for s in args.seeds
            ]
            vm = np.nanmean(vals)
            vstd = np.nanstd(vals)
            advs = [
                (v - sc) / abs(sc) * 100 if sc != 0 else float("nan")
                for v, sc in zip(vals, scratch_vals, strict=True)
            ]
            am = np.nanmean(advs)
            print(f"    {label:<18s} {vm:>7.0f}+/-{vstd:<5.0f}  adv={am:>+5.1f}%")

    output = {
        "seeds": args.seeds,
        "conditions": labels,
        "days": DAYS,
        "per_seed": {str(k): v for k, v in all_results.items()},
    }
    output_path = save_dir / "all_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
