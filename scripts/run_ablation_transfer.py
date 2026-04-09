"""Transfer ablation experiment: scratch vs zero-shot vs context vs adapter.

Runs all conditions across 3 seeds x 3 budgets (1d/3d/7d).
Results are saved per-seed and aggregated into a summary table.

Usage:
    uv run python scripts/run_ablation_transfer.py
    uv run python scripts/run_ablation_transfer.py --conditions scratch zero_shot context
    uv run python scripts/run_ablation_transfer.py --seeds 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger

from src.agent.transfer_experiment import FewShotTransferExperiment
from src.schemas import TrainSchema

DICT_PATH = "output/pretrained/dict_k128_source_only.pt"
SOURCE = [
    {"env_name": "Eplus-5zone-hot-continuous-v1", "building_id": "office_hot"},
    {"env_name": "Eplus-5zone-mixed-continuous-v1", "building_id": "office_mixed"},
]
TARGET = {"env_name": "Eplus-5zone-cool-continuous-v1", "building_id": "office_cool"}
DAYS = [1, 3, 7]
DEFAULT_SEEDS = [42, 123, 7]
ALL_CONDITIONS = ["scratch", "zero_shot", "context", "adapter"]


def main():
    parser = argparse.ArgumentParser(description="Transfer ablation experiment")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=ALL_CONDITIONS,
        choices=ALL_CONDITIONS,
        help="Conditions to run",
    )
    parser.add_argument(
        "--save-dir",
        default="output/results/ablation_transfer",
        help="Output directory",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[int, dict] = {}

    for seed in args.seeds:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"SEED {seed} | Conditions: {args.conditions}")
        logger.info(f"{'=' * 60}")

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
            save_dir=str(save_dir / f"s{seed}"),
            context_mode=True,
        )
        results = experiment.run_ablation(conditions=args.conditions)
        all_results[seed] = results

    # === Summary table ===
    print(f"\n{'=' * 80}")
    print(f"ABLATION SUMMARY | Seeds: {args.seeds} | Conditions: {args.conditions}")
    print(f"{'=' * 80}")

    # Header
    conds = [c for c in args.conditions if c != "scratch"]
    header = f"{'Days':>4s} | {'Scratch':>14s}"
    for c in conds:
        header += f" | {c:>14s} | {'Adv%':>8s}"
    print(header)
    print("-" * len(header))

    for d in DAYS:
        scratch_vals = [
            all_results[s].get(f"scratch_{d}d", float("nan")) for s in args.seeds
        ]
        sm, sstd = np.nanmean(scratch_vals), np.nanstd(scratch_vals)
        row = f"{d:>3d}d | {sm:>7.0f}±{sstd:<5.0f}"

        for c in conds:
            vals = [all_results[s].get(f"{c}_{d}d", float("nan")) for s in args.seeds]
            vm, vstd = np.nanmean(vals), np.nanstd(vals)
            advs = [
                (v - sc) / abs(sc) * 100 if sc != 0 else float("nan")
                for v, sc in zip(vals, scratch_vals, strict=True)
            ]
            am, astd = np.nanmean(advs), np.nanstd(advs)
            row += f" | {vm:>7.0f}±{vstd:<5.0f} | {am:>+5.1f}±{astd:<3.1f}"

        print(row)

    # Per-seed detail
    print(f"\n{'=' * 80}")
    print("PER-SEED DETAIL")
    print(f"{'=' * 80}")

    for s in args.seeds:
        print(f"\n  Seed {s}:")
        for d in DAYS:
            parts = []
            for c in args.conditions:
                key = f"{c}_{d}d"
                val = all_results[s].get(key, float("nan"))
                parts.append(f"{c}={val:.0f}")
            print(f"    {d}d: {', '.join(parts)}")

    # Save
    output = {
        "seeds": args.seeds,
        "conditions": args.conditions,
        "days": DAYS,
        "per_seed": {str(k): v for k, v in all_results.items()},
    }
    output_path = save_dir / "all_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
