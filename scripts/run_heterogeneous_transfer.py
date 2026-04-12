"""Heterogeneous building transfer experiment.

Tests cross-building-type transfer using UniversalObsEncoder:
  Source: 5zone office (hot+mixed, 17d obs)
  Target: warehouse (22d obs), shop (34d obs), 5zone-cool (17d, baseline)

Usage:
    # In Docker:
    python scripts/run_heterogeneous_transfer.py --seeds 42
    python scripts/run_heterogeneous_transfer.py --seeds 42 123 7
"""

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger

from src.agent.universal_transfer import UniversalTransferExperiment

SOURCE = [
    {"env_name": "Eplus-5zone-hot-continuous-v1", "building_id": "5zone_hot"},
    {"env_name": "Eplus-5zone-mixed-continuous-v1", "building_id": "5zone_mixed"},
]

TARGETS = [
    {"env_name": "Eplus-5zone-cool-continuous-v1", "building_id": "5zone_cool"},
    {"env_name": "Eplus-warehouse-hot-continuous-v1", "building_id": "warehouse_hot"},
    {"env_name": "Eplus-shop-hot-continuous-v1", "building_id": "shop_hot"},
]

DEFAULT_SEEDS = [42, 123, 7]


def main():
    parser = argparse.ArgumentParser(description="Heterogeneous transfer experiment")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds"
    )
    parser.add_argument(
        "--save-dir",
        default="output/results/heterogeneous_transfer",
        help="Output directory",
    )
    parser.add_argument(
        "--timesteps", type=int, default=35040, help="Training steps per source"
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[int, dict] = {}

    for seed in args.seeds:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"SEED {seed}")
        logger.info(f"{'=' * 60}")

        experiment = UniversalTransferExperiment(
            source_configs=SOURCE,
            target_configs=TARGETS,
            seed=seed,
            total_timesteps=args.timesteps,
            save_dir=str(save_dir / f"s{seed}"),
            device="auto",
        )
        results = experiment.run()
        all_results[seed] = results

    # Summary
    print(f"\n{'=' * 70}")
    print("HETEROGENEOUS TRANSFER SUMMARY")
    print(f"{'=' * 70}")

    target_ids = [t["building_id"] for t in TARGETS]
    for tid in target_ids:
        zs_vals = [
            all_results[s].get(f"zero_shot_{tid}", float("nan")) for s in args.seeds
        ]
        sc_vals = [
            all_results[s].get(f"scratch_{tid}", float("nan")) for s in args.seeds
        ]
        adv_vals = [
            all_results[s].get(f"advantage_{tid}", float("nan")) for s in args.seeds
        ]

        zm, zstd = np.nanmean(zs_vals), np.nanstd(zs_vals)
        sm, sstd = np.nanmean(sc_vals), np.nanstd(sc_vals)
        am, astd = np.nanmean(adv_vals), np.nanstd(adv_vals)

        print(f"\n  {tid}:")
        print(f"    Zero-shot: {zm:>8.0f} +/- {zstd:<6.0f}")
        print(f"    Scratch:   {sm:>8.0f} +/- {sstd:<6.0f}")
        print(f"    Advantage: {am:>+7.1f} +/- {astd:<5.1f}%")

    output = {
        "seeds": args.seeds,
        "source": [s["building_id"] for s in SOURCE],
        "targets": target_ids,
        "per_seed": {str(k): v for k, v in all_results.items()},
    }
    output_path = save_dir / "all_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
