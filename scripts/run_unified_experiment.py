"""Unified transfer experiment: shared source + target across all conditions.

Runs ALL transfer conditions (scratch, pure_zero_shot, no_encoder_ft, context,
and their gated variants) in ONE workflow per seed, sharing:
  - Target data collection (same raw data used by every condition)
  - Source training for {non-gated conditions}
  - Source training for {gated conditions}

This ensures that same-seed comparisons across conditions are actually fair —
the previous split scripts created different source models and collected
different target data, contaminating the comparison.

Workflow per seed:
    Phase 1a: Train ungated source → S_ungated
    Phase 1b: Train gated source → S_gated
    Phase 2:  Collect target data once → T
    Phase 3:  Run all conditions:
        - scratch (no source, uses T)
        - pure_zero_shot (S_ungated, T)
        - pure_zero_shot_gated (S_gated, T)
        - no_encoder_ft (S_ungated, T, 1d/3d/7d)
        - no_encoder_ft_gated (S_gated, T, 1d/3d/7d)
        - context (S_ungated, T, 1d/3d/7d)
        - context_gated (S_gated, T, 1d/3d/7d)

Usage:
    uv run python scripts/run_unified_experiment.py
    uv run python scripts/run_unified_experiment.py --seeds 42
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

# Non-gated conditions (use S_ungated)
UNGATED_CONDITIONS = ["scratch", "pure_zero_shot", "no_encoder_ft", "context"]
# Gated conditions (use S_gated, scratch not repeated)
GATED_CONDITIONS = ["pure_zero_shot", "no_encoder_ft", "context"]


def _make_experiment(
    use_gating: bool,
    seed: int,
    save_dir: Path,
    suffix: str,
) -> FewShotTransferExperiment:
    config = TrainSchema(
        mode="context",
        seed=seed,
        total_timesteps=35040,
        device="auto",
        context=ContextEncoderSchema(use_context_gating=use_gating),
    )
    return FewShotTransferExperiment(
        source_configs=SOURCE,
        target_config=TARGET,
        dict_path=DICT_PATH,
        config=config,
        adaptation_days=DAYS,
        seed=seed,
        save_dir=str(save_dir / f"s{seed}" / suffix),
        context_mode=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Unified transfer experiment")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds"
    )
    parser.add_argument(
        "--save-dir",
        default="output/results/unified_experiment",
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

        # ---- Phase 1a + 2: Ungated source + shared target data ----
        logger.info(f"\n--- [Seed {seed}] Ungated source group ---")
        exp_ungated = _make_experiment(
            use_gating=False, seed=seed, save_dir=save_dir, suffix="ungated"
        )
        source_ungated = exp_ungated._train_source()
        target_data = exp_ungated._collect_target_data()

        # Run ungated conditions
        ungated_results = exp_ungated.run_ablation(
            conditions=UNGATED_CONDITIONS,
            target_data=target_data,
            source_dyna=source_ungated,
        )

        # ---- Phase 1b: Gated source (same seed, different config) ----
        logger.info(f"\n--- [Seed {seed}] Gated source group ---")
        exp_gated = _make_experiment(
            use_gating=True, seed=seed, save_dir=save_dir, suffix="gated"
        )
        source_gated = exp_gated._train_source()

        # Run gated conditions using SAME target_data
        gated_results_raw = exp_gated.run_ablation(
            conditions=GATED_CONDITIONS,
            target_data=target_data,
            source_dyna=source_gated,
        )

        # Rename gated keys: e.g. "context_1d" → "context_gated_1d"
        gated_results: dict[str, float] = {}
        for k, v in gated_results_raw.items():
            # Split at last underscore to get condition and budget
            parts = k.rsplit(
                "_", 1
            )  # e.g. ["context", "1d"] or ["pure_zero_shot", "1d"]
            if len(parts) == 2:
                cond, budget = parts
                gated_results[f"{cond}_gated_{budget}"] = v
            else:
                gated_results[k] = v

        # Merge
        seed_results = {**ungated_results, **gated_results}
        all_results[seed] = seed_results

        logger.info(f"\n[Seed {seed}] Conditions collected: {len(seed_results)}")

    # === Summary ===
    print(f"\n{'=' * 100}")
    print(f"UNIFIED EXPERIMENT SUMMARY | Seeds: {args.seeds}")
    print(f"{'=' * 100}")

    all_conditions = [
        "scratch",
        "pure_zero_shot",
        "pure_zero_shot_gated",
        "no_encoder_ft",
        "no_encoder_ft_gated",
        "context",
        "context_gated",
    ]

    for d in DAYS:
        print(f"\n  {d}-day budget:")
        scratch_vals = [
            all_results[s].get(f"scratch_{d}d", float("nan")) for s in args.seeds
        ]
        sm, sstd = np.nanmean(scratch_vals), np.nanstd(scratch_vals)
        print(f"    {'scratch':<22s} {sm:>7.0f} ± {sstd:<5.0f}")

        for cond in all_conditions:
            if cond == "scratch":
                continue
            vals = [
                all_results[s].get(f"{cond}_{d}d", float("nan")) for s in args.seeds
            ]
            vm = np.nanmean(vals)
            vstd = np.nanstd(vals)
            advs = [
                (v - sc) / abs(sc) * 100 if sc != 0 else float("nan")
                for v, sc in zip(vals, scratch_vals, strict=True)
            ]
            am = np.nanmean(advs)
            astd = np.nanstd(advs)
            print(
                f"    {cond:<22s} {vm:>7.0f} ± {vstd:<5.0f}  "
                f"adv={am:>+5.1f} ± {astd:<4.1f}%"
            )

    # === Per-seed detail ===
    print(f"\n{'=' * 100}")
    print("PER-SEED DETAIL")
    print(f"{'=' * 100}")
    for s in args.seeds:
        print(f"\n  Seed {s}:")
        for d in DAYS:
            parts = []
            for cond in all_conditions:
                key = f"{cond}_{d}d"
                val = all_results[s].get(key, float("nan"))
                parts.append(
                    f"{cond.split('_')[0][:4]}{'_g' if 'gated' in cond else ''}={val:.0f}"
                )
            print(f"    {d}d: {', '.join(parts)}")

    output = {
        "seeds": args.seeds,
        "days": DAYS,
        "conditions": all_conditions,
        "per_seed": {str(k): v for k, v in all_results.items()},
    }
    output_path = save_dir / "all_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
