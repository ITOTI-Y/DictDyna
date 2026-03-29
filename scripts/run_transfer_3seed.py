"""3-seed transfer experiment: freeze-dict + full-data context inference."""

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
    experiment = FewShotTransferExperiment(
        source_configs=SOURCE,
        target_config=TARGET,
        dict_path=DICT_PATH,
        config=config,
        adaptation_days=DAYS,
        seed=seed,
        save_dir=f"output/results/verify_transfer_3seed/s{seed}",
        context_mode=True,
    )
    results = experiment.run()
    all_results[seed] = results
    for k, v in results.items():
        print(f"  [s{seed}] {k}: {v:.1f}")

# Aggregate
print("\n" + "=" * 70)
print("3-SEED SUMMARY (freeze-dict + full-data context)")
print("=" * 70)

for d in DAYS:
    ts = [all_results[s][f"transfer_{d}d"] for s in SEEDS]
    ss = [all_results[s][f"scratch_{d}d"] for s in SEEDS]
    advs = [(t - s) / abs(s) * 100 for t, s in zip(ts, ss, strict=True)]
    tm, tstd = np.mean(ts), np.std(ts)
    sm, sstd = np.mean(ss), np.std(ss)
    am, astd = np.mean(advs), np.std(advs)
    print(
        f"  {d}d: Transfer={tm:.0f}+/-{tstd:.0f}  "
        f"Scratch={sm:.0f}+/-{sstd:.0f}  "
        f"Advantage={am:+.1f}+/-{astd:.1f}%"
    )
    for i, s in enumerate(SEEDS):
        print(f"      s{s}: T={ts[i]:.0f}  S={ss[i]:.0f}  adv={advs[i]:+.1f}%")

# Monotonicity check
print("\nMonotonicity (transfer reward, higher=better):")
for s in SEEDS:
    t1 = all_results[s]["transfer_1d"]
    t3 = all_results[s]["transfer_3d"]
    t7 = all_results[s]["transfer_7d"]
    ok = "OK" if t7 >= t1 else "FAIL"
    print(f"  s{s}: 1d={t1:.0f} -> 3d={t3:.0f} -> 7d={t7:.0f}  [{ok}]")

print("=" * 70)

with open("output/results/verify_transfer_3seed/all_results.json", "w") as f:
    json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
