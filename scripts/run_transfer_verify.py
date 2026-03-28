"""Verify refactored code with full multi-building transfer experiment.

Runs context-conditioned transfer: source (hot+mixed) → target (cool)
with 1/3/7 day adaptation budgets, single seed.
"""

import json

from loguru import logger

from src.agent.transfer_experiment import FewShotTransferExperiment
from src.schemas import TrainSchema

SEED = 42
DICT_PATH = "output/pretrained/dict_k128_source_only.pt"
SAVE_DIR = "output/results/verify_transfer_context"

SOURCE_BUILDINGS = [
    {"env_name": "Eplus-5zone-hot-continuous-v1", "building_id": "office_hot"},
    {"env_name": "Eplus-5zone-mixed-continuous-v1", "building_id": "office_mixed"},
]
TARGET_BUILDING = {
    "env_name": "Eplus-5zone-cool-continuous-v1",
    "building_id": "office_cool",
}

config = TrainSchema(
    mode="context",
    seed=SEED,
    total_timesteps=35040,
    device="auto",
)

logger.info(f"Transfer experiment: mode={config.mode}, seed={SEED}")
logger.info(f"Source: {[b['building_id'] for b in SOURCE_BUILDINGS]}")
logger.info(f"Target: {TARGET_BUILDING['building_id']}")

experiment = FewShotTransferExperiment(
    source_configs=SOURCE_BUILDINGS,
    target_config=TARGET_BUILDING,
    dict_path=DICT_PATH,
    config=config,
    adaptation_days=[1, 3, 7],
    seed=SEED,
    save_dir=SAVE_DIR,
    context_mode=True,
)
results = experiment.run()

print("\n" + "=" * 50)
print("TRANSFER VERIFICATION RESULTS")
print("=" * 50)
for k, v in results.items():
    print(f"  {k}: {v:.1f}")

# Compute advantages
for days in [1, 3, 7]:
    t = results[f"transfer_{days}d"]
    s = results[f"scratch_{days}d"]
    adv = (t - s) / abs(s) * 100
    print(f"  {days}d advantage: {adv:+.1f}%")

print("\nWorld model mode: context")
print("=" * 50)

# Save summary
with open(f"{SAVE_DIR}/summary.json", "w") as f:
    summary = dict(results)
    for days in [1, 3, 7]:
        t = results[f"transfer_{days}d"]
        s = results[f"scratch_{days}d"]
        summary[f"advantage_{days}d"] = (t - s) / abs(s) * 100
    json.dump(summary, f, indent=2)
