"""Run Sinergym's official RBC5Zone baseline on all 3 5zone buildings.

Sinergym's built-in RBC5Zone (sinergym.utils.controllers.RBC5Zone) is
an ASHRAE Standard 55-2004 based rule-based controller that outputs
comfort-band-edge setpoints:
    Summer (Jun-Sep): (heating_sp=23, cooling_sp=26)  — upper edge for power saving
    Winter (Oct-May): (heating_sp=20, cooling_sp=23.5)

This is a MUCH stronger baseline than the previous constant-midpoint RBC.
It is reward-aware (matches Sinergym's LinearReward comfort band) and
approximately optimal within the setpoint-control action space.

Results saved to output/results/rbc5zone_baseline/rbc5zone_results.json.
"""

import json
from pathlib import Path

import gymnasium
from loguru import logger

import sinergym  # noqa: F401 — register envs
from sinergym.utils.controllers import RBC5Zone

BUILDINGS = {
    "5zone_hot": "Eplus-5zone-hot-continuous-v1",
    "5zone_mixed": "Eplus-5zone-mixed-continuous-v1",
    "5zone_cool": "Eplus-5zone-cool-continuous-v1",
}

SAVE_DIR = Path("output/results/rbc5zone_baseline")


def run_rbc(env_name: str, seed: int = 42) -> float:
    """Run RBC5Zone for one full episode, return total reward."""
    env = gymnasium.make(env_name)
    rbc = RBC5Zone(env.unwrapped)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    done = False
    while not done:
        action = rbc.act(obs)
        obs, reward, term, trunc, _ = env.step(action)
        total += float(reward)
        done = term or trunc
    env.close()
    return total


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, float] = {}

    for bid, env_name in BUILDINGS.items():
        logger.info(f"Running RBC5Zone on {bid}...")
        reward = run_rbc(env_name, seed=42)
        results[bid] = reward
        logger.info(f"  {bid}: reward = {reward:.1f}")

    with open(SAVE_DIR / "rbc5zone_results.json", "w") as f:
        json.dump(
            {
                "controller": "sinergym.utils.controllers.RBC5Zone",
                "basis": "ASHRAE Standard 55-2004",
                "setpoints_summer": [23.0, 26.0],
                "setpoints_winter": [20.0, 23.5],
                "seed": 42,
                "n_episodes": 1,
                "rewards": results,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 60)
    print("SINERGYM OFFICIAL RBC5Zone BASELINE")
    print("=" * 60)
    for bid, r in results.items():
        print(f"  {bid:<20s}: {r:>10.1f}")
    print(f"\nSaved to {SAVE_DIR / 'rbc5zone_results.json'}")


if __name__ == "__main__":
    main()
