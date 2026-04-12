"""T1.3: Reward surface analysis across buildings.

Tests H2: The optimal HVAC policy is approximately invariant to climate —
reward = f(indoor_temp, HVAC_power) has the same structure regardless of
outdoor conditions.

Visualizes the reward as a function of indoor temperature and HVAC power
for representative states from each building, showing the optimal region
is consistent across climates.

Outputs:
    figures/reward_surface.png — Reward heatmap (indoor_temp × HVAC_power)
                                 for each building across seasons
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from _share import (
    BUILDING_COLORS,
    BUILDING_LABELS,
    BUILDINGS,
    FIG_DIR,
    ensure_dirs,
    load_building_transitions,
    load_pretrained_dict,
    normalize_states_fast,
    setup_figure,
)

from src.world_model.reward_estimator import SinergymRewardEstimator


def main():
    setup_figure()
    ensure_dirs()

    pretrained = load_pretrained_dict()
    obs_mean_t = torch.tensor(pretrained["obs_mean"], dtype=torch.float32)
    obs_std_t = torch.tensor(pretrained["obs_std"], dtype=torch.float32)

    estimator = SinergymRewardEstimator(obs_mean=obs_mean_t, obs_std=obs_std_t)

    # Load real states from each building to get representative obs
    all_raw: dict[str, np.ndarray] = {}
    for bid in BUILDINGS:
        trans = load_building_transitions(bid)
        all_raw[bid] = trans["states"]  # raw, unnormalized

    # Sweep indoor_temp (dim 9) and HVAC_power (dim 15) to show reward surface.
    # Fix all other dims to building medians for summer/winter.
    temp_range = np.linspace(15.0, 35.0, 50)
    power_range = np.linspace(0.0, 15000.0, 50)
    temp_grid, power_grid = np.meshgrid(temp_range, power_range)

    seasons = {
        "Winter (Jan)": lambda s: s[:, 0] == 1,
        "Summer (Jul)": lambda s: s[:, 0] == 7,
    }

    fig, axes = plt.subplots(
        len(seasons), len(BUILDINGS),
        figsize=(9, 5),
        sharex=True, sharey=True,
    )

    for row, (season_name, season_mask_fn) in enumerate(seasons.items()):
        for col, bid in enumerate(BUILDINGS):
            ax = axes[row, col]
            raw = all_raw[bid]

            # Get median state for this season
            mask = season_mask_fn(raw)
            if mask.sum() < 10:
                ax.set_title(f"{BUILDING_LABELS[bid]} - {season_name}\n(no data)")
                continue

            median_state = np.median(raw[mask], axis=0).copy()

            # Build grid of states varying indoor_temp and HVAC_power
            N = temp_grid.size
            states = np.tile(median_state, (N, 1)).astype(np.float32)
            states[:, 9] = temp_grid.ravel()   # indoor_temp
            states[:, 15] = power_grid.ravel()  # HVAC_power

            # Normalize with source stats (what the reward estimator expects)
            states_norm = normalize_states_fast(
                states, pretrained["obs_mean"], pretrained["obs_std"]
            )
            states_t = torch.tensor(states_norm)

            with torch.no_grad():
                rewards = estimator.estimate(states_t).numpy()

            rewards_grid = rewards.reshape(temp_grid.shape)

            im = ax.contourf(
                temp_range, power_range, rewards_grid,
                levels=20, cmap="RdYlGn",
            )
            ax.set_title(
                f"{BUILDING_LABELS[bid]} - {season_name}",
                fontsize=8, color=BUILDING_COLORS[bid],
            )

            if col == 0:
                ax.set_ylabel("HVAC Power (W)")
            if row == len(seasons) - 1:
                ax.set_xlabel("Indoor Temp (°C)")

    fig.suptitle(
        "Reward Surface: R(indoor_temp, HVAC_power)\nHigher (green) = better",
        fontsize=10,
    )
    fig.colorbar(im, ax=axes, label="Reward", shrink=0.8)
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    fig.savefig(FIG_DIR / "reward_surface.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'reward_surface.png'}")

    # === Summary: optimal indoor_temp per building/season ===
    print("\n=== Reward Surface Summary ===")
    print("Optimal indoor temp (max reward at zero power):")
    for bid in BUILDINGS:
        raw = all_raw[bid]
        for season_name, mask_fn in seasons.items():
            mask = mask_fn(raw)
            if mask.sum() < 10:
                continue
            median_state = np.median(raw[mask], axis=0).copy()

            # Sweep temp only, fix power = 0
            states = np.tile(median_state, (len(temp_range), 1)).astype(np.float32)
            states[:, 9] = temp_range
            states[:, 15] = 0.0  # zero power
            states_norm = normalize_states_fast(
                states, pretrained["obs_mean"], pretrained["obs_std"]
            )
            with torch.no_grad():
                r = estimator.estimate(torch.tensor(states_norm)).numpy()
            best_temp = temp_range[np.argmax(r)]
            print(f"  {BUILDING_LABELS[bid]:>6s} {season_name}: {best_temp:.1f}°C")


if __name__ == "__main__":
    main()
