"""Verify whether the source-trained actor learned f1 (climate-invariant)
or f2 (climate-dependent) policy.

Critical question raised 2026-04-14: Sinergym's reward is building-independent
(depends only on indoor_temp, HVAC_power, and month-based comfort band).
The optimal setpoint policy is the SAME across hot/mixed/cool climates
(T1.3 confirmed: Winter 20.3°C, Summer 23.2°C for all 3 buildings).

This means "+67% pure zero-shot transfer" observed in Phase 7 might NOT
be cross-climate knowledge transfer — it could just be source having more
training steps than scratch, with the policy itself being climate-invariant.

Tests:
1. **f1 test**: Given same (month, hour), are actions the same across buildings?
   If yes → f1 (climate-invariant), transfer is trivial
2. **Counterfactual test**: Fix all other obs, vary outdoor_temp, does action change?
   If no → f1; If yes → f2 (climate-dependent)
3. **Within-month analysis**: Within fixed month, does action correlate with outdoor_temp?
   If no correlation → f1

Output:
    figures/actor_action_by_month.png  — per-month action distributions (3 buildings)
    figures/actor_action_vs_temp.png   — action vs outdoor_temp scatter
    figures/actor_counterfactual.png   — counterfactual outdoor_temp sweep
    tables/actor_f1_f2_stats.csv       — summary statistics
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from _share import (
    BUILDING_COLORS,
    BUILDING_LABELS,
    BUILDINGS,
    FIG_DIR,
    TABLE_DIR,
    ensure_dirs,
    load_building_transitions,
    setup_figure,
)

CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "output/results/dyna_low_ratio/checkpoint_step105120.pt"
)


class ReplayActor(nn.Module):
    """Replay the stored actor for deterministic action computation.

    Matches the GaussianActor architecture used in dyna_low_ratio checkpoint:
      trunk: Linear(17, 256) → ReLU → Linear(256, 256) → ReLU
      mean_head: Linear(256, 2)
    """

    def __init__(self, actor_state: dict):
        super().__init__()
        # Move all tensors to CPU (checkpoint may be on CUDA)
        self.action_scale = actor_state["action_scale"].cpu()
        self.action_bias = actor_state["action_bias"].cpu()
        self.obs_mean = actor_state["obs_norm.mean"].cpu()
        self.obs_std = actor_state["obs_norm.std"].cpu()

        self.trunk = nn.Sequential(
            nn.Linear(17, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.trunk[0].weight.data = actor_state["trunk.0.weight"].cpu()
        self.trunk[0].bias.data = actor_state["trunk.0.bias"].cpu()
        self.trunk[2].weight.data = actor_state["trunk.2.weight"].cpu()
        self.trunk[2].bias.data = actor_state["trunk.2.bias"].cpu()

        self.mean_head = nn.Linear(256, 2)
        self.mean_head.weight.data = actor_state["mean_head.weight"].cpu()
        self.mean_head.bias.data = actor_state["mean_head.bias"].cpu()

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action (mean, tanh-squashed and scaled)."""
        if obs.ndim == 1:
            obs = obs[None]
        obs_t = torch.tensor(obs, dtype=torch.float32)
        # Normalize
        normed = torch.clamp(
            (obs_t - self.obs_mean) / self.obs_std, -10.0, 10.0
        )
        h = self.trunk(normed)
        mean = self.mean_head(h)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action.numpy()


def main():
    setup_figure()
    ensure_dirs()

    # Load actor
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
    actor = ReplayActor(ckpt["actor"])

    # Process all 3 buildings
    all_obs: dict[str, np.ndarray] = {}
    all_actions: dict[str, np.ndarray] = {}
    for bid in BUILDINGS:
        trans = load_building_transitions(bid)
        obs = trans["states"]
        actions = actor.act(obs)
        all_obs[bid] = obs
        all_actions[bid] = actions
        print(f"{bid}: obs shape={obs.shape}, actions shape={actions.shape}")
        print(f"  heating_sp mean={actions[:, 0].mean():.2f} std={actions[:, 0].std():.3f}")
        print(f"  cooling_sp mean={actions[:, 1].mean():.2f} std={actions[:, 1].std():.3f}")

    # === Figure: Action vs Month (per building) ===
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for act_idx, act_name in enumerate(["heating_setpoint", "cooling_setpoint"]):
        ax = axes[act_idx]
        for bid in BUILDINGS:
            obs = all_obs[bid]
            actions = all_actions[bid]
            months = obs[:, 0].astype(int)  # dim 0 = month
            month_means = []
            month_stds = []
            month_x = sorted(set(months))
            for m in month_x:
                mask = months == m
                month_means.append(actions[mask, act_idx].mean())
                month_stds.append(actions[mask, act_idx].std())
            month_means = np.array(month_means)
            month_stds = np.array(month_stds)
            ax.errorbar(
                month_x, month_means, yerr=month_stds,
                label=BUILDING_LABELS[bid], color=BUILDING_COLORS[bid],
                capsize=3, marker="o", markersize=4,
            )
        ax.set_xlabel("Month")
        ax.set_ylabel(act_name)
        ax.legend(fontsize=7)
        ax.set_title(f"Action ({act_name}) by Month — source actor (trained on hot)")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "actor_action_by_month.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'actor_action_by_month.png'}")

    # === Figure: Action vs Outdoor Temp (within-month correlation) ===
    fig2, axes2 = plt.subplots(2, 3, figsize=(11, 6))

    for col_idx, bid in enumerate(BUILDINGS):
        obs = all_obs[bid]
        actions = all_actions[bid]
        outdoor_t = obs[:, 3]  # dim 3 = outdoor_temperature

        for row_idx, act_name in enumerate(["heating_sp", "cooling_sp"]):
            ax = axes2[row_idx, col_idx]
            # Subsample for readability
            rng = np.random.default_rng(42)
            idx = rng.choice(len(obs), size=2000, replace=False)
            ax.scatter(
                outdoor_t[idx], actions[idx, row_idx],
                c=BUILDING_COLORS[bid], alpha=0.2, s=3,
            )
            ax.set_xlabel("Outdoor Temp (°C)", fontsize=8)
            ax.set_ylabel(act_name, fontsize=8)
            corr = np.corrcoef(outdoor_t, actions[:, row_idx])[0, 1]
            ax.set_title(
                f"{BUILDING_LABELS[bid]} — corr={corr:.3f}", fontsize=8
            )
            ax.tick_params(labelsize=6)

    fig2.suptitle("Action vs Outdoor Temperature by Building")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "actor_action_vs_temp.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {FIG_DIR / 'actor_action_vs_temp.png'}")

    # === Counterfactual Test ===
    # Take a cool-building obs, vary outdoor_temp from -20 to 40, keep all else
    cool_obs = all_obs["office_cool"]
    # Pick a representative "typical summer noon" obs
    rng = np.random.default_rng(0)
    summer_noon_idx = np.where(
        (cool_obs[:, 0] == 7) & (cool_obs[:, 2] == 12)  # July, noon
    )[0]
    winter_noon_idx = np.where(
        (cool_obs[:, 0] == 1) & (cool_obs[:, 2] == 12)  # January, noon
    )[0]
    if len(summer_noon_idx) == 0 or len(winter_noon_idx) == 0:
        print("Warning: no summer/winter noon samples, using arbitrary")
        summer_ref = cool_obs[0].copy()
        winter_ref = cool_obs[1].copy()
    else:
        summer_ref = cool_obs[summer_noon_idx[0]].copy()
        winter_ref = cool_obs[winter_noon_idx[0]].copy()

    temp_range = np.linspace(-20, 40, 50)

    fig3, axes3 = plt.subplots(2, 2, figsize=(9, 6))
    for row_idx, (label, ref_obs) in enumerate(
        [("Winter Noon", winter_ref), ("Summer Noon", summer_ref)]
    ):
        for col_idx, act_name in enumerate(["heating_setpoint", "cooling_setpoint"]):
            ax = axes3[row_idx, col_idx]
            actions_list = []
            for t in temp_range:
                test_obs = ref_obs.copy()
                test_obs[3] = t  # outdoor_temp
                a = actor.act(test_obs[None])[0]
                actions_list.append(a[col_idx])
            ax.plot(temp_range, actions_list, linewidth=2, color="#1f77b4")
            ax.axvline(
                ref_obs[3], color="red", linestyle="--", alpha=0.5,
                label=f"real temp = {ref_obs[3]:.1f}",
            )
            ax.set_xlabel("Outdoor Temp (°C)")
            ax.set_ylabel(act_name)
            ax.set_title(f"{label} — {act_name} sensitivity")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

    fig3.suptitle(
        "Counterfactual: Fix all obs except outdoor_temp on cool-building obs\n"
        "Flat line → f1 (climate-invariant); Sloped → f2 (climate-dependent)"
    )
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / "actor_counterfactual.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved {FIG_DIR / 'actor_counterfactual.png'}")

    # === Summary Statistics Table ===
    rows = []
    for bid in BUILDINGS:
        obs = all_obs[bid]
        actions = all_actions[bid]
        outdoor_t = obs[:, 3]
        months = obs[:, 0].astype(int)

        for act_idx, act_name in enumerate(["heating_sp", "cooling_sp"]):
            overall_std = actions[:, act_idx].std()
            # Compute within-month std (average across months)
            within_month_std = []
            for m in sorted(set(months)):
                mask = months == m
                if mask.sum() > 10:
                    within_month_std.append(actions[mask, act_idx].std())
            avg_within_month_std = np.mean(within_month_std)
            # Correlation with outdoor_temp
            corr_outdoor = np.corrcoef(outdoor_t, actions[:, act_idx])[0, 1]
            # Within-month correlation with outdoor_temp
            within_corrs = []
            for m in sorted(set(months)):
                mask = months == m
                if mask.sum() > 10:
                    c = np.corrcoef(outdoor_t[mask], actions[mask, act_idx])[0, 1]
                    if not np.isnan(c):
                        within_corrs.append(c)
            avg_within_corr = np.mean(within_corrs) if within_corrs else np.nan

            rows.append({
                "building": BUILDING_LABELS[bid],
                "action": act_name,
                "mean": f"{actions[:, act_idx].mean():.3f}",
                "overall_std": f"{overall_std:.4f}",
                "within_month_std": f"{avg_within_month_std:.4f}",
                "corr_outdoor_temp": f"{corr_outdoor:.4f}",
                "within_month_corr_outdoor": f"{avg_within_corr:.4f}",
            })

    # Counterfactual sensitivity
    cf_winter_heat = max(actions_list) - min(actions_list)
    # Re-compute for each reference
    cf_summary = {}
    for label, ref in [("winter_noon", winter_ref), ("summer_noon", summer_ref)]:
        for act_idx, act_name in enumerate(["heating_sp", "cooling_sp"]):
            acts = []
            for t in temp_range:
                test = ref.copy()
                test[3] = t
                a = actor.act(test[None])[0]
                acts.append(a[act_idx])
            cf_summary[f"{label}_{act_name}_range"] = max(acts) - min(acts)

    # Save table
    table_path = TABLE_DIR / "actor_f1_f2_stats.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {table_path}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("f1 vs f2 VERDICT SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"{'Building':<8s} {'Action':<12s} {'mean':>8s} {'std':>8s} "
        f"{'within_mo_std':>15s} {'corr':>8s} {'within_mo_corr':>15s}"
    )
    print("-" * 75)
    for r in rows:
        print(
            f"{r['building']:<8s} {r['action']:<12s} {r['mean']:>8s} {r['overall_std']:>8s} "
            f"{r['within_month_std']:>15s} {r['corr_outdoor_temp']:>8s} "
            f"{r['within_month_corr_outdoor']:>15s}"
        )

    print(f"\nCounterfactual sensitivity (action range as outdoor_temp varies [-20, 40]°C):")
    for k, v in cf_summary.items():
        print(f"  {k:<35s} {v:.4f}")

    print(f"\n{'=' * 70}")
    print("INTERPRETATION:")
    print(f"{'=' * 70}")
    print("  - Low `within_month_std` + low `within_month_corr` + low counterfactual")
    print("    sensitivity → f1 (climate-invariant, cross-climate transfer trivial)")
    print("  - High within-month variation + high corr + high CF sensitivity")
    print("    → f2 (climate-dependent, cross-climate transfer non-trivial)")


if __name__ == "__main__":
    main()
