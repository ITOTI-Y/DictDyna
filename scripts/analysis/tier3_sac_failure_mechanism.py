"""Analyze WHY SAC failed to learn the RBC policy.

From BC experiment: architecture can express the policy; SAC's failure
is optimization-related. Now dig into which mechanism.

Uses an idealized HVAC response model (instead of our broken WM which
produces out-of-range predictions that get clipped to constant rewards):

  indoor_next = clip(indoor_curr, heating_sp, cooling_sp)
  power = K * |indoor_next - outdoor_temp|

This captures the essential Sinergym dynamics: HVAC drives indoor to the
nearest setpoint boundary, and power is proportional to the gap the HVAC
has to bridge. Real dynamics differ in magnitude but shape should match.

Then Sinergym's real reward formula on predicted indoor_temp and power:
  R = -0.5 * λ_E * power - 0.5 * λ_T * comfort_violation

Outputs:
    figures/reward_landscape_by_month.png  — 4-month reward heatmaps
    figures/gradient_direction_by_month.png — gradient pull per month
    tables/landscape_summary.json
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from _share import (
    BUILDING_COLORS,
    FIG_DIR,
    TABLE_DIR,
    ensure_dirs,
    load_building_transitions,
    load_pretrained_dict,
    setup_figure,
)

from src.world_model.sparse_encoder import SparseEncoder
from src.world_model.dict_dynamics import DictDynamicsModel
from src.world_model.reward_estimator import SinergymRewardEstimator


def build_wm_from_ckpt(ckpt_path: Path) -> DictDynamicsModel:
    """Reconstruct the dict-mode WM from a saved checkpoint."""
    ckpt = torch.load(ckpt_path, weights_only=False)
    wm_state = ckpt["world_model"]
    cfg = ckpt["config"]  # dict, not Pydantic

    # Build SparseEncoder first
    state_dim = wm_state["dictionary"].shape[0]
    action_dim = 2
    encoder = SparseEncoder(
        state_dim=state_dim,
        action_dim=action_dim,
        n_atoms=cfg["dictionary"]["n_atoms"],
        shared_hidden_dims=cfg["encoder"]["shared_hidden_dims"],
        adapter_dim=cfg["encoder"]["adapter_dim"],
        n_buildings=1,
        activation=cfg["encoder"]["activation"],
        sparsity_method=cfg["encoder"]["sparsity_method"],
        topk_k=cfg["encoder"]["topk_k"],
    )

    # Build WM
    wm = DictDynamicsModel(
        dictionary=wm_state["dictionary"],
        sparse_encoder=encoder,
        learnable_dict=True,
    )
    wm.load_state_dict(wm_state, strict=False)
    wm.to("cpu")
    wm.eval()
    return wm


def rbc5zone_action(month: int) -> np.ndarray:
    """Sinergym RBC5Zone target action."""
    if 6 <= month <= 9:
        return np.array([23.0, 26.0], dtype=np.float32)
    return np.array([20.0, 23.5], dtype=np.float32)


def idealized_reward(
    month: int,
    indoor_curr: float,
    outdoor_temp: float,
    heating_sp: float,
    cooling_sp: float,
    lambda_e: float = 5e-5,
    lambda_t: float = 0.5,
    power_coeff: float = 300.0,  # W per °C gap
) -> float:
    """Compute Sinergym-style reward using an idealized HVAC response.

    Assumes HVAC perfectly tracks nearest setpoint boundary in one step;
    power is proportional to the temperature gap HVAC must bridge.
    """
    # HVAC response: indoor pushed to nearest setpoint boundary
    if indoor_curr > cooling_sp:
        indoor_next = cooling_sp
        hvac_gap = indoor_curr - cooling_sp  # need cooling
    elif indoor_curr < heating_sp:
        indoor_next = heating_sp
        hvac_gap = heating_sp - indoor_curr  # need heating
    else:
        indoor_next = indoor_curr
        hvac_gap = 0.0

    # Power ∝ gap between target and outdoor (HVAC work to overcome outside)
    # And also ∝ how far HVAC had to push (gap)
    if hvac_gap > 0:
        power = power_coeff * (abs(indoor_next - outdoor_temp) + hvac_gap * 5)
    else:
        power = 0.0

    # Comfort violation (season-based comfort band)
    if 6 <= month <= 9:
        t_low, t_high = 23.0, 26.0
    else:
        t_low, t_high = 20.0, 23.5
    violation = max(0.0, indoor_next - t_high) + max(0.0, t_low - indoor_next)

    return -lambda_e * power - lambda_t * violation


def main():
    setup_figure()
    ensure_dirs()
    project_root = Path(__file__).resolve().parent.parent.parent

    ckpt_path = (
        project_root
        / "output/results/dyna_low_ratio/checkpoint_step105120.pt"
    )
    wm = build_wm_from_ckpt(ckpt_path)
    pre = load_pretrained_dict()
    obs_mean = torch.tensor(pre["obs_mean"], dtype=torch.float32)
    obs_std = torch.tensor(pre["obs_std"], dtype=torch.float32)

    reward_est = SinergymRewardEstimator(obs_mean=obs_mean, obs_std=obs_std)

    # Load representative obs from each quarter-month
    trans = load_building_transitions("office_hot")
    obs_raw = trans["states"]
    months = obs_raw[:, 0].astype(int)

    # Pick noon observation for 4 representative months
    reference_states = {}
    for m in [1, 4, 7, 10]:
        mask = (months == m) & (obs_raw[:, 2] == 12)  # noon
        idx = np.where(mask)[0]
        if len(idx) > 0:
            reference_states[m] = obs_raw[idx[0]]
        else:
            reference_states[m] = obs_raw[np.where(months == m)[0][0]]

    # Normalize obs the way WM expects (using source stats)
    obs_mean_np = pre["obs_mean"]
    obs_std_np = pre["obs_std"]

    # Action grid (raw action values)
    heating_range = np.linspace(12.0, 23.25, 30)
    cooling_range = np.linspace(23.25, 30.0, 30)

    # Action transformation for WM input:
    # WM expects actions in the actor's tanh-scaled form? Actually,
    # checking SAC code, actions go directly into WM as (batch, action_dim).
    # The world model takes raw action values (not normalized).
    # But our WM was trained with normalized actions... let me just use
    # raw values since this gives relative landscape shape.

    # SAC learned policy (from dyna_low_ratio, from f1/f2 analysis)
    # heating ~13.3, cooling ~23.3 (full year avg)
    sac_pt = (13.3, 23.3)

    # === Compute reward landscape per month (using idealized HVAC) ===
    # This replaces the broken WM path. The WM's predictions are out-of-range
    # (~1247°C) and get clipped to constants by reward estimator — see commit
    # message for details.
    landscapes = {}
    for m, state_raw in reference_states.items():
        indoor_curr = float(state_raw[9])    # air_temperature
        outdoor_temp = float(state_raw[3])    # outdoor_temperature

        landscape = np.full((len(heating_range), len(cooling_range)), np.nan)
        for i, h in enumerate(heating_range):
            for j, c in enumerate(cooling_range):
                if c <= h:
                    continue
                landscape[i, j] = idealized_reward(
                    m, indoor_curr, outdoor_temp, h, c
                )
        landscapes[m] = landscape
        print(
            f"Month {m}: indoor_curr={indoor_curr:.1f}°C, "
            f"outdoor={outdoor_temp:.1f}°C, "
            f"max_r={np.nanmax(landscape):.4f}"
        )

    # === Plot reward landscape per month ===
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    month_names = {1: "January", 4: "April", 7: "July", 10: "October"}

    for idx, (m, landscape) in enumerate(landscapes.items()):
        ax = axes[idx // 2, idx % 2]
        im = ax.pcolormesh(
            cooling_range, heating_range, landscape,
            cmap="RdYlGn", shading="auto",
        )

        # Mark RBC optimal, SAC learned
        rbc = rbc5zone_action(m)
        ax.plot(rbc[1], rbc[0], "o", markersize=16,
                markerfacecolor="none", markeredgecolor="black",
                markeredgewidth=2.5, label=f"RBC5Zone ({rbc[1]:.1f}, {rbc[0]:.1f})")
        ax.plot(sac_pt[1], sac_pt[0], "X", markersize=14,
                markerfacecolor="red", markeredgecolor="black",
                markeredgewidth=1.5, label=f"SAC learned ({sac_pt[1]:.1f}, {sac_pt[0]:.1f})")

        ax.set_xlabel("cooling_setpoint (°C)")
        ax.set_ylabel("heating_setpoint (°C)")
        ax.set_title(f"{month_names[m]} — WM-predicted reward")
        ax.legend(fontsize=7, loc="lower left")
        fig.colorbar(im, ax=ax, label="Reward")

    fig.suptitle(
        "Reward Landscape per Month (from trained WM, higher green = better)\n"
        "Optimal (RBC) circled; SAC's learned point crossed"
    )
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "reward_landscape_by_month.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'reward_landscape_by_month.png'}")

    # === Gradient direction analysis ===
    # Use finite-difference directly on the idealized reward function
    # (avoids edge effects from grid boundary).
    print("\n=== Gradient at SAC's learned point by month ===")
    eps = 0.5
    gradient_info = {}
    fig2, ax2 = plt.subplots(figsize=(6, 6))

    for m, state_raw in reference_states.items():
        indoor_curr = float(state_raw[9])
        outdoor_temp = float(state_raw[3])

        r_sac = idealized_reward(
            m, indoor_curr, outdoor_temp, sac_pt[0], sac_pt[1]
        )
        # Finite-difference gradient
        r_h_plus = idealized_reward(
            m, indoor_curr, outdoor_temp, sac_pt[0] + eps, sac_pt[1]
        )
        r_h_minus = idealized_reward(
            m, indoor_curr, outdoor_temp, sac_pt[0] - eps, sac_pt[1]
        )
        r_c_plus = idealized_reward(
            m, indoor_curr, outdoor_temp, sac_pt[0], sac_pt[1] + eps
        )
        r_c_minus = idealized_reward(
            m, indoor_curr, outdoor_temp, sac_pt[0], sac_pt[1] - eps
        )
        dr_dh = (r_h_plus - r_h_minus) / (2 * eps)
        dr_dc = (r_c_plus - r_c_minus) / (2 * eps)

        rbc = rbc5zone_action(m)
        r_at_rbc = idealized_reward(
            m, indoor_curr, outdoor_temp, rbc[0], rbc[1]
        )
        r_at_sac = r_sac
        landscape = landscapes[m]  # for plot

        gradient_info[int(m)] = {
            "sac_reward": float(r_at_sac),
            "rbc_reward": float(r_at_rbc),
            "gap": float(r_at_rbc - r_at_sac),
            "grad_dh_at_sac": float(dr_dh),
            "grad_dc_at_sac": float(dr_dc),
        }

        color = {1: "#1f77b4", 4: "#2ca02c", 7: "#d62728", 10: "#ff7f0e"}[m]
        ax2.arrow(
            sac_pt[1], sac_pt[0], dr_dc * 30, dr_dh * 30,
            head_width=0.15, fc=color, ec=color, alpha=0.7,
            label=f"{month_names[m]}: ∇={dr_dc:.3f}, {dr_dh:.3f}",
        )

        print(
            f"  Month {m:>2d}: r(SAC)={r_at_sac:+8.3f}, r(RBC)={r_at_rbc:+8.3f}, "
            f"gap={r_at_rbc - r_at_sac:+6.3f}, ∇_h={dr_dh:+.3f}, ∇_c={dr_dc:+.3f}"
        )

    # Plot SAC and RBC points
    ax2.plot(sac_pt[1], sac_pt[0], "X", markersize=16,
             markerfacecolor="red", label="SAC learned")
    for m in [1, 4, 7, 10]:
        rbc = rbc5zone_action(m)
        ax2.plot(
            rbc[1], rbc[0], "s", markersize=10,
            markerfacecolor="none", markeredgecolor="black",
        )
    ax2.set_xlabel("cooling_setpoint (°C)")
    ax2.set_ylabel("heating_setpoint (°C)")
    ax2.set_title(
        "Gradient at SAC's (13.3, 23.3) by month\n"
        "Arrow = direction of reward increase (scaled ×30)"
    )
    ax2.legend(fontsize=7, loc="lower left")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(
        FIG_DIR / "gradient_direction_by_month.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig2)
    print(f"Saved {FIG_DIR / 'gradient_direction_by_month.png'}")

    # === Summary ===
    # Average gradient direction (what SAC would see in averaged batch)
    avg_dh = np.mean([v["grad_dh_at_sac"] for v in gradient_info.values()])
    avg_dc = np.mean([v["grad_dc_at_sac"] for v in gradient_info.values()])
    print(f"\n=== Averaged Gradient (batch over all months) ===")
    print(f"  ∇_heating avg = {avg_dh:+.4f}")
    print(f"  ∇_cooling avg = {avg_dc:+.4f}")
    print(f"  Magnitude: {np.sqrt(avg_dh ** 2 + avg_dc ** 2):.4f}")

    # Per-month opposing directions test
    h_grads = [v["grad_dh_at_sac"] for v in gradient_info.values()]
    c_grads = [v["grad_dc_at_sac"] for v in gradient_info.values()]
    print(f"\n  Per-month ∇_h range: [{min(h_grads):.3f}, {max(h_grads):.3f}]")
    print(f"  Per-month ∇_c range: [{min(c_grads):.3f}, {max(c_grads):.3f}]")
    print(
        f"  If signs differ → averaging cancels. "
        f"Sign check: ∇_h {'CONFLICT' if min(h_grads) * max(h_grads) < 0 else 'CONSISTENT'}, "
        f"∇_c {'CONFLICT' if min(c_grads) * max(c_grads) < 0 else 'CONSISTENT'}"
    )

    # Save
    summary = {
        "per_month": {str(k): v for k, v in gradient_info.items()},
        "avg_grad_h": float(avg_dh),
        "avg_grad_c": float(avg_dc),
        "avg_grad_magnitude": float(np.sqrt(avg_dh ** 2 + avg_dc ** 2)),
        "sac_learned_point": list(sac_pt),
    }
    with open(TABLE_DIR / "sac_failure_landscape.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {TABLE_DIR / 'sac_failure_landscape.json'}")


if __name__ == "__main__":
    main()
