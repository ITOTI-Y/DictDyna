"""Thought experiment: Can a simple MLP learn the RBC5Zone policy via BC?

Motivation: Why didn't SAC learn the optimal comfort-band-edge policy?
Is the mapping inexpressible, or is SAC's gradient optimization the problem?

Approach:
1. Use offline obs from 5zone_hot (35,040 steps)
2. Generate "optimal actions" using RBC5Zone's rule:
     Summer (Jun-Sep): (heating=23, cooling=26)
     Winter (else):    (heating=20, cooling=23.5)
3. Train an MLP via behavior cloning (MSE loss) to predict these actions
4. Measure how quickly it learns, whether it matches RBC exactly

Interpretation:
  - If BC trivially learns RBC policy → architecture OK, SAC's failure is
    due to gradient dynamics / compromise across months / no-HVAC local opt
  - If BC struggles → observation insufficient or MLP can't express lookup
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from _share import (
    ensure_dirs,
    load_building_transitions,
    load_pretrained_dict,
    setup_figure,
)


def rbc5zone_action(month: int) -> np.ndarray:
    """Sinergym RBC5Zone optimal action given month."""
    if 6 <= month <= 9:  # Summer
        return np.array([23.0, 26.0], dtype=np.float32)
    return np.array([20.0, 23.5], dtype=np.float32)


def main():
    setup_figure()
    ensure_dirs()
    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = project_root / "output/analysis/bc_thought_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Step 1: Load hot offline data + build dataset ===
    trans = load_building_transitions("office_hot")
    obs = trans["states"].astype(np.float32)  # (35040, 17)
    print(f"Loaded hot obs: {obs.shape}")

    # Generate "optimal" actions from RBC rule
    months = obs[:, 0].astype(int)
    targets = np.stack([rbc5zone_action(m) for m in months]).astype(np.float32)
    print(f"Target action distribution:")
    unique_actions, counts = np.unique(targets, axis=0, return_counts=True)
    for ua, c in zip(unique_actions, counts, strict=True):
        print(f"  {ua}: {c} steps ({c / len(targets):.1%})")

    # Normalize obs (using source pretrained stats — same as SAC training)
    pre = load_pretrained_dict()
    obs_mean = pre["obs_mean"].astype(np.float32)
    obs_std = pre["obs_std"].astype(np.float32)
    obs_norm = np.clip((obs - obs_mean) / np.maximum(obs_std, 1e-8), -10, 10)

    # === Step 2: Behavior Cloning ===
    torch.manual_seed(42)
    mlp = nn.Sequential(
        nn.Linear(17, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 2),
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    X = torch.tensor(obs_norm)
    y = torch.tensor(targets)

    n_epochs = 50
    batch_size = 2048  # bigger batch = faster
    losses = []
    heating_mae = []
    cooling_mae = []
    month_breakdown: list[dict] = []

    n_steps_per_epoch = max(1, len(X) // batch_size)
    for epoch in range(n_epochs):
        perm = torch.randperm(len(X))
        total_loss = 0.0
        for i in range(n_steps_per_epoch):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            pred = mlp(X[idx])
            loss = ((pred - y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / n_steps_per_epoch
        losses.append(avg_loss)

        with torch.no_grad():
            pred_all = mlp(X).numpy()
        h_mae = float(np.abs(pred_all[:, 0] - targets[:, 0]).mean())
        c_mae = float(np.abs(pred_all[:, 1] - targets[:, 1]).mean())
        heating_mae.append(h_mae)
        cooling_mae.append(c_mae)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{n_epochs}: loss={avg_loss:.4f}, "
                f"heating MAE={h_mae:.3f}°C, cooling MAE={c_mae:.3f}°C"
            )

    # === Step 3: Per-month predicted action analysis ===
    with torch.no_grad():
        pred_final = mlp(X).numpy()

    print("\n=== Per-Month BC-learned Policy ===")
    print(f"{'Month':<6s} {'Target (H, C)':<18s} {'BC pred (H, C)':<22s} {'Err':<10s}")
    for m in range(1, 13):
        mask = months == m
        if not mask.any():
            continue
        target_m = targets[mask][0]
        pred_m = pred_final[mask].mean(axis=0)
        err = np.abs(pred_m - target_m)
        month_breakdown.append({
            "month": m,
            "target": target_m.tolist(),
            "predicted": pred_m.tolist(),
            "error": err.tolist(),
        })
        print(
            f"{m:<6d} ({target_m[0]:.1f}, {target_m[1]:.1f})   "
            f"({pred_m[0]:.2f}, {pred_m[1]:.2f})       "
            f"({err[0]:.2f}, {err[1]:.2f})"
        )

    # === Step 4: Compare with SAC's learned policy ===
    # Load dyna_low_ratio (SAC trained on hot)
    sac_ckpt_path = project_root / "output/results/dyna_low_ratio/checkpoint_step105120.pt"
    sac_ckpt = torch.load(sac_ckpt_path, weights_only=False)
    actor_state = sac_ckpt["actor"]

    # Build minimal actor with same structure
    class SACActor(nn.Module):
        def __init__(self, state: dict):
            super().__init__()
            self.action_scale = state["action_scale"].cpu()
            self.action_bias = state["action_bias"].cpu()
            self.obs_mean = state["obs_norm.mean"].cpu()
            self.obs_std = state["obs_norm.std"].cpu()
            self.trunk = nn.Sequential(
                nn.Linear(17, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
            )
            self.trunk[0].weight.data = state["trunk.0.weight"].cpu()
            self.trunk[0].bias.data = state["trunk.0.bias"].cpu()
            self.trunk[2].weight.data = state["trunk.2.weight"].cpu()
            self.trunk[2].bias.data = state["trunk.2.bias"].cpu()
            self.mean_head = nn.Linear(256, 2)
            self.mean_head.weight.data = state["mean_head.weight"].cpu()
            self.mean_head.bias.data = state["mean_head.bias"].cpu()

        @torch.no_grad()
        def act(self, obs_raw: torch.Tensor) -> torch.Tensor:
            normed = torch.clamp(
                (obs_raw - self.obs_mean) / self.obs_std, -10.0, 10.0
            )
            h = self.trunk(normed)
            mean = self.mean_head(h)
            return torch.tanh(mean) * self.action_scale + self.action_bias

    sac_actor = SACActor(actor_state)
    sac_pred = sac_actor.act(torch.tensor(obs)).numpy()

    # === Plot: per-month comparison ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, name in enumerate(["heating_setpoint", "cooling_setpoint"]):
        ax = axes[i]
        rbc_by_month = [rbc5zone_action(m)[i] for m in range(1, 13)]
        bc_by_month = []
        sac_by_month = []
        for m in range(1, 13):
            mask = months == m
            if mask.any():
                bc_by_month.append(pred_final[mask, i].mean())
                sac_by_month.append(sac_pred[mask, i].mean())
            else:
                bc_by_month.append(np.nan)
                sac_by_month.append(np.nan)

        ax.plot(range(1, 13), rbc_by_month, "o-", label="RBC5Zone (target)",
                color="green", linewidth=2.5, markersize=8)
        ax.plot(range(1, 13), bc_by_month, "s--", label="BC (MLP supervised)",
                color="blue", markersize=7)
        ax.plot(range(1, 13), sac_by_month, "^:", label="SAC (RL)",
                color="red", markersize=7)
        ax.set_xlabel("Month")
        ax.set_ylabel(f"{name} (°C)")
        ax.set_title(f"{name}: RBC vs BC vs SAC")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Thought Experiment: Can BC learn RBC policy? Yes → SAC's failure is gradient dynamics, not architecture",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(
        out_dir / "bc_vs_sac_policies.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"\nSaved {out_dir / 'bc_vs_sac_policies.png'}")

    # === Plot: BC loss curve ===
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.plot(losses, label="BC train loss (MSE)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE loss")
    ax2.set_yscale("log")
    ax2.set_title(
        "BC training loss — if this converges to ~0, architecture can express optimal policy"
    )
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(
        out_dir / "bc_loss_curve.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig2)

    # === Summary ===
    final_h_mae = heating_mae[-1]
    final_c_mae = cooling_mae[-1]
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")
    print(f"BC final errors: heating MAE = {final_h_mae:.3f}°C, cooling MAE = {final_c_mae:.3f}°C")
    if final_h_mae < 0.5 and final_c_mae < 0.5:
        print(
            "→ BC CAN learn RBC policy easily. Architecture is sufficient.\n"
            "  SAC's failure must be due to:\n"
            "  1. Gradient averaging across months (compromise solution)\n"
            "  2. Local 'no-HVAC' optimum in reward landscape\n"
            "  3. Reward scale favoring risk-averse wide-deadband policy\n"
            "  4. Insufficient training (105K steps) for policy to carve seasonal splits"
        )
    else:
        print(
            "→ BC cannot fully match RBC. Possible issues:\n"
            "  - Obs encoding loses seasonal info\n"
            "  - MLP capacity insufficient\n"
            "  - Month signal too weak after normalization"
        )

    # SAC per-month errors
    sac_h_mae = float(np.abs(sac_pred[:, 0] - targets[:, 0]).mean())
    sac_c_mae = float(np.abs(sac_pred[:, 1] - targets[:, 1]).mean())
    print(f"\nSAC errors vs RBC target: heating MAE = {sac_h_mae:.3f}°C, "
          f"cooling MAE = {sac_c_mae:.3f}°C")
    print(f"BC/SAC error ratio: heating {final_h_mae / max(sac_h_mae, 1e-6):.3f}, "
          f"cooling {final_c_mae / max(sac_c_mae, 1e-6):.3f}")

    # Save
    summary = {
        "bc_final_heating_mae": final_h_mae,
        "bc_final_cooling_mae": final_c_mae,
        "sac_heating_mae": sac_h_mae,
        "sac_cooling_mae": sac_c_mae,
        "per_month": month_breakdown,
        "interpretation": (
            "BC succeeds → architecture sufficient, SAC's failure is optimization"
            if (final_h_mae < 0.5 and final_c_mae < 0.5)
            else "BC also struggles → deeper architectural issue"
        ),
    }
    with open(out_dir / "bc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
