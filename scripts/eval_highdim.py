"""Focused evaluation on high-variance and reward-relevant dimensions.

Analyzes multi-step prediction quality specifically for:
1. High-variance dimensions (largest state diffs)
2. Reward-relevant dimensions (indoor_temp=dim9, hvac_power=dim15)
3. Per-dimension error growth curves across horizons H=1..10
"""

import numpy as np
import torch
from loguru import logger

from src.world_model.dict_dynamics import DictDynamicsModel
from src.world_model.model_trainer import WorldModelTrainer
from src.world_model.sparse_encoder import SparseEncoder

# Sinergym 5-zone observation dims (from reward_estimator + PROJECT.md atom analysis)
DIM_NAMES = {
    0: "month",
    1: "day_of_month",
    2: "hour",
    3: "outdoor_temp",
    4: "outdoor_hum",
    5: "wind_speed",
    6: "wind_dir",
    7: "direct_solar",
    8: "diffuse_solar",
    9: "indoor_temp",       # REWARD-RELEVANT
    10: "indoor_hum",
    11: "air_temp_zone",
    12: "htg_setpoint",     # near-constant
    13: "clg_setpoint",     # near-constant
    14: "air_flow",         # near-constant
    15: "hvac_power",       # REWARD-RELEVANT
    16: "occupancy",
}
REWARD_DIMS = [9, 15]


def load_and_split(building_id: str, data_dir: str = "data/offline_rollouts"):
    data = np.load(f"{data_dir}/{building_id}_transitions.npz")
    n = len(data["states"])
    split = int(0.8 * n)
    train_states = data["states"][:split]
    norm_mean = train_states.mean(axis=0)
    norm_std = np.maximum(train_states.std(axis=0), 1e-8)

    def norm(x):
        return (x - norm_mean) / norm_std

    return {
        "train": {
            "states": norm(data["states"][:split]),
            "actions": data["actions"][:split],
            "next_states": norm(data["next_states"][:split]),
        },
        "test": {
            "states": norm(data["states"][split:]),
            "actions": data["actions"][split:],
            "next_states": norm(data["next_states"][split:]),
        },
        "norm_mean": norm_mean,
        "norm_std": norm_std,
    }


def build_and_train(data, n_epochs, improved=False, label="model"):
    state_dim = data["train"]["states"].shape[1]
    action_dim = data["train"]["actions"].shape[1]
    n_atoms = 128

    torch.manual_seed(42)
    dictionary = torch.randn(state_dim, n_atoms)
    dictionary = dictionary / dictionary.norm(dim=0, keepdim=True)

    encoder = SparseEncoder(
        state_dim=state_dim, action_dim=action_dim, n_atoms=n_atoms,
        shared_hidden_dims=[256, 256], adapter_dim=64,
        sparsity_method="topk", topk_k=16,
        use_layernorm=improved,
    )
    model = DictDynamicsModel(
        dictionary=dictionary, sparse_encoder=encoder, learnable_dict=True,
    )
    trainer = WorldModelTrainer(
        model=model, encoder_lr=1e-3, dict_lr=1e-5, sparsity_lambda=0.1,
        grad_clip_norm=1.0 if improved else float("inf"),
        grad_clip_dict_norm=0.1 if improved else float("inf"),
        identity_penalty_lambda=0.5 if improved else 0.0,
        dim_weight_ema_decay=0.99,
        use_dim_weighting=improved,
    )

    train_s = torch.tensor(data["train"]["states"], dtype=torch.float32)
    train_a = torch.tensor(data["train"]["actions"], dtype=torch.float32)
    train_ns = torch.tensor(data["train"]["next_states"], dtype=torch.float32)

    n_train = len(train_s)
    for epoch in range(n_epochs):
        perm = torch.randperm(n_train)
        for start in range(0, n_train, 256):
            idx = perm[start : start + 256]
            trainer.train_step(train_s[idx], train_a[idx], train_ns[idx])
        if epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                pred, _ = model(train_s[:1000], train_a[:1000])
                mse = ((train_ns[:1000] - pred) ** 2).mean().item()
            logger.info(f"[{label}] Final epoch {epoch}: train MSE={mse:.6f}")
    return model


def eval_multistep_perdim(model, data, horizons, n_samples=2000):
    """Per-dimension MSE at each horizon."""
    model.eval()
    states = data["test"]["states"]
    actions = data["test"]["actions"]
    next_states = data["test"]["next_states"]

    max_h = max(horizons)
    n = len(states) - max_h
    n_samples = min(n_samples, n)
    starts = np.random.choice(n, n_samples, replace=False)
    state_dim = states.shape[1]

    results = {}
    for h in horizons:
        preds, trues, srcs = [], [], []
        for idx in starts:
            s = torch.tensor(states[idx], dtype=torch.float32).unsqueeze(0)
            for step in range(h):
                a = torch.tensor(actions[idx + step], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    s = model.predict(s, a)
            preds.append(s.squeeze(0).numpy())
            trues.append(next_states[idx + h - 1])
            srcs.append(states[idx])

        pred_arr = np.array(preds)
        true_arr = np.array(trues)
        src_arr = np.array(srcs)

        per_dim_mse = np.zeros(state_dim)
        identity_mse = np.zeros(state_dim)
        for d in range(state_dim):
            per_dim_mse[d] = ((true_arr[:, d] - pred_arr[:, d]) ** 2).mean()
            identity_mse[d] = ((true_arr[:, d] - src_arr[:, d]) ** 2).mean()

        results[h] = {"per_dim_mse": per_dim_mse, "identity_mse": identity_mse}
    return results


def main():
    np.random.seed(42)
    logger.info("Loading office_hot data...")
    data = load_and_split("office_hot")
    state_dim = data["train"]["states"].shape[1]
    norm_std = data["norm_std"]

    # Analyze state diff variance per dimension
    diffs = data["train"]["next_states"] - data["train"]["states"]
    diff_var = np.var(diffs, axis=0)

    print(f"\n{'=' * 70}")
    print("STATE DIMENSION ANALYSIS (office_hot)")
    print(f"{'=' * 70}")
    print(f"{'Dim':>4} | {'Name':>16} | {'NormStd':>8} | {'DiffVar':>10} | {'Category'}")
    print("-" * 70)

    # Sort by diff variance to identify high-variance dims
    sorted_dims = np.argsort(diff_var)[::-1]
    for d in range(state_dim):
        name = DIM_NAMES.get(d, f"dim_{d}")
        tag = ""
        if d in REWARD_DIMS:
            tag = "REWARD"
        elif diff_var[d] < 1e-6:
            tag = "near-const"
        elif diff_var[d] > 0.01:
            tag = "HIGH-VAR"
        print(f"{d:4d} | {name:>16} | {norm_std[d]:8.2f} | {diff_var[d]:10.6f} | {tag}")

    high_var_dims = [d for d in range(state_dim) if diff_var[d] > 0.001]
    print(f"\nHigh-variance dims (diff_var > 0.001): {high_var_dims}")
    print(f"Reward-relevant dims: {REWARD_DIMS}")

    # Train models
    n_epochs = 50
    horizons = [1, 2, 3, 5, 8, 10]

    logger.info(f"\n=== Training BASELINE ({n_epochs} epochs) ===")
    baseline = build_and_train(data, n_epochs, improved=False, label="Baseline")
    logger.info(f"\n=== Training IMPROVED ({n_epochs} epochs) ===")
    improved = build_and_train(data, n_epochs, improved=True, label="Improved")

    # Evaluate
    np.random.seed(123)
    b_res = eval_multistep_perdim(baseline, data, horizons)
    np.random.seed(123)
    i_res = eval_multistep_perdim(improved, data, horizons)

    # === HIGH-VARIANCE DIMENSIONS: error growth across horizons ===
    focus_dims = sorted(set(high_var_dims + REWARD_DIMS))
    print(f"\n{'=' * 90}")
    print(f"HIGH-VARIANCE + REWARD DIMS: Multi-Step Error Growth")
    print(f"{'=' * 90}")

    for d in focus_dims:
        name = DIM_NAMES.get(d, f"dim_{d}")
        tag = " [REWARD]" if d in REWARD_DIMS else ""
        print(f"\n  Dim {d}: {name}{tag} (diff_var={diff_var[d]:.6f})")
        print(f"  {'H':>4} | {'Identity':>10} | {'Baseline':>10} | {'Improved':>10} | "
              f"{'B/Id':>6} | {'I/Id':>6} | {'I vs B':>8}")
        print(f"  {'-' * 72}")
        for h in horizons:
            id_m = b_res[h]["identity_mse"][d]
            b_m = b_res[h]["per_dim_mse"][d]
            i_m = i_res[h]["per_dim_mse"][d]
            b_r = b_m / (id_m + 1e-10)
            i_r = i_m / (id_m + 1e-10)
            chg = (i_m - b_m) / (b_m + 1e-10) * 100
            print(f"  {h:4d} | {id_m:10.4f} | {b_m:10.4f} | {i_m:10.4f} | "
                  f"{b_r:5.2f}x | {i_r:5.2f}x | {chg:+7.1f}%")

    # === SUMMARY TABLE: improvement % at H=5 for all non-trivial dims ===
    h = 5
    print(f"\n{'=' * 90}")
    print(f"SUMMARY AT H={h}: Per-Dimension Improvement (sorted by improvement)")
    print(f"{'=' * 90}")
    print(f"{'Dim':>4} | {'Name':>16} | {'Base MSE':>10} | {'Impr MSE':>10} | "
          f"{'Change':>8} | {'Base/Id':>7} | {'Impr/Id':>7}")
    print("-" * 90)

    improvements = []
    for d in range(state_dim):
        id_m = b_res[h]["identity_mse"][d]
        if id_m < 1e-3:
            continue
        b_m = b_res[h]["per_dim_mse"][d]
        i_m = i_res[h]["per_dim_mse"][d]
        chg = (i_m - b_m) / (b_m + 1e-10) * 100
        improvements.append((d, chg, b_m, i_m, id_m))

    improvements.sort(key=lambda x: x[1])
    for d, chg, b_m, i_m, id_m in improvements:
        name = DIM_NAMES.get(d, f"dim_{d}")
        tag = " *" if d in REWARD_DIMS else ""
        b_r = b_m / (id_m + 1e-10)
        i_r = i_m / (id_m + 1e-10)
        print(f"{d:4d} | {name:>16}{tag} | {b_m:10.4f} | {i_m:10.4f} | "
              f"{chg:+7.1f}% | {b_r:6.2f}x | {i_r:6.2f}x")

    # Aggregate stats
    all_nontrivial = [d for d in range(state_dim)
                      if b_res[h]["identity_mse"][d] >= 1e-3]
    b_total = np.mean([b_res[h]["per_dim_mse"][d] for d in all_nontrivial])
    i_total = np.mean([i_res[h]["per_dim_mse"][d] for d in all_nontrivial])
    hv_dims = [d for d in all_nontrivial if diff_var[d] > 0.01]
    b_hv = np.mean([b_res[h]["per_dim_mse"][d] for d in hv_dims])
    i_hv = np.mean([i_res[h]["per_dim_mse"][d] for d in hv_dims])
    b_rw = np.mean([b_res[h]["per_dim_mse"][d] for d in REWARD_DIMS])
    i_rw = np.mean([i_res[h]["per_dim_mse"][d] for d in REWARD_DIMS])

    print(f"\n{'=' * 60}")
    print(f"AGGREGATE AT H={h}")
    print(f"{'=' * 60}")
    print(f"{'Group':>20} | {'Baseline':>10} | {'Improved':>10} | {'Change':>8}")
    print("-" * 60)
    print(f"{'All non-trivial':>20} | {b_total:10.4f} | {i_total:10.4f} | "
          f"{(i_total-b_total)/b_total*100:+7.1f}%")
    print(f"{'High-variance':>20} | {b_hv:10.4f} | {i_hv:10.4f} | "
          f"{(i_hv-b_hv)/b_hv*100:+7.1f}%")
    print(f"{'Reward-relevant':>20} | {b_rw:10.4f} | {i_rw:10.4f} | "
          f"{(i_rw-b_rw)/b_rw*100:+7.1f}%")


if __name__ == "__main__":
    main()
