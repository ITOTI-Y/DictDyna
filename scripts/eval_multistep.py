"""Evaluate multi-step autoregressive prediction: old vs improved world model.

This is the critical test — single-step MSE differences are modest, but
multi-step error accumulation is where identity guard, grad clip, and
dim weighting should dramatically diverge from baseline.
"""

import numpy as np
import torch
from loguru import logger

from src.world_model.dict_dynamics import DictDynamicsModel
from src.world_model.model_trainer import WorldModelTrainer
from src.world_model.sparse_encoder import SparseEncoder


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
    }


def build_and_train(
    data: dict,
    n_epochs: int = 50,
    use_layernorm: bool = False,
    use_dim_weighting: bool = False,
    identity_penalty_lambda: float = 0.0,
    grad_clip_norm: float = float("inf"),
    grad_clip_dict_norm: float = float("inf"),
    label: str = "model",
) -> DictDynamicsModel:
    """Build and train a world model, return the trained model."""
    state_dim = data["train"]["states"].shape[1]
    action_dim = data["train"]["actions"].shape[1]
    n_atoms = 128

    torch.manual_seed(42)
    dictionary = torch.randn(state_dim, n_atoms)
    dictionary = dictionary / dictionary.norm(dim=0, keepdim=True)

    encoder = SparseEncoder(
        state_dim=state_dim,
        action_dim=action_dim,
        n_atoms=n_atoms,
        shared_hidden_dims=[256, 256],
        adapter_dim=64,
        sparsity_method="topk",
        topk_k=16,
        use_layernorm=use_layernorm,
    )

    model = DictDynamicsModel(
        dictionary=dictionary,
        sparse_encoder=encoder,
        learnable_dict=True,
    )

    trainer = WorldModelTrainer(
        model=model,
        encoder_lr=1e-3,
        dict_lr=1e-5,
        sparsity_lambda=0.1,
        grad_clip_norm=grad_clip_norm,
        grad_clip_dict_norm=grad_clip_dict_norm,
        identity_penalty_lambda=identity_penalty_lambda,
        dim_weight_ema_decay=0.99,
        use_dim_weighting=use_dim_weighting,
    )

    train_s = torch.tensor(data["train"]["states"], dtype=torch.float32)
    train_a = torch.tensor(data["train"]["actions"], dtype=torch.float32)
    train_ns = torch.tensor(data["train"]["next_states"], dtype=torch.float32)

    batch_size = 256
    n_train = len(train_s)
    for epoch in range(n_epochs):
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            trainer.train_step(train_s[idx], train_a[idx], train_ns[idx])
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                pred, _ = model(train_s[:1000], train_a[:1000])
                mse = ((train_ns[:1000] - pred) ** 2).mean().item()
            logger.info(f"[{label}] Epoch {epoch}: train MSE={mse:.6f}")

    return model


def evaluate_multistep(
    model: DictDynamicsModel,
    data: dict,
    horizons: list[int],
    n_samples: int = 2000,
) -> dict[int, dict]:
    """Evaluate multi-step autoregressive prediction.

    For each horizon H, starting from a real state s_0:
    - Feed s_0, a_0 → s_hat_1
    - Feed s_hat_1, a_1 → s_hat_2  (autoregressive!)
    - ...
    - Compare s_hat_H with true s_H

    Returns per-horizon: total MSE, per-dim MSE, identity MSE.
    """
    model.eval()
    states = data["test"]["states"]
    actions = data["test"]["actions"]
    next_states = data["test"]["next_states"]

    max_h = max(horizons)
    n = len(states) - max_h
    n_samples = min(n_samples, n)
    starts = np.random.choice(n, n_samples, replace=False)

    results = {}
    for h in horizons:
        all_pred = []
        all_true = []
        all_start = []

        for idx in starts:
            s = torch.tensor(
                states[idx], dtype=torch.float32
            ).unsqueeze(0)

            # Autoregressive rollout
            for step in range(h):
                a = torch.tensor(
                    actions[idx + step], dtype=torch.float32
                ).unsqueeze(0)
                with torch.no_grad():
                    s = model.predict(s, a)

            all_pred.append(s.squeeze(0).numpy())
            all_true.append(next_states[idx + h - 1])
            all_start.append(states[idx])

        pred_arr = np.array(all_pred)
        true_arr = np.array(all_true)
        start_arr = np.array(all_start)

        per_dim_mse = ((true_arr - pred_arr) ** 2).mean(axis=0)
        identity_mse = ((true_arr - start_arr) ** 2).mean(axis=0)

        results[h] = {
            "total_mse": per_dim_mse.mean(),
            "per_dim_mse": per_dim_mse,
            "identity_mse": identity_mse,
            "ratio": per_dim_mse / (identity_mse + 1e-10),
        }

    return results


def main():
    np.random.seed(42)
    logger.info("Loading office_hot data...")
    data = load_and_split("office_hot")

    n_epochs = 50
    horizons = [1, 2, 3, 5, 8]

    # Train baseline
    logger.info(f"\n=== Training BASELINE ({n_epochs} epochs) ===")
    baseline_model = build_and_train(
        data, n_epochs=n_epochs,
        use_layernorm=False, use_dim_weighting=False,
        identity_penalty_lambda=0.0,
        grad_clip_norm=float("inf"), grad_clip_dict_norm=float("inf"),
        label="Baseline",
    )

    # Train improved
    logger.info(f"\n=== Training IMPROVED ({n_epochs} epochs) ===")
    improved_model = build_and_train(
        data, n_epochs=n_epochs,
        use_layernorm=True, use_dim_weighting=True,
        identity_penalty_lambda=0.5,
        grad_clip_norm=1.0, grad_clip_dict_norm=0.1,
        label="Improved",
    )

    # Evaluate multi-step
    logger.info("\n=== Evaluating multi-step autoregressive prediction ===")
    np.random.seed(123)
    baseline_results = evaluate_multistep(baseline_model, data, horizons)
    np.random.seed(123)
    improved_results = evaluate_multistep(improved_model, data, horizons)

    # Identity baseline (predict s' = s_0 for all horizons)
    print(f"\n{'=' * 75}")
    print("MULTI-STEP AUTOREGRESSIVE PREDICTION (office_hot)")
    print(f"{'=' * 75}")
    print(f"{'H':>3} | {'Identity':>10} | {'Baseline':>10} | {'Improved':>10} | "
          f"{'B vs Id':>8} | {'I vs Id':>8} | {'I vs B':>8}")
    print("-" * 75)

    for h in horizons:
        id_mse = baseline_results[h]["identity_mse"].mean()
        b_mse = baseline_results[h]["total_mse"]
        i_mse = improved_results[h]["total_mse"]
        b_vs_id = b_mse / (id_mse + 1e-10)
        i_vs_id = i_mse / (id_mse + 1e-10)
        i_vs_b = (i_mse - b_mse) / (b_mse + 1e-10) * 100
        print(f"{h:3d} | {id_mse:10.4f} | {b_mse:10.4f} | {i_mse:10.4f} | "
              f"{b_vs_id:7.2f}x | {i_vs_id:7.2f}x | {i_vs_b:+7.1f}%")

    # Error growth rate analysis
    print(f"\n{'=' * 75}")
    print("ERROR GROWTH RATE (MSE[H] / MSE[1])")
    print(f"{'=' * 75}")
    print(f"{'H':>3} | {'Baseline':>10} | {'Improved':>10} | {'Ratio diff':>10}")
    print("-" * 45)

    b_h1 = baseline_results[1]["total_mse"]
    i_h1 = improved_results[1]["total_mse"]
    for h in horizons:
        b_growth = baseline_results[h]["total_mse"] / (b_h1 + 1e-10)
        i_growth = improved_results[h]["total_mse"] / (i_h1 + 1e-10)
        print(f"{h:3d} | {b_growth:10.2f}x | {i_growth:10.2f}x | {i_growth - b_growth:+9.2f}")

    # Per-dimension analysis at H=5 (focus on non-trivial dims)
    h5 = 5
    print(f"\n{'=' * 75}")
    print(f"PER-DIMENSION MSE AT H={h5} (non-trivial dims only)")
    print(f"{'=' * 75}")
    print(f"{'Dim':>4} | {'Identity':>10} | {'Baseline':>10} | {'Improved':>10} | "
          f"{'B/Id':>6} | {'I/Id':>6} | {'I vs B':>8}")
    print("-" * 75)

    state_dim = baseline_results[h5]["per_dim_mse"].shape[0]
    for d in range(state_dim):
        id_mse = baseline_results[h5]["identity_mse"][d]
        if id_mse < 1e-3:
            continue
        b_mse = baseline_results[h5]["per_dim_mse"][d]
        i_mse = improved_results[h5]["per_dim_mse"][d]
        b_ratio = b_mse / (id_mse + 1e-10)
        i_ratio = i_mse / (id_mse + 1e-10)
        change = (i_mse - b_mse) / (b_mse + 1e-10) * 100
        flag = " !!!" if b_ratio > 1.5 else ""
        print(f"{d:4d} | {id_mse:10.4f} | {b_mse:10.4f} | {i_mse:10.4f} | "
              f"{b_ratio:5.2f}x | {i_ratio:5.2f}x | {change:+7.1f}%{flag}")


if __name__ == "__main__":
    main()
