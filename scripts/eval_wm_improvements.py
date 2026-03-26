"""Evaluate world model improvements: old vs new loss settings.

Compares per-dimension MSE and identity ratio between:
- Baseline: uniform MSE, no grad clip, no LayerNorm
- Improved: dim-weighted MSE + identity guard + grad clip + LayerNorm
"""

import numpy as np
import torch
from loguru import logger

from src.utils import get_device
from src.world_model.dict_dynamics import DictDynamicsModel
from src.world_model.model_trainer import WorldModelTrainer
from src.world_model.sparse_encoder import SparseEncoder


def load_and_split(building_id: str, data_dir: str = "data/offline_rollouts"):
    """Load transitions and split 80/20."""
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


def train_and_evaluate(
    data: dict,
    n_epochs: int = 30,
    use_layernorm: bool = False,
    use_dim_weighting: bool = False,
    identity_penalty_lambda: float = 0.0,
    grad_clip_norm: float = float("inf"),
    grad_clip_dict_norm: float = float("inf"),
    label: str = "model",
) -> dict[str, np.ndarray]:
    """Train a world model and evaluate per-dimension MSE."""
    device = get_device()
    state_dim = data["train"]["states"].shape[1]
    action_dim = data["train"]["actions"].shape[1]
    n_atoms = 128

    # Random dictionary (init with unit norm)
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
    ).to(device)

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

    # Convert to tensors
    train_s = torch.tensor(data["train"]["states"], dtype=torch.float32, device=device)
    train_a = torch.tensor(data["train"]["actions"], dtype=torch.float32, device=device)
    train_ns = torch.tensor(
        data["train"]["next_states"], dtype=torch.float32, device=device
    )
    test_s = torch.tensor(data["test"]["states"], dtype=torch.float32, device=device)
    test_a = torch.tensor(data["test"]["actions"], dtype=torch.float32, device=device)
    test_ns = torch.tensor(
        data["test"]["next_states"], dtype=torch.float32, device=device
    )

    # Train
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
                pred, _ = model(test_s, test_a)
                per_dim_mse = ((test_ns - pred) ** 2).mean(dim=0).cpu().numpy()
                identity_mse = ((test_ns - test_s) ** 2).mean(dim=0).cpu().numpy()
            total_mse = per_dim_mse.mean()
            logger.info(f"[{label}] Epoch {epoch}: MSE={total_mse:.6f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred, _ = model(test_s, test_a)
        per_dim_mse = ((test_ns - pred) ** 2).mean(dim=0).numpy()
        identity_mse = ((test_ns - test_s) ** 2).mean(dim=0).numpy()

    return {
        "per_dim_mse": per_dim_mse,
        "identity_mse": identity_mse,
        "ratio": per_dim_mse / (identity_mse + 1e-10),
    }


def main():
    logger.info("Loading office_hot data...")
    data = load_and_split("office_hot")
    state_dim = data["train"]["states"].shape[1]
    n_train = len(data["train"]["states"])
    n_test = len(data["test"]["states"])
    logger.info(f"State dim: {state_dim}, Train: {n_train}, Test: {n_test}")

    n_epochs = 50

    # Baseline: old settings
    logger.info(f"\n=== BASELINE (uniform MSE, no improvements, {n_epochs} epochs) ===")
    baseline = train_and_evaluate(
        data,
        n_epochs=n_epochs,
        use_layernorm=False,
        use_dim_weighting=False,
        identity_penalty_lambda=0.0,
        grad_clip_norm=float("inf"),
        grad_clip_dict_norm=float("inf"),
        label="Baseline",
    )

    # Variant A: grad clip + LayerNorm only (engineering basics)
    logger.info("\n=== VARIANT A (grad clip + LayerNorm only) ===")
    variant_a = train_and_evaluate(
        data,
        n_epochs=n_epochs,
        use_layernorm=True,
        use_dim_weighting=False,
        identity_penalty_lambda=0.0,
        grad_clip_norm=1.0,
        grad_clip_dict_norm=0.1,
        label="GradClip+LN",
    )

    # Variant B: identity guard only (no dim weighting)
    logger.info("\n=== VARIANT B (identity guard only, lambda=0.5) ===")
    variant_b = train_and_evaluate(
        data,
        n_epochs=n_epochs,
        use_layernorm=False,
        use_dim_weighting=False,
        identity_penalty_lambda=0.5,
        grad_clip_norm=1.0,
        grad_clip_dict_norm=0.1,
        label="IdGuard",
    )

    # Full: all improvements
    logger.info("\n=== FULL (dim-weighted + identity guard + grad clip + LN) ===")
    improved = train_and_evaluate(
        data,
        n_epochs=n_epochs,
        use_layernorm=True,
        use_dim_weighting=True,
        identity_penalty_lambda=0.5,
        grad_clip_norm=1.0,
        grad_clip_dict_norm=0.1,
        label="Full",
    )

    # Collect all variants
    variants = {
        "Baseline": baseline,
        "GradClip+LN": variant_a,
        "IdGuard": variant_b,
        "Full": improved,
    }

    # Print comparison (exclude near-constant dims where identity MSE < 1e-3)
    print(f"\n{'=' * 100}")
    print(f"PER-DIMENSION MSE COMPARISON (office_hot, {n_epochs} epochs)")
    print(f"{'=' * 100}")

    header = f"{'Dim':>4} | {'Identity':>10}"
    for name in variants:
        header += f" | {name:>12}"
    print(header)
    print("-" * 100)

    nontrivial = []
    for d in range(state_dim):
        id_mse = baseline["identity_mse"][d]
        row = f"{d:4d} | {id_mse:10.6f}"
        trivial = id_mse < 1e-3
        for _name, v in variants.items():
            mse = v["per_dim_mse"][d]
            if trivial:
                row += f" | {mse:12.8f}"
            else:
                row += f" | {mse:12.6f}"
        if trivial:
            row += "  (near-const)"
        else:
            nontrivial.append(d)
        print(row)

    print("-" * 100)
    # Summary: mean of non-trivial dimensions only
    print(f"\n{'=' * 80}")
    print(f"SUMMARY (non-trivial dims {nontrivial}, identity MSE >= 1e-3)")
    print(f"{'=' * 80}")
    print(f"{'Variant':>15} | {'Mean MSE':>10} | {'vs Baseline':>12}")
    print("-" * 45)

    b_mean = baseline["per_dim_mse"][nontrivial].mean()
    for name, v in variants.items():
        v_mean = v["per_dim_mse"][nontrivial].mean()
        change = (v_mean - b_mean) / b_mean * 100
        print(f"{name:>15} | {v_mean:10.6f} | {change:+11.1f}%")

    # Total MSE (all dims)
    print(f"\n{'Variant':>15} | {'Total MSE':>10} | {'vs Baseline':>12}")
    print("-" * 45)
    b_total = baseline["per_dim_mse"].mean()
    for name, v in variants.items():
        v_total = v["per_dim_mse"].mean()
        change = (v_total - b_total) / b_total * 100
        print(f"{name:>15} | {v_total:10.6f} | {change:+11.1f}%")


if __name__ == "__main__":
    main()
