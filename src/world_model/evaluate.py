"""World model prediction evaluation on real Sinergym data."""

from pathlib import Path

import numpy as np
import torch
from loguru import logger

from src.obs_config import OBS_CONFIG
from src.utils import build_dim_weights, get_device
from src.world_model.dict_dynamics import DictDynamicsModel
from src.world_model.model_trainer import WorldModelTrainer
from src.world_model.sparse_encoder import SparseEncoder


def load_transitions(building_id: str, data_dir: str = "data/offline_rollouts"):
    """Load raw transitions for a building."""
    path = Path(data_dir) / f"{building_id}_transitions.npz"
    data = np.load(path)
    return {
        "states": data["states"],
        "actions": data["actions"],
        "next_states": data["next_states"],
        "rewards": data["rewards"],
    }


def train_world_model(
    dict_path: str = "output/pretrained/dict_k128.pt",
    building_ids: list[str] | None = None,
    data_dir: str = "data/offline_rollouts",
    n_epochs: int = 50,
    batch_size: int = 256,
    encoder_lr: float = 1e-3,
    dict_lr: float = 1e-5,
    sparsity_lambda: float = 0.1,
    sparsity_method: str = "l1_penalty",
    topk_k: int = 16,
    device: str = "auto",
) -> tuple[DictDynamicsModel, dict]:
    """Train world model on offline transitions and evaluate.

    Returns:
        (trained_model, metrics_dict)
    """
    building_ids = building_ids or ["office_hot"]
    dev = get_device(device)

    # Load dictionary
    dict_data = torch.load(dict_path, weights_only=False)
    dictionary = dict_data["dictionary"].to(dev)
    norm_mean = dict_data["obs_mean"].numpy()
    norm_std = dict_data["obs_std"].numpy()
    n_atoms = dictionary.shape[1]
    state_dim = dictionary.shape[0]

    logger.info(f"Dictionary: {dictionary.shape}, {len(building_ids)} buildings")

    # Load all building data
    all_data: dict[str, dict[str, np.ndarray]] = {}
    for bid in building_ids:
        all_data[bid] = load_transitions(bid, data_dir)
        logger.info(f"  {bid}: {all_data[bid]['states'].shape[0]} transitions")

    action_dim = all_data[building_ids[0]]["actions"].shape[1]

    # Build model
    encoder = SparseEncoder(
        state_dim=state_dim,
        action_dim=action_dim,
        n_atoms=n_atoms,
        shared_hidden_dims=[256, 256],
        adapter_dim=64,
        n_buildings=len(building_ids),
        sparsity_method=sparsity_method,
        topk_k=topk_k,
    ).to(dev)

    dim_weights = build_dim_weights(
        state_dim,
        [OBS_CONFIG.AIR_TEMPERATURE, OBS_CONFIG.HVAC_POWER],
        5.0,
        dev,
    )
    model = DictDynamicsModel(
        dictionary=dictionary,
        sparse_encoder=encoder,
        learnable_dict=dict_lr > 0,
        dim_weights=dim_weights,
    ).to(dev)

    trainer = WorldModelTrainer(
        model=model,
        encoder_lr=encoder_lr,
        dict_lr=dict_lr,
        sparsity_lambda=sparsity_lambda,
    )

    # Split data: 80% train, 20% test
    # Compute normalization stats from TRAIN split only (avoid test data leakage)
    train_data_by_bid: dict[str, dict[str, torch.Tensor]] = {}
    test_data_by_bid: dict[str, dict[str, torch.Tensor]] = {}

    all_train_states = []
    splits: dict[str, int] = {}
    for bid in building_ids:
        d = all_data[bid]
        n = len(d["states"])
        split = int(0.8 * n)
        splits[bid] = split
        all_train_states.append(d["states"][:split])

    # Normalization stats from train data only
    train_states_cat = np.concatenate(all_train_states, axis=0)
    norm_mean = train_states_cat.mean(axis=0)
    norm_std = np.maximum(train_states_cat.std(axis=0), 1e-8)
    logger.info(
        f"Normalization stats from train split: {len(train_states_cat)} samples"
    )

    def normalize(x):
        return (x - norm_mean) / norm_std

    for bid in building_ids:
        d = all_data[bid]
        split = splits[bid]
        train_data_by_bid[bid] = {
            "states": torch.tensor(
                normalize(d["states"][:split]), dtype=torch.float32, device=dev
            ),
            "actions": torch.tensor(
                d["actions"][:split], dtype=torch.float32, device=dev
            ),
            "next_states": torch.tensor(
                normalize(d["next_states"][:split]), dtype=torch.float32, device=dev
            ),
        }
        test_data_by_bid[bid] = {
            "states": torch.tensor(
                normalize(d["states"][split:]), dtype=torch.float32, device=dev
            ),
            "actions": torch.tensor(
                d["actions"][split:], dtype=torch.float32, device=dev
            ),
            "next_states": torch.tensor(
                normalize(d["next_states"][split:]), dtype=torch.float32, device=dev
            ),
        }

    # Training loop
    history: list[dict] = []
    for epoch in range(n_epochs):
        epoch_metrics: dict[str, float] = {}
        for i, bid in enumerate(building_ids):
            td = train_data_by_bid[bid]
            n = td["states"].shape[0]
            # Shuffle and batch
            perm = torch.randperm(n, device=dev)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                metrics = trainer.train_step(
                    td["states"][idx],
                    td["actions"][idx],
                    td["next_states"][idx],
                    building_id=str(i),
                )
                total_loss += metrics["total_loss"]
                n_batches += 1
            epoch_metrics[f"{bid}_train_loss"] = total_loss / max(n_batches, 1)

        # Evaluate on test set
        for i, bid in enumerate(building_ids):
            td = test_data_by_bid[bid]
            test_metrics = trainer.evaluate(
                td["states"], td["actions"], td["next_states"], building_id=str(i)
            )
            epoch_metrics[f"{bid}_test_mse"] = test_metrics["mse_loss"]
            epoch_metrics[f"{bid}_test_sparsity"] = test_metrics["sparsity"]

        history.append(epoch_metrics)
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            loss_str = ", ".join(
                f"{k}={v:.6f}" for k, v in epoch_metrics.items() if "test_mse" in k
            )
            spar_str = ", ".join(
                f"{k}={v:.2%}" for k, v in epoch_metrics.items() if "sparsity" in k
            )
            logger.info(f"Epoch {epoch}: {loss_str} | {spar_str}")

    return model, {"history": history, "norm_mean": norm_mean, "norm_std": norm_std}


def evaluate_multistep(
    model: DictDynamicsModel,
    building_id: str,
    building_idx: str,
    data: dict[str, np.ndarray],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    horizons: list[int] | None = None,
    n_samples: int = 1000,
    device: str = "auto",
) -> dict[int, float]:
    """Evaluate multi-step prediction error for different horizons.

    Returns:
        Dict mapping horizon H to mean MSE.
    """
    horizons = horizons or [1, 2, 3, 5]
    dev = get_device(device)
    model.eval()

    def normalize(x):
        return (x - norm_mean) / np.maximum(norm_std, 1e-8)

    states = data["states"]
    actions = data["actions"]
    next_states = data["next_states"]

    max_h = max(horizons)
    n = len(states) - max_h
    n_samples = min(n_samples, n)
    start_indices = np.random.choice(n, n_samples, replace=False)

    results: dict[int, float] = {}
    for h in horizons:
        total_mse = 0.0
        for idx in start_indices:
            s = torch.tensor(
                normalize(states[idx]), dtype=torch.float32, device=dev
            ).unsqueeze(0)
            for step in range(h):
                a = torch.tensor(
                    actions[idx + step], dtype=torch.float32, device=dev
                ).unsqueeze(0)
                with torch.no_grad():
                    s = model.predict(s, a, building_idx)
            # Compare with actual state at idx+h
            actual = normalize(next_states[idx + h - 1])
            pred = s.cpu().numpy().squeeze(0)
            total_mse += np.mean((pred - actual) ** 2)
        results[h] = total_mse / n_samples
        logger.info(f"  H={h}: MSE={results[h]:.6f}")

    return results
