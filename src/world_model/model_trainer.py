"""World model training loop."""

import torch

from src.world_model.context_dynamics import ContextDynamicsModel
from src.world_model.dict_dynamics import DictDynamicsModel


class WorldModelTrainer:
    """Train the DictDyna world model (encoder + optional dictionary update).

    Args:
        model: DictDynamicsModel instance.
        encoder_lr: Learning rate for sparse encoder.
        dict_lr: Learning rate for dictionary (0 = frozen).
        sparsity_lambda: L1 sparsity weight.
    """

    def __init__(
        self,
        model: DictDynamicsModel,
        encoder_lr: float = 1e-3,
        dict_lr: float = 1e-5,
        sparsity_lambda: float = 0.1,
    ) -> None:
        self.model = model
        self.sparsity_lambda = sparsity_lambda

        # Separate parameter groups for encoder and dictionary
        param_groups = [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
        ]
        if dict_lr > 0 and isinstance(model.dictionary, torch.nn.Parameter):
            param_groups.append({"params": [model.dictionary], "lr": dict_lr})

        self.optimizer = torch.optim.Adam(param_groups)

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        building_id: str = "0",
        sample_weights: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Single training step.

        Args:
            sample_weights: Per-sample importance weights for reward-aware
                training. Higher weight = more accurate reconstruction.

        Returns:
            Dict of training metrics.
        """
        self.model.train()
        loss, metrics = self.model.compute_loss(
            states,
            actions,
            next_states,
            building_id,
            self.sparsity_lambda,
            sample_weights,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Normalize dictionary atoms after update
        self.model.normalize_atoms()

        return metrics

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        building_id: str = "0",
    ) -> dict[str, float]:
        """Train for one epoch over a dataloader.

        Expects dataloader to yield (states, actions, next_states) batches.

        Returns:
            Average metrics over the epoch.
        """
        total_metrics: dict[str, float] = {}
        n_batches = 0

        for batch in dataloader:
            states, actions, next_states = batch
            metrics = self.train_step(states, actions, next_states, building_id)

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            n_batches += 1

        avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}
        return avg_metrics

    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        building_id: str = "0",
    ) -> dict[str, float]:
        """Evaluate model without gradient updates."""
        self.model.eval()
        with torch.no_grad():
            _, metrics = self.model.compute_loss(
                states, actions, next_states, building_id, self.sparsity_lambda
            )
        return metrics


class ContextWorldModelTrainer:
    """Train context-conditioned world model.

    Args:
        model: ContextDynamicsModel instance.
        encoder_lr: Learning rate for conditioned encoder.
        context_lr: Learning rate for context encoder.
        dict_lr: Learning rate for dictionary (0 = frozen).
        sparsity_lambda: L1 sparsity weight.
    """

    def __init__(
        self,
        model: ContextDynamicsModel,
        encoder_lr: float = 1e-3,
        context_lr: float = 1e-3,
        dict_lr: float = 1e-5,
        sparsity_lambda: float = 0.1,
    ) -> None:
        self.model = model
        self.sparsity_lambda = sparsity_lambda

        param_groups = [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.context_encoder.parameters(), "lr": context_lr},
        ]
        if dict_lr > 0 and isinstance(model.dictionary, torch.nn.Parameter):
            param_groups.append({"params": [model.dictionary], "lr": dict_lr})

        self.optimizer = torch.optim.Adam(param_groups)

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        context: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Single training step with context vector.

        Args:
            context: Pre-computed context vector z, shape (batch, context_dim).
            sample_weights: Per-sample importance weights.

        Returns:
            Dict of training metrics.
        """
        self.model.train()
        loss, metrics = self.model.compute_loss(
            states,
            actions,
            next_states,
            context,
            self.sparsity_lambda,
            sample_weights,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.model.normalize_atoms()

        return metrics
