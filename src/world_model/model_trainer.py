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
        grad_clip_norm: Max gradient norm for encoder parameters.
        grad_clip_dict_norm: Max gradient norm for dictionary parameters.
        identity_penalty_lambda: Penalty for being worse than identity mapping.
        dim_weight_ema_decay: EMA decay for per-dimension loss weights.
        use_dim_weighting: Enable per-dimension adaptive loss weighting.
    """

    def __init__(
        self,
        model: DictDynamicsModel,
        encoder_lr: float = 1e-3,
        dict_lr: float = 1e-5,
        sparsity_lambda: float = 0.1,
        grad_clip_norm: float = 1.0,
        grad_clip_dict_norm: float = 0.1,
        identity_penalty_lambda: float = 0.0,
        dim_weight_ema_decay: float = 0.99,
        use_dim_weighting: bool = False,
        residual_lambda: float = 0.0,
    ) -> None:
        self.model = model
        self.sparsity_lambda = sparsity_lambda
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_dict_norm = grad_clip_dict_norm
        self.identity_penalty_lambda = identity_penalty_lambda
        self.dim_weight_ema_decay = dim_weight_ema_decay
        self.use_dim_weighting = use_dim_weighting
        self.residual_lambda = residual_lambda

        # Separate parameter groups for encoder and dictionary
        self._encoder_params = list(model.encoder.parameters())
        self._dict_params: list[torch.nn.Parameter] = []

        param_groups = [
            {"params": self._encoder_params, "lr": encoder_lr},
        ]
        if dict_lr > 0 and isinstance(model.dictionary, torch.nn.Parameter):
            self._dict_params = [model.dictionary]
            param_groups.append({"params": self._dict_params, "lr": dict_lr})

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
            identity_penalty_lambda=self.identity_penalty_lambda,
            dim_weight_ema_decay=self.dim_weight_ema_decay,
            use_dim_weighting=self.use_dim_weighting,
            residual_lambda=self.residual_lambda,
        )

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping: separate norms for encoder and dictionary
        if self._encoder_params:
            torch.nn.utils.clip_grad_norm_(self._encoder_params, self.grad_clip_norm)
        if self._dict_params:
            torch.nn.utils.clip_grad_norm_(self._dict_params, self.grad_clip_dict_norm)

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

    def train_multistep(
        self,
        states_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        next_states_seq: torch.Tensor,
        building_id: str = "0",
        discount: float = 0.95,
        teacher_forcing_ratio: float = 0.5,
    ) -> dict[str, float]:
        """Multi-step consistency training with teacher forcing.

        Trains on H-step rollouts, penalizing cumulative prediction drift.
        Teacher forcing randomly replaces model predictions with ground truth.

        Args:
            states_seq: Shape (batch, H, d) - consecutive states.
            actions_seq: Shape (batch, H, m) - consecutive actions.
            next_states_seq: Shape (batch, H, d) - consecutive next states.
            building_id: Building identifier.
            discount: Discount for multi-step error (lambda^h).
            teacher_forcing_ratio: Probability of using true state as input.

        Returns:
            Dict of training metrics.
        """
        self.model.train()
        batch_size, horizon, _ = states_seq.shape

        total_mse = torch.tensor(0.0, device=states_seq.device)
        current = states_seq[:, 0]  # (batch, d)

        for h in range(horizon):
            action = actions_seq[:, h]
            target = next_states_seq[:, h]

            pred, alpha = self.model(current, action, building_id)
            step_mse = ((target - pred) ** 2).mean()
            total_mse = total_mse + (discount**h) * step_mse

            # Teacher forcing: use ground truth with probability p
            if h < horizon - 1:
                use_teacher = torch.rand(batch_size, 1, device=pred.device)
                current = torch.where(use_teacher < teacher_forcing_ratio, target, pred)

        # Add sparsity from last step (representative)
        l1_loss = torch.mean(torch.abs(alpha))
        loss = total_mse / horizon + self.sparsity_lambda * l1_loss

        self.optimizer.zero_grad()
        loss.backward()

        if self._encoder_params:
            torch.nn.utils.clip_grad_norm_(self._encoder_params, self.grad_clip_norm)
        if self._dict_params:
            torch.nn.utils.clip_grad_norm_(self._dict_params, self.grad_clip_dict_norm)

        self.optimizer.step()
        self.model.normalize_atoms()

        return {
            "multistep_mse": (total_mse / horizon).item(),
            "multistep_l1": l1_loss.item(),
            "multistep_loss": loss.item(),
            "multistep_horizon": horizon,
        }

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
        grad_clip_norm: Max gradient norm for encoder parameters.
        grad_clip_dict_norm: Max gradient norm for dictionary parameters.
        identity_penalty_lambda: Penalty for being worse than identity mapping.
        dim_weight_ema_decay: EMA decay for per-dimension loss weights.
        use_dim_weighting: Enable per-dimension adaptive loss weighting.
    """

    def __init__(
        self,
        model: ContextDynamicsModel,
        encoder_lr: float = 1e-3,
        context_lr: float = 1e-3,
        dict_lr: float = 1e-5,
        sparsity_lambda: float = 0.1,
        grad_clip_norm: float = 1.0,
        grad_clip_dict_norm: float = 0.1,
        identity_penalty_lambda: float = 0.0,
        dim_weight_ema_decay: float = 0.99,
        use_dim_weighting: bool = False,
        residual_lambda: float = 0.0,
    ) -> None:
        self.model = model
        self.sparsity_lambda = sparsity_lambda
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_dict_norm = grad_clip_dict_norm
        self.identity_penalty_lambda = identity_penalty_lambda
        self.dim_weight_ema_decay = dim_weight_ema_decay
        self.use_dim_weighting = use_dim_weighting
        self.residual_lambda = residual_lambda

        self._encoder_params = list(model.encoder.parameters()) + list(
            model.context_encoder.parameters()
        )
        self._dict_params: list[torch.nn.Parameter] = []

        param_groups = [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.context_encoder.parameters(), "lr": context_lr},
        ]
        if dict_lr > 0 and isinstance(model.dictionary, torch.nn.Parameter):
            self._dict_params = [model.dictionary]
            param_groups.append({"params": self._dict_params, "lr": dict_lr})

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
            identity_penalty_lambda=self.identity_penalty_lambda,
            dim_weight_ema_decay=self.dim_weight_ema_decay,
            use_dim_weighting=self.use_dim_weighting,
            residual_lambda=self.residual_lambda,
        )

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping: separate norms for encoder and dictionary
        if self._encoder_params:
            torch.nn.utils.clip_grad_norm_(self._encoder_params, self.grad_clip_norm)
        if self._dict_params:
            torch.nn.utils.clip_grad_norm_(self._dict_params, self.grad_clip_dict_norm)

        self.optimizer.step()

        self.model.normalize_atoms()

        return metrics
