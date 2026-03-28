"""Unified world model training loop.

Works with any BaseDictDynamics subclass (DictDynamicsModel,
ContextDynamicsModel). Routing arguments (``building_id`` or ``context``)
are passed as keyword arguments via ``**model_kwargs``.
"""

import torch

from src.world_model._share import BaseDictDynamics


class WorldModelTrainer:
    """Train a dictionary-based world model (encoder + optional dict update).

    Auto-detects parameter groups: if the model has a ``context_encoder``
    attribute, its parameters get a separate learning rate.

    Args:
        model: Any BaseDictDynamics subclass.
        encoder_lr: Learning rate for sparse/conditioned encoder.
        context_lr: Learning rate for context encoder (if present).
        dict_lr: Learning rate for dictionary (0 = frozen).
        sparsity_lambda: L1 sparsity weight.
        grad_clip_norm: Max gradient norm for encoder parameters.
        grad_clip_dict_norm: Max gradient norm for dictionary parameters.
        identity_penalty_lambda: Penalty for being worse than identity mapping.
        dim_weight_ema_decay: EMA decay for per-dimension loss weights.
        use_dim_weighting: Enable per-dimension adaptive loss weighting.
        residual_lambda: L2 regularization weight on residual output.
    """

    def __init__(
        self,
        model: BaseDictDynamics,
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

        # Build parameter groups
        param_groups: list[dict] = [
            {"params": list(model.encoder.parameters()), "lr": encoder_lr},
        ]
        # Context encoder gets separate LR if present
        if hasattr(model, "context_encoder"):
            param_groups.append(
                {"params": list(model.context_encoder.parameters()), "lr": context_lr},  # type: ignore[union-attr]
            )

        self._encoder_params = []
        for pg in param_groups:
            self._encoder_params.extend(pg["params"])

        self._dict_params: list[torch.nn.Parameter] = []
        if dict_lr > 0 and isinstance(model.dictionary, torch.nn.Parameter):
            self._dict_params = [model.dictionary]
            param_groups.append({"params": self._dict_params, "lr": dict_lr})

        self.optimizer = torch.optim.Adam(param_groups)

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
        **model_kwargs: object,
    ) -> dict[str, float]:
        """Single training step.

        Args:
            sample_weights: Per-sample importance weights for reward-aware
                training. Higher weight = more accurate reconstruction.
            **model_kwargs: Forwarded to model.compute_loss (e.g.
                ``building_id="0"`` or ``context=z``).

        Returns:
            Dict of training metrics.
        """
        self.model.train()
        loss, metrics = self.model.compute_loss(
            states,
            actions,
            next_states,
            sparsity_lambda=self.sparsity_lambda,
            sample_weights=sample_weights,
            identity_penalty_lambda=self.identity_penalty_lambda,
            dim_weight_ema_decay=self.dim_weight_ema_decay,
            use_dim_weighting=self.use_dim_weighting,
            residual_lambda=self.residual_lambda,
            **model_kwargs,
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
        **model_kwargs: object,
    ) -> dict[str, float]:
        """Train for one epoch over a dataloader.

        Expects dataloader to yield (states, actions, next_states) batches.
        """
        total_metrics: dict[str, float] = {}
        n_batches = 0

        for batch in dataloader:
            states, actions, next_states = batch
            metrics = self.train_step(states, actions, next_states, **model_kwargs)

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            n_batches += 1

        return {k: v / n_batches for k, v in total_metrics.items()}

    def train_multistep(
        self,
        states_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        next_states_seq: torch.Tensor,
        discount: float = 0.95,
        teacher_forcing_ratio: float = 0.5,
        **model_kwargs: object,
    ) -> dict[str, float]:
        """Multi-step consistency training with teacher forcing.

        Args:
            states_seq: Shape (batch, H, d) - consecutive states.
            actions_seq: Shape (batch, H, m) - consecutive actions.
            next_states_seq: Shape (batch, H, d) - consecutive next states.
            discount: Discount for multi-step error (lambda^h).
            teacher_forcing_ratio: Probability of using true state as input.
            **model_kwargs: Forwarded to model (e.g. building_id or context).
        """
        self.model.train()
        batch_size, horizon, _ = states_seq.shape

        total_mse = torch.tensor(0.0, device=states_seq.device)
        current = states_seq[:, 0]  # (batch, d)

        for h in range(horizon):
            action = actions_seq[:, h]
            target = next_states_seq[:, h]

            pred, alpha = self.model(current, action, **model_kwargs)
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
        **model_kwargs: object,
    ) -> dict[str, float]:
        """Evaluate model without gradient updates."""
        self.model.eval()
        with torch.no_grad():
            _, metrics = self.model.compute_loss(
                states,
                actions,
                next_states,
                sparsity_lambda=self.sparsity_lambda,
                **model_kwargs,
            )
        return metrics
