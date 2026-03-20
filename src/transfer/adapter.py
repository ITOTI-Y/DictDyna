"""Few-shot transfer via adapter fine-tuning."""

import torch
from loguru import logger

from src.world_model.dict_dynamics import DictDynamicsModel


class AdapterTransfer:
    """Few-shot transfer to a new building.

    Freezes the shared dictionary D and encoder theta,
    then trains only the new building's adapter phi_j.

    Args:
        model: Trained DictDynamicsModel with existing adapters.
        freeze_dictionary: Whether to freeze D.
        freeze_shared_encoder: Whether to freeze shared trunk theta.
        adapter_lr: Learning rate for new adapter.
        sparsity_lambda: L1 sparsity weight.
    """

    def __init__(
        self,
        model: DictDynamicsModel,
        freeze_dictionary: bool = True,
        freeze_shared_encoder: bool = True,
        adapter_lr: float = 1e-3,
        sparsity_lambda: float = 0.1,
    ) -> None:
        self.model = model
        self.freeze_dictionary = freeze_dictionary
        self.freeze_shared_encoder = freeze_shared_encoder
        self.adapter_lr = adapter_lr
        self.sparsity_lambda = sparsity_lambda

    def add_building(
        self,
        building_id: str,
        init_from: str | None = None,
    ) -> None:
        """Add a new building adapter.

        Args:
            building_id: New building identifier.
            init_from: Existing building_id to copy adapter weights from.
        """
        self.model.encoder.add_adapter(building_id)

        if init_from and init_from in self.model.encoder.adapters:
            src_state = self.model.encoder.adapters[init_from].state_dict()
            self.model.encoder.adapters[building_id].load_state_dict(src_state)
            logger.info(f"Initialized adapter '{building_id}' from '{init_from}'")

    def adapt(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        building_id: str,
        n_epochs: int = 50,
    ) -> list[dict[str, float]]:
        """Fine-tune adapter on new building data.

        Args:
            states: States from new building, shape (N, d).
            actions: Actions, shape (N, m).
            next_states: Next states, shape (N, d).
            building_id: New building identifier.
            n_epochs: Number of adaptation epochs.

        Returns:
            List of per-epoch metrics.
        """
        # Freeze parameters
        if self.freeze_dictionary and isinstance(
            self.model.dictionary, torch.nn.Parameter
        ):
            self.model.dictionary.requires_grad_(False)

        if self.freeze_shared_encoder:
            for param in self.model.encoder.get_shared_params():
                param.requires_grad_(False)
            # Freeze all existing adapters
            for bid, adapter in self.model.encoder.adapters.items():
                if bid != building_id:
                    for param in adapter.parameters():
                        param.requires_grad_(False)

        # Only optimize new adapter params
        adapter_params = self.model.encoder.get_adapter_params(building_id)
        optimizer = torch.optim.Adam(adapter_params, lr=self.adapter_lr)

        history = []
        for epoch in range(n_epochs):
            self.model.train()
            loss, metrics = self.model.compute_loss(
                states, actions, next_states, building_id, self.sparsity_lambda
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history.append(metrics)
            if epoch % 10 == 0:
                logger.info(
                    f"Adaptation epoch {epoch}: "
                    f"MSE={metrics['mse_loss']:.6f}, "
                    f"L1={metrics['l1_loss']:.6f}"
                )

        # Restore grad flags
        for param in self.model.parameters():
            param.requires_grad_(True)

        return history
