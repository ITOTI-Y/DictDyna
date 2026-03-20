"""Tests for transfer learning module."""

import torch

from src.transfer.adapter import AdapterTransfer
from src.world_model.dict_dynamics import DictDynamicsModel
from src.world_model.sparse_encoder import SparseEncoder

STATE_DIM = 10
ACTION_DIM = 2
N_ATOMS = 32


def _make_model():
    encoder = SparseEncoder(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        n_atoms=N_ATOMS,
        shared_hidden_dims=[64, 64],
        adapter_dim=32,
        n_buildings=2,
    )
    dictionary = torch.randn(STATE_DIM, N_ATOMS)
    dictionary = dictionary / dictionary.norm(dim=0, keepdim=True)
    return DictDynamicsModel(dictionary, encoder, learnable_dict=True)


class TestAdapterTransfer:
    def test_freeze_params(self):
        """During transfer, only new adapter should have gradients."""
        model = _make_model()
        transfer = AdapterTransfer(
            model,
            freeze_dictionary=True,
            freeze_shared_encoder=True,
        )

        # Add new building
        transfer.add_building("new_building")

        # Create some fake data
        states = torch.randn(16, STATE_DIM)
        actions = torch.randn(16, ACTION_DIM)
        next_states = torch.randn(16, STATE_DIM)

        # Run adaptation
        history = transfer.adapt(
            states,
            actions,
            next_states,
            building_id="new_building",
            n_epochs=3,
        )
        assert len(history) == 3
        assert "mse_loss" in history[0]

    def test_init_from_existing(self):
        """Test initializing new adapter from existing building."""
        model = _make_model()
        transfer = AdapterTransfer(model)

        # Get weights from building "0"
        original_weights = {
            k: v.clone() for k, v in model.encoder.adapters["0"].state_dict().items()
        }

        # Add new building initialized from "0"
        transfer.add_building("new", init_from="0")

        # Check weights match
        new_weights = model.encoder.adapters["new"].state_dict()
        for k in original_weights:
            torch.testing.assert_close(original_weights[k], new_weights[k])

    def test_loss_decreases(self):
        """Adaptation should reduce loss on training data."""
        model = _make_model()
        transfer = AdapterTransfer(
            model,
            adapter_lr=1e-2,
            sparsity_lambda=0.01,
        )
        transfer.add_building("target")

        # Create structured data (not pure noise)
        states = torch.randn(50, STATE_DIM)
        actions = torch.randn(50, ACTION_DIM)
        next_states = states + 0.1 * torch.randn(50, STATE_DIM)

        history = transfer.adapt(
            states,
            actions,
            next_states,
            building_id="target",
            n_epochs=50,
        )
        assert history[-1]["mse_loss"] < history[0]["mse_loss"]
