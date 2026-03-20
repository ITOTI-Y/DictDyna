"""Tests for world model module."""

import torch

from src.world_model.dict_dynamics import DictDynamicsModel
from src.world_model.reward_estimator import SinergymRewardEstimator
from src.world_model.sparse_encoder import SparseEncoder

STATE_DIM = 10
ACTION_DIM = 2
N_ATOMS = 32
BATCH_SIZE = 16


def _make_model(n_buildings: int = 2, sparsity_method: str = "l1_penalty"):
    encoder = SparseEncoder(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        n_atoms=N_ATOMS,
        shared_hidden_dims=[64, 64],
        adapter_dim=32,
        n_buildings=n_buildings,
        sparsity_method=sparsity_method,
    )
    dictionary = torch.randn(STATE_DIM, N_ATOMS)
    # Normalize columns
    dictionary = dictionary / dictionary.norm(dim=0, keepdim=True)
    model = DictDynamicsModel(dictionary, encoder, learnable_dict=True)
    return model


class TestSparseEncoder:
    def test_output_shape(self):
        encoder = SparseEncoder(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            n_atoms=N_ATOMS,
            n_buildings=2,
        )
        s = torch.randn(BATCH_SIZE, STATE_DIM)
        a = torch.randn(BATCH_SIZE, ACTION_DIM)
        alpha = encoder(s, a, building_id="0")
        assert alpha.shape == (BATCH_SIZE, N_ATOMS)

    def test_sparsity_topk(self):
        encoder = SparseEncoder(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            n_atoms=N_ATOMS,
            n_buildings=1,
            sparsity_method="topk",
            topk_k=8,
        )
        s = torch.randn(BATCH_SIZE, STATE_DIM)
        a = torch.randn(BATCH_SIZE, ACTION_DIM)
        alpha = encoder(s, a, building_id="0")
        # Each sample should have exactly topk_k nonzero values
        for i in range(BATCH_SIZE):
            nnz = (alpha[i].abs() > 1e-8).sum().item()
            assert nnz == 8, f"Sample {i} has {nnz} nonzeros, expected 8"

    def test_adapter_isolation(self):
        encoder = SparseEncoder(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            n_atoms=N_ATOMS,
            n_buildings=2,
        )
        s = torch.randn(1, STATE_DIM)
        a = torch.randn(1, ACTION_DIM)
        alpha_0 = encoder(s, a, building_id="0")
        alpha_1 = encoder(s, a, building_id="1")
        # Different adapters should produce different outputs
        assert not torch.allclose(alpha_0, alpha_1)


class TestDictDynamicsModel:
    def test_forward_residual(self):
        model = _make_model()
        s = torch.randn(BATCH_SIZE, STATE_DIM)
        a = torch.randn(BATCH_SIZE, ACTION_DIM)
        pred_next, alpha = model(s, a, building_id="0")

        # Verify residual structure: pred = s + D @ alpha
        expected = s + alpha @ model.dictionary.T
        torch.testing.assert_close(pred_next, expected)

    def test_normalize_atoms(self):
        model = _make_model()
        # Manually un-normalize
        model.dictionary.data *= 3.0
        model.normalize_atoms()
        norms = torch.norm(model.dictionary, dim=0)
        torch.testing.assert_close(norms, torch.ones(N_ATOMS), atol=1e-5, rtol=0)

    def test_compute_loss(self):
        model = _make_model()
        s = torch.randn(BATCH_SIZE, STATE_DIM)
        a = torch.randn(BATCH_SIZE, ACTION_DIM)
        s_next = torch.randn(BATCH_SIZE, STATE_DIM)

        loss, metrics = model.compute_loss(s, a, s_next, building_id="0")
        assert loss.requires_grad
        assert "mse_loss" in metrics
        assert "l1_loss" in metrics
        assert "sparsity" in metrics

    def test_predict(self):
        model = _make_model()
        s = torch.randn(BATCH_SIZE, STATE_DIM)
        a = torch.randn(BATCH_SIZE, ACTION_DIM)
        pred = model.predict(s, a, building_id="0")
        assert pred.shape == (BATCH_SIZE, STATE_DIM)


class TestSinergymRewardEstimator:
    def test_reward_from_state(self):
        estimator = SinergymRewardEstimator(
            comfort_weight=0.5,
            temp_target=23.0,
            temp_band=2.0,
            state_indices={"indoor_temp": 4, "hvac_power": 5},
        )
        # Create state where temp is within comfort band and power is low
        state = torch.zeros(4, STATE_DIM)
        state[:, 4] = 23.0  # indoor_temp = target
        state[:, 5] = 0.5  # low hvac power

        reward = estimator.estimate(state)
        assert reward.shape == (4,)
        # No comfort violation, only energy cost
        expected = -(0.5 * 0.5)  # -energy_weight * power
        torch.testing.assert_close(reward, torch.full((4,), expected))

    def test_comfort_violation(self):
        estimator = SinergymRewardEstimator(
            comfort_weight=0.5,
            temp_target=23.0,
            temp_band=2.0,
            state_indices={"indoor_temp": 0, "hvac_power": 1},
        )
        state = torch.zeros(1, STATE_DIM)
        state[0, 0] = 28.0  # 3 degrees above upper band (25)
        state[0, 1] = 0.0  # no power

        reward = estimator.estimate(state)
        # Comfort violation = 28 - 25 = 3, energy = 0
        expected = -(0.5 * 3.0)
        torch.testing.assert_close(reward, torch.tensor([expected]))
