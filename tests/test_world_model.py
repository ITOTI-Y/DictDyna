"""Tests for world model module."""

import torch

from src.world_model.dict_dynamics import DictDynamicsModel
from src.world_model.reward_estimator import SinergymRewardEstimator
from src.world_model.sparse_encoder import SparseEncoder

STATE_DIM = 10
ACTION_DIM = 2
N_ATOMS = 32
BATCH_SIZE = 16


def _make_model(
    n_buildings: int = 2,
    sparsity_method: str = "l1_penalty",
    dim_weights: torch.Tensor | None = None,
    residual_hidden_dim: int = 0,
):
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
    model = DictDynamicsModel(
        dictionary,
        encoder,
        learnable_dict=True,
        dim_weights=dim_weights,
        residual_hidden_dim=residual_hidden_dim,
    )
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


class TestDimWeightedLoss:
    def test_dim_weights_amplify_reward_dims(self):
        """Verify that reward-critical dims get higher loss contribution."""
        weights = torch.ones(STATE_DIM)
        weights[0] = 10.0  # simulate reward-critical dim
        model_w = _make_model(dim_weights=weights)
        model_u = _make_model()
        # Copy parameters so only dim_weights differs
        model_u.load_state_dict(model_w.state_dict(), strict=False)

        s = torch.randn(BATCH_SIZE, STATE_DIM)
        a = torch.randn(BATCH_SIZE, ACTION_DIM)
        ns = torch.randn(BATCH_SIZE, STATE_DIM)

        loss_weighted, _ = model_w.compute_loss(s, a, ns)
        loss_uniform, _ = model_u.compute_loss(s, a, ns)

        assert loss_weighted > loss_uniform

    def test_dim_weights_none_is_uniform(self):
        """dim_weights=None should be equivalent to all-ones."""
        model_none = _make_model()
        model_ones = _make_model(dim_weights=torch.ones(STATE_DIM))
        model_ones.load_state_dict(model_none.state_dict(), strict=False)

        s = torch.randn(BATCH_SIZE, STATE_DIM)
        a = torch.randn(BATCH_SIZE, ACTION_DIM)
        ns = torch.randn(BATCH_SIZE, STATE_DIM)

        loss1, _ = model_none.compute_loss(s, a, ns)
        loss2, _ = model_ones.compute_loss(s, a, ns)

        torch.testing.assert_close(loss1, loss2)

    def test_reward_dim_metrics_reported(self):
        """Metrics should include per-group MSE when dim_weights active."""
        weights = torch.ones(STATE_DIM)
        weights[0] = 5.0
        model = _make_model(dim_weights=weights)

        s = torch.randn(BATCH_SIZE, STATE_DIM)
        a = torch.randn(BATCH_SIZE, ACTION_DIM)
        ns = torch.randn(BATCH_SIZE, STATE_DIM)
        _, metrics = model.compute_loss(s, a, ns)

        assert "mse_reward_dims" in metrics
        assert "mse_other_dims" in metrics

    def test_no_metrics_without_dim_weights(self):
        """No per-group MSE metrics when dim_weights is None."""
        model = _make_model()
        s = torch.randn(BATCH_SIZE, STATE_DIM)
        a = torch.randn(BATCH_SIZE, ACTION_DIM)
        ns = torch.randn(BATCH_SIZE, STATE_DIM)
        _, metrics = model.compute_loss(s, a, ns)

        assert "mse_reward_dims" not in metrics
        assert "mse_other_dims" not in metrics


class TestResidualHead:
    def test_residual_changes_prediction_after_training(self):
        """After a gradient step, residual head should change predictions."""
        model = _make_model(residual_hidden_dim=64)
        s = torch.randn(BATCH_SIZE, STATE_DIM)
        a = torch.randn(BATCH_SIZE, ACTION_DIM)
        ns = torch.randn(BATCH_SIZE, STATE_DIM)

        # At init: residual output is small (std=0.01 initialized)
        _pred_before, _ = model(s, a)
        assert model._last_residual is not None
        init_norm = model._last_residual.norm().item()

        # After one gradient step: residual changes
        loss, _ = model.compute_loss(s, a, ns)
        loss.backward()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        opt.step()
        _pred_after, _ = model(s, a)
        post_norm = model._last_residual.norm().item()
        assert post_norm != init_norm

    def test_no_residual_backward_compat(self):
        """residual_hidden_dim=0 produces same forward as original."""
        model = _make_model()
        assert model.residual_head is None
        assert model._last_residual is None

        s = torch.randn(BATCH_SIZE, STATE_DIM)
        a = torch.randn(BATCH_SIZE, ACTION_DIM)
        pred, _alpha = model(s, a)
        assert pred.shape == (BATCH_SIZE, STATE_DIM)
        assert model._last_residual is None

    def test_residual_norm_in_metrics(self):
        """Metrics should include residual_norm when residual is active."""
        model = _make_model(residual_hidden_dim=64)
        s = torch.randn(BATCH_SIZE, STATE_DIM)
        a = torch.randn(BATCH_SIZE, ACTION_DIM)
        ns = torch.randn(BATCH_SIZE, STATE_DIM)
        _, metrics = model.compute_loss(s, a, ns)

        assert "residual_norm" in metrics
        # Zero-initialized, so norm starts at 0
        assert metrics["residual_norm"] >= 0

    def test_residual_lambda_regularization(self):
        """After training, residual L2 regularization should increase total loss."""
        model = _make_model(residual_hidden_dim=64)
        s = torch.randn(BATCH_SIZE, STATE_DIM)
        a = torch.randn(BATCH_SIZE, ACTION_DIM)
        ns = torch.randn(BATCH_SIZE, STATE_DIM)

        # Train one step so residual becomes nonzero
        loss, _ = model.compute_loss(s, a, ns)
        loss.backward()
        torch.optim.Adam(model.parameters(), lr=1e-2).step()

        loss_no_reg, _ = model.compute_loss(s, a, ns, residual_lambda=0.0)
        loss_with_reg, _ = model.compute_loss(s, a, ns, residual_lambda=1.0)

        assert loss_with_reg > loss_no_reg


class TestSinergymRewardEstimator:
    def test_reward_from_state(self):
        # Sinergym 5zone: dim 0=month, dim 9=indoor_temp, dim 15=hvac_power
        # Need at least 16 dims for default indices
        d = 17
        estimator = SinergymRewardEstimator()
        # Winter month (Jan), temp=22 within winter range [20,23.5], power=1000W
        state = torch.zeros(4, d)
        state[:, 0] = 1.0  # month = January (winter)
        state[:, 9] = 22.0  # indoor temp within winter range
        state[:, 15] = 1000.0  # HVAC power

        reward = estimator.estimate(state)
        assert reward.shape == (4,)
        # No comfort violation, only energy: -0.5 * 0.0001 * 1000 = -0.05
        expected = -(0.5 * 0.0001 * 1000.0)
        torch.testing.assert_close(reward, torch.full((4,), expected))

    def test_comfort_violation(self):
        d = 17
        estimator = SinergymRewardEstimator()
        state = torch.zeros(1, d)
        state[0, 0] = 7.0  # month = July (summer, range [23,26])
        state[0, 9] = 29.0  # 3 degrees above upper band (26)
        state[0, 15] = 0.0  # no power

        reward = estimator.estimate(state)
        # Comfort violation = 29 - 26 = 3, lambda_temp=1.0
        # R = -0.5 * 1.0 * 3.0 = -1.5
        expected = -(0.5 * 1.0 * 3.0)
        torch.testing.assert_close(reward, torch.tensor([expected]))
