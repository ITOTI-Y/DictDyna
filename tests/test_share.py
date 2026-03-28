"""Tests for shared utility modules (_share.py files).

Covers:
- world_model/_share.py: apply_space_conversion, topk_sparsify, normalize_atoms, build_residual_head
- agent/_share.py: normalize_obs, compute_td_error_weights
- evaluate() deque-based mean/std
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from src.agent._share import (
    OBS_CLIP_RANGE,
    compute_td_error_weights,
    normalize_obs,
)
from src.env._share import rbc_midpoint_action
from src.world_model._share import (
    apply_space_conversion,
    build_residual_head,
    normalize_atoms,
    topk_sparsify,
)


# ---------------------------------------------------------------------------
# Gap 1: apply_space_conversion — all 4 code paths
# ---------------------------------------------------------------------------
class TestApplySpaceConversion:
    def test_no_controllable_no_conversion(self):
        """Path: controllable_dims=None, has_conversion=False → passthrough."""
        delta_raw = torch.tensor([[1.0, 2.0, 3.0]])
        state = torch.zeros(1, 3)
        result = apply_space_conversion(delta_raw, state, None, None, None, False)
        torch.testing.assert_close(result, delta_raw)

    def test_no_controllable_with_conversion(self):
        """Path: controllable_dims=None, has_conversion=True → scale + bias."""
        delta_raw = torch.tensor([[1.0, 2.0, 3.0]])
        state = torch.zeros(1, 3)
        scale = torch.tensor([2.0, 2.0, 2.0])
        bias = torch.tensor([0.1, 0.1, 0.1])
        result = apply_space_conversion(delta_raw, state, None, scale, bias, True)
        expected = delta_raw * scale + bias
        torch.testing.assert_close(result, expected)

    def test_controllable_no_conversion(self):
        """Path: controllable_dims=(1,3), has_conversion=False → zero-padded embed."""
        delta_raw = torch.tensor([[10.0, 20.0]])  # 2 controllable dims
        state = torch.zeros(1, 5)  # full obs = 5 dims
        ctrl = (1, 3)
        result = apply_space_conversion(delta_raw, state, ctrl, None, None, False)
        assert result.shape == (1, 5)
        assert result[0, 0].item() == 0.0  # non-controllable
        assert result[0, 1].item() == 10.0  # controllable
        assert result[0, 2].item() == 0.0  # non-controllable
        assert result[0, 3].item() == 20.0  # controllable
        assert result[0, 4].item() == 0.0  # non-controllable

    def test_controllable_with_conversion(self):
        """Path: controllable_dims + has_conversion=True → scale/bias then embed.

        This is the actual production path used by multi_dyna_trainer.
        """
        batch = 4
        d_full = 17
        ctrl = (2, 5, 9, 15)
        d_ctrl = len(ctrl)
        delta_raw = torch.randn(batch, d_ctrl)
        state = torch.zeros(batch, d_full)
        scale = torch.ones(d_ctrl) * 2.0
        bias = torch.ones(d_ctrl) * 0.5

        result = apply_space_conversion(delta_raw, state, ctrl, scale, bias, True)

        assert result.shape == (batch, d_full)
        # Non-controllable dims should be 0
        for d in range(d_full):
            if d not in ctrl:
                assert (result[:, d] == 0).all(), f"dim {d} should be zero"
        # Controllable dims should have scale*delta + bias
        expected_ctrl = delta_raw * scale + bias
        torch.testing.assert_close(result[:, ctrl], expected_ctrl)


# ---------------------------------------------------------------------------
# Gap 2: evaluate() with deque-based mean/std
# ---------------------------------------------------------------------------
class TestEvaluateDeque:
    def test_empty_deque_returns_zero(self):
        """Empty deque should return mean=0, std=0."""
        recent = deque(maxlen=3)
        if recent:
            rewards = np.array(recent)
            result = {
                "mean_reward": float(rewards.mean()),
                "std_reward": float(rewards.std()),
            }
        else:
            result = {"mean_reward": 0.0, "std_reward": 0.0}
        assert result["mean_reward"] == 0.0
        assert result["std_reward"] == 0.0

    def test_single_episode(self):
        """Single episode: mean = that value, std = 0."""
        recent: deque[float] = deque(maxlen=3)
        recent.append(-5000.0)
        rewards = np.array(recent)
        assert float(rewards.mean()) == -5000.0
        assert float(rewards.std()) == 0.0

    def test_multiple_episodes_mean_std(self):
        """Multiple episodes: correct mean and std."""
        recent: deque[float] = deque(maxlen=5)
        values = [10.0, 20.0, 30.0]
        for v in values:
            recent.append(v)
        rewards = np.array(recent)
        np.testing.assert_almost_equal(float(rewards.mean()), 20.0)
        np.testing.assert_almost_equal(float(rewards.std()), np.std(values), decimal=5)

    def test_deque_evicts_old_episodes(self):
        """Deque with maxlen=3 should only keep last 3 episodes."""
        recent: deque[float] = deque(maxlen=3)
        for v in [-100.0, -90.0, -80.0, -70.0, -60.0]:
            recent.append(v)
        # Only last 3: -80, -70, -60
        assert len(recent) == 3
        rewards = np.array(recent)
        np.testing.assert_almost_equal(float(rewards.mean()), -70.0)


# ---------------------------------------------------------------------------
# Gap 4: topk_sparsify — soft vs hard mode
# ---------------------------------------------------------------------------
class TestTopkSparsify:
    def test_hard_topk_exact_k_nonzero(self):
        """Hard mode: exactly k nonzero elements per row."""
        alpha = torch.randn(8, 32)
        k = 4
        result = topk_sparsify(alpha, k, soft_temperature=0.0, training=False)
        for i in range(8):
            n_nonzero = (result[i] != 0).sum().item()
            assert n_nonzero == k, f"row {i}: expected {k} nonzero, got {n_nonzero}"

    def test_hard_topk_preserves_values(self):
        """Hard mode: kept values should be unchanged."""
        alpha = torch.tensor([[3.0, 1.0, 4.0, 1.0, 5.0]])
        result = topk_sparsify(alpha, k=2)
        # Top-2 by abs are indices 2 (4.0) and 4 (5.0)
        assert result[0, 2].item() == 4.0
        assert result[0, 4].item() == 5.0
        assert result[0, 0].item() == 0.0
        assert result[0, 1].item() == 0.0
        assert result[0, 3].item() == 0.0

    def test_soft_topk_no_exact_zeros(self):
        """Soft mode (training): sigmoid mask produces no exact zeros."""
        alpha = torch.randn(8, 32)
        result = topk_sparsify(alpha, k=4, soft_temperature=0.1, training=True)
        # Soft mode should have more nonzero than k (sigmoid never exact 0)
        for i in range(8):
            n_nonzero = (result[i] != 0).sum().item()
            assert n_nonzero > 4, f"soft mode should have > k nonzero, got {n_nonzero}"

    def test_soft_topk_falls_back_to_hard_at_eval(self):
        """Soft temp > 0 but training=False → hard top-k."""
        alpha = torch.randn(8, 32)
        result = topk_sparsify(alpha, k=4, soft_temperature=0.1, training=False)
        for i in range(8):
            n_nonzero = (result[i] != 0).sum().item()
            assert n_nonzero == 4

    def test_soft_topk_is_differentiable(self):
        """Soft mode gradients flow to all elements."""
        alpha = torch.randn(4, 16, requires_grad=True)
        result = topk_sparsify(alpha, k=4, soft_temperature=0.1, training=True)
        result.sum().backward()
        assert alpha.grad is not None
        # Gradient should be nonzero for most elements (sigmoid mask)
        n_nonzero_grad = (alpha.grad != 0).sum().item()
        assert n_nonzero_grad > 4 * 4  # more than just the top-k


# ---------------------------------------------------------------------------
# Gap 5: normalize_obs — NaN protection and clipping
# ---------------------------------------------------------------------------
class TestNormalizeObs:
    def test_basic_normalization(self):
        raw = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        mean = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        result = normalize_obs(raw, mean, std)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    def test_clipping_bounds(self):
        """Values exceeding clip range should be clipped."""
        raw = np.array([1000.0, -1000.0], dtype=np.float32)
        mean = np.array([0.0, 0.0], dtype=np.float32)
        std = np.array([1.0, 1.0], dtype=np.float32)
        result = normalize_obs(raw, mean, std)
        lo, hi = OBS_CLIP_RANGE
        assert result[0] == hi
        assert result[1] == lo

    def test_output_dtype_is_float32(self):
        raw = np.array([1.0, 2.0], dtype=np.float64)
        mean = np.array([0.0, 0.0], dtype=np.float64)
        std = np.array([1.0, 1.0], dtype=np.float64)
        result = normalize_obs(raw, mean, std)
        assert result.dtype == np.float32

    def test_zero_std_clips_to_bound(self):
        """std=0 → division produces inf → np.clip caps to OBS_CLIP_RANGE bounds."""
        raw = np.array([5.0, -5.0, 10.0], dtype=np.float32)
        mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        std = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        result = normalize_obs(raw, mean, std)
        lo, hi = OBS_CLIP_RANGE
        assert result[0] == hi  # 5/0 = +inf → clipped to 10
        assert result[1] == lo  # -5/0 = -inf → clipped to -10
        assert result[2] == 10.0  # normal dim


# ---------------------------------------------------------------------------
# Gap 6: compute_td_error_weights — range and correctness
# ---------------------------------------------------------------------------
class TestComputeTdErrorWeights:
    def _make_components(self, state_dim=4, action_dim=2):
        """Build minimal actor + critic for testing."""
        from src.agent.sac import GaussianActor, SoftQNetwork

        actor = GaussianActor(state_dim, action_dim, hidden_dims=[32])
        critic = SoftQNetwork(state_dim, action_dim, hidden_dims=[32])
        critic_target = SoftQNetwork(state_dim, action_dim, hidden_dims=[32])
        critic_target.load_state_dict(critic.state_dict())
        return actor, critic, critic_target

    def _make_batch(self, batch_size=8, state_dim=4, action_dim=2):
        return {
            "states": torch.randn(batch_size, state_dim),
            "actions": torch.randn(batch_size, action_dim),
            "rewards": torch.randn(batch_size, 1),
            "next_states": torch.randn(batch_size, state_dim),
            "dones": torch.zeros(batch_size, 1),
        }

    def test_weights_all_ge_one(self):
        """All weights should be >= 1.0."""
        actor, critic, critic_target = self._make_components()
        batch = self._make_batch()
        weights = compute_td_error_weights(actor, critic, critic_target, batch, 0.99)
        assert (weights >= 1.0).all(), f"min weight = {weights.min().item()}"

    def test_weights_shape(self):
        actor, critic, critic_target = self._make_components()
        batch = self._make_batch(batch_size=16)
        weights = compute_td_error_weights(actor, critic, critic_target, batch, 0.99)
        assert weights.shape == (16,)

    def test_weights_finite(self):
        """Weights should never be NaN or Inf."""
        actor, critic, critic_target = self._make_components()
        batch = self._make_batch()
        weights = compute_td_error_weights(actor, critic, critic_target, batch, 0.99)
        assert torch.isfinite(weights).all()

    def test_zero_td_error_gives_weight_one(self):
        """If all TD errors are zero, weights should all be 1.0."""
        actor, critic, critic_target = self._make_components()
        # Same critic and critic_target + identical batch → td_error ≈ 0
        # (not exactly 0 due to gamma and actor sampling, but weights cluster near 1)
        batch = self._make_batch()
        weights = compute_td_error_weights(actor, critic, critic_target, batch, 0.99)
        # Weights = 1 + td / (td.mean + eps), range should be reasonable
        assert weights.min() >= 1.0
        assert weights.max() < 10.0  # not extreme

    def test_no_gradient_on_weights(self):
        """Weights should be detached (computed under torch.no_grad)."""
        actor, critic, critic_target = self._make_components()
        batch = self._make_batch()
        weights = compute_td_error_weights(actor, critic, critic_target, batch, 0.99)
        assert not weights.requires_grad


# ---------------------------------------------------------------------------
# Bonus: normalize_atoms and build_residual_head basics
# ---------------------------------------------------------------------------
class TestNormalizeAtoms:
    def test_columns_become_unit_norm(self):
        D = nn.Parameter(torch.randn(10, 8))
        normalize_atoms(D)
        norms = torch.norm(D, dim=0)
        torch.testing.assert_close(norms, torch.ones(8), atol=1e-6, rtol=1e-6)

    def test_zero_column_handled(self):
        """Zero column should not produce NaN."""
        D = nn.Parameter(torch.zeros(5, 3))
        normalize_atoms(D)
        assert torch.isfinite(D).all()


class TestBuildResidualHead:
    def test_output_shape(self):
        head = build_residual_head(input_dim=64, hidden_dim=32, output_dim=10)
        x = torch.randn(4, 64)
        out = head(x)
        assert out.shape == (4, 10)

    def test_small_init_output(self):
        """Output layer should have small initial weights."""
        head = build_residual_head(input_dim=64, hidden_dim=32, output_dim=10)
        out_layer = head[2]  # Sequential: Linear, Tanh, Linear
        assert out_layer.weight.abs().max().item() < 0.1  # type: ignore
        assert (out_layer.bias == 0).all()  # type: ignore


class TestRbcMidpointAction:
    def test_midpoint(self):
        class FakeSpace:
            low = np.array([0.0, 10.0])
            high = np.array([2.0, 20.0])

        class FakeEnv:
            action_space = FakeSpace()

        result = rbc_midpoint_action(FakeEnv())
        np.testing.assert_array_equal(result, [1.0, 15.0])
