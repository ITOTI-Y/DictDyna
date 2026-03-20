"""Tests for SAC agent module."""

import numpy as np
import torch

from src.agent.replay_buffer import MixedReplayBuffer, ReplayBuffer
from src.agent.sac import GaussianActor, SACTrainer, SoftQNetwork

STATE_DIM = 10
ACTION_DIM = 2
BATCH_SIZE = 32


class TestSoftQNetwork:
    def test_output_shape(self):
        q = SoftQNetwork(STATE_DIM, ACTION_DIM)
        s = torch.randn(BATCH_SIZE, STATE_DIM)
        a = torch.randn(BATCH_SIZE, ACTION_DIM)
        q1, q2 = q(s, a)
        assert q1.shape == (BATCH_SIZE, 1)
        assert q2.shape == (BATCH_SIZE, 1)

    def test_twin_independence(self):
        q = SoftQNetwork(STATE_DIM, ACTION_DIM)
        s = torch.randn(BATCH_SIZE, STATE_DIM)
        a = torch.randn(BATCH_SIZE, ACTION_DIM)
        q1, q2 = q(s, a)
        # Q1 and Q2 should produce different values (independent weights)
        assert not torch.allclose(q1, q2)


class TestGaussianActor:
    def test_action_bounds(self):
        actor = GaussianActor(STATE_DIM, ACTION_DIM, action_scale=1.0)
        s = torch.randn(BATCH_SIZE, STATE_DIM)
        action, _log_prob = actor(s)
        assert action.shape == (BATCH_SIZE, ACTION_DIM)
        # Squashed Gaussian: action should be in (-scale, +scale)
        assert (action.abs() < 1.0 + 1e-5).all()

    def test_log_prob_shape(self):
        actor = GaussianActor(STATE_DIM, ACTION_DIM)
        s = torch.randn(BATCH_SIZE, STATE_DIM)
        _, log_prob = actor(s)
        assert log_prob.shape == (BATCH_SIZE, 1)

    def test_deterministic_action(self):
        actor = GaussianActor(STATE_DIM, ACTION_DIM, action_scale=1.0)
        s = torch.randn(1, STATE_DIM)
        det_action = actor.get_action(s)
        assert det_action.shape == (1, ACTION_DIM)
        assert (det_action.abs() <= 1.0 + 1e-5).all()


class TestSACTrainer:
    def test_update_step(self):
        actor = GaussianActor(STATE_DIM, ACTION_DIM)
        critic = SoftQNetwork(STATE_DIM, ACTION_DIM)
        critic_target = SoftQNetwork(STATE_DIM, ACTION_DIM)
        critic_target.load_state_dict(critic.state_dict())

        trainer = SACTrainer(
            actor,
            critic,
            critic_target,
            autotune_alpha=True,
            target_entropy=-ACTION_DIM,
        )

        metrics = trainer.update(
            states=torch.randn(BATCH_SIZE, STATE_DIM),
            actions=torch.randn(BATCH_SIZE, ACTION_DIM),
            rewards=torch.randn(BATCH_SIZE, 1),
            next_states=torch.randn(BATCH_SIZE, STATE_DIM),
            dones=torch.zeros(BATCH_SIZE, 1),
        )
        assert "critic_loss" in metrics
        assert "actor_loss" in metrics
        assert "alpha" in metrics


class TestReplayBuffer:
    def test_add_and_sample(self):
        buf = ReplayBuffer(100, STATE_DIM, ACTION_DIM)
        for _ in range(50):
            buf.add(
                np.random.randn(STATE_DIM).astype(np.float32),
                np.random.randn(ACTION_DIM).astype(np.float32),
                1.0,
                np.random.randn(STATE_DIM).astype(np.float32),
                False,
            )
        assert len(buf) == 50

        batch = buf.sample(16)
        assert batch["states"].shape == (16, STATE_DIM)
        assert batch["actions"].shape == (16, ACTION_DIM)

    def test_circular_overwrite(self):
        buf = ReplayBuffer(10, STATE_DIM, ACTION_DIM)
        for _ in range(20):
            buf.add(
                np.random.randn(STATE_DIM).astype(np.float32),
                np.random.randn(ACTION_DIM).astype(np.float32),
                0.0,
                np.random.randn(STATE_DIM).astype(np.float32),
                False,
            )
        assert len(buf) == 10


class TestMixedReplayBuffer:
    def test_mixed_sampling(self):
        buf = MixedReplayBuffer(100, 100, STATE_DIM, ACTION_DIM)
        for _ in range(30):
            buf.add_real(
                np.random.randn(STATE_DIM).astype(np.float32),
                np.random.randn(ACTION_DIM).astype(np.float32),
                1.0,
                np.random.randn(STATE_DIM).astype(np.float32),
                False,
            )
            buf.add_model(
                np.random.randn(STATE_DIM).astype(np.float32),
                np.random.randn(ACTION_DIM).astype(np.float32),
                0.5,
                np.random.randn(STATE_DIM).astype(np.float32),
                False,
            )
        assert buf.real_size == 30
        assert buf.model_size == 30

        batch = buf.sample(20, model_ratio=0.5)
        assert batch["states"].shape == (20, STATE_DIM)
