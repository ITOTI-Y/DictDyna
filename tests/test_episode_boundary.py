"""Tests for episode boundary handling in replay buffer and multi-step training."""

import numpy as np
import torch

from src.agent.replay_buffer import ReplayBuffer
from src.world_model.dict_dynamics import DictDynamicsModel
from src.world_model.model_trainer import WorldModelTrainer
from src.world_model.sparse_encoder import SparseEncoder


class TestSampleSequenceEpisodeBoundary:
    """Verify sample_sequence() never crosses episode boundaries."""

    def _make_buffer_with_episodes(
        self, ep_lengths: list[int], state_dim: int = 4, action_dim: int = 2
    ) -> ReplayBuffer:
        """Create buffer with multiple episodes of given lengths."""
        buf = ReplayBuffer(
            capacity=sum(ep_lengths) + 100, state_dim=state_dim, action_dim=action_dim
        )
        for ep_len in ep_lengths:
            for step in range(ep_len):
                done = step == ep_len - 1
                buf.add(
                    state=np.random.randn(state_dim).astype(np.float32),
                    action=np.random.randn(action_dim).astype(np.float32),
                    reward=0.0,
                    next_state=np.random.randn(state_dim).astype(np.float32),
                    done=done,
                )
        return buf

    def test_no_cross_episode(self):
        """Sampled sequences must not contain done=True except possibly at last step."""
        buf = self._make_buffer_with_episodes([50, 50])  # 2 episodes
        seq_len = 5
        batch = buf.sample_sequence(batch_size=32, seq_len=seq_len)
        dones = batch["dones"].numpy()  # (32, 5, 1)

        for i in range(32):
            # Interior steps (0..seq_len-2) must all be done=False
            interior_dones = dones[i, : seq_len - 1].flatten()
            assert np.all(interior_dones < 0.5), (
                f"Sequence {i} crosses episode boundary: dones={dones[i].flatten()}"
            )

    def test_short_episodes(self):
        """Buffer with very short episodes still works."""
        buf = self._make_buffer_with_episodes([3, 3, 3, 3, 100])
        batch = buf.sample_sequence(batch_size=8, seq_len=3)
        dones = batch["dones"].numpy()
        for i in range(8):
            interior_dones = dones[i, :2].flatten()
            assert np.all(interior_dones < 0.5)

    def test_seq_len_1_always_valid(self):
        """seq_len=1 should always work regardless of episode boundaries."""
        buf = self._make_buffer_with_episodes([1, 1, 1, 10])
        batch = buf.sample_sequence(batch_size=5, seq_len=1)
        assert batch["states"].shape == (5, 1, 4)

    def test_circular_wrap(self):
        """Buffer that wraps around should not produce invalid sequences."""
        buf = ReplayBuffer(capacity=20, state_dim=2, action_dim=1)
        # Fill 30 transitions (wraps around at capacity=20)
        for i in range(30):
            done = i == 14 or i == 29  # episode boundaries
            buf.add(
                np.array([float(i), 0.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                0.0,
                np.array([float(i + 1), 0.0], dtype=np.float32),
                done,
            )
        batch = buf.sample_sequence(batch_size=10, seq_len=3)
        dones = batch["dones"].numpy()
        for i in range(10):
            assert np.all(dones[i, :2].flatten() < 0.5)


class TestTrainMultistepDoneMask:
    """Verify train_multistep masks loss at episode boundaries."""

    def _make_model_and_trainer(self, state_dim=4, action_dim=2, n_atoms=16):
        torch.manual_seed(42)
        dictionary = torch.randn(state_dim, n_atoms)
        dictionary = dictionary / dictionary.norm(dim=0, keepdim=True)
        encoder = SparseEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            n_atoms=n_atoms,
            shared_hidden_dims=[32],
            sparsity_method="topk",
            topk_k=4,
        )
        model = DictDynamicsModel(
            dictionary=dictionary, sparse_encoder=encoder, learnable_dict=True
        )
        trainer = WorldModelTrainer(model=model, encoder_lr=1e-3, dict_lr=0)
        return model, trainer

    def test_done_mask_reduces_effective_steps(self):
        """When done=True at step 1, steps 2+ should be masked."""
        _, trainer = self._make_model_and_trainer()

        batch_size, horizon, state_dim, action_dim = 4, 5, 4, 2
        states = torch.randn(batch_size, horizon, state_dim)
        actions = torch.randn(batch_size, horizon, action_dim)
        next_states = torch.randn(batch_size, horizon, state_dim)

        # All active
        dones_all_active = torch.zeros(batch_size, horizon)
        m1 = trainer.train_multistep(
            states, actions, next_states, dones_seq=dones_all_active
        )

        # Done at step 0 → only step 0 contributes
        dones_early = torch.zeros(batch_size, horizon)
        dones_early[:, 0] = 1.0
        m2 = trainer.train_multistep(
            states, actions, next_states, dones_seq=dones_early
        )

        # Both should run without error and produce valid metrics
        assert m1["multistep_mse"] > 0
        assert m2["multistep_mse"] > 0
        assert m2["multistep_horizon"] == horizon

    def test_no_dones_backward_compatible(self):
        """train_multistep without dones_seq should work as before."""
        _, trainer = self._make_model_and_trainer()
        states = torch.randn(4, 3, 4)
        actions = torch.randn(4, 3, 2)
        next_states = torch.randn(4, 3, 4)

        metrics = trainer.train_multistep(states, actions, next_states)
        assert "multistep_mse" in metrics
        assert metrics["multistep_horizon"] == 3
