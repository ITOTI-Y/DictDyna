"""Tests for Dyna-SAC integration."""

import numpy as np
import torch

from src.agent.dyna_sac import DynaSAC
from src.agent.rollout import ModelRollout
from src.agent.sac import GaussianActor
from src.schemas import (
    DictionarySchema,
    DynaSchema,
    SACSchema,
    SparseEncoderSchema,
    TrainSchema,
)
from src.world_model.dict_dynamics import DictDynamicsModel
from src.world_model.reward_estimator import SinergymRewardEstimator
from src.world_model.sparse_encoder import SparseEncoder

STATE_DIM = 10
ACTION_DIM = 2
N_ATOMS = 32


def _make_rollout_gen():
    encoder = SparseEncoder(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        n_atoms=N_ATOMS,
        shared_hidden_dims=[64, 64],
        adapter_dim=32,
        n_buildings=1,
    )
    dictionary = torch.randn(STATE_DIM, N_ATOMS)
    dictionary = dictionary / dictionary.norm(dim=0, keepdim=True)
    model = DictDynamicsModel(dictionary, encoder, learnable_dict=False)

    actor = GaussianActor(STATE_DIM, ACTION_DIM, hidden_dims=[64, 64])
    reward_est = SinergymRewardEstimator()

    return ModelRollout(model, actor, reward_est)


class TestModelRollout:
    def test_rollout_shape(self):
        gen = _make_rollout_gen()
        M = 5  # number of rollouts
        H = 3  # horizon
        start_states = np.random.randn(M, STATE_DIM).astype(np.float32)

        result = gen.generate(start_states, building_id="0", horizon=H)
        assert result["states"].shape == (M * H, STATE_DIM)
        assert result["actions"].shape == (M * H, ACTION_DIM)
        assert result["rewards"].shape == (M * H, 1)
        assert result["next_states"].shape == (M * H, STATE_DIM)
        assert result["dones"].shape == (M * H, 1)


class TestDynaSAC:
    def _make_dyna(self):
        config = TrainSchema(
            batch_size=16,
            buffer_size=200,
            dictionary=DictionarySchema(
                n_atoms=N_ATOMS,
                state_dim=STATE_DIM,
                sparsity_lambda=0.1,
                pretrain_lr=1e-3,
                slow_update_lr=1e-5,
            ),
            encoder=SparseEncoderSchema(
                shared_hidden_dims=[64, 64],
                adapter_dim=32,
            ),
            dyna=DynaSchema(
                rollout_horizon=2,
                rollouts_per_step=5,
                model_to_real_ratio=0.5,
                model_update_freq=1,
            ),
            sac=SACSchema(hidden_dims=[64, 64]),
            device="cpu",
        )
        dictionary = torch.randn(STATE_DIM, N_ATOMS)
        dictionary = dictionary / dictionary.norm(dim=0, keepdim=True)

        return DynaSAC(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            building_ids=["building_0"],
            dictionary=dictionary,
            config=config,
        )

    def test_select_action(self):
        dyna = self._make_dyna()
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action = dyna.select_action(state)
        assert action.shape == (ACTION_DIM,)

    def test_training_loop_smoke(self):
        """Smoke test: run a few train steps without crashing."""
        dyna = self._make_dyna()

        # Fill buffer with some initial data
        for _ in range(20):
            s = np.random.randn(STATE_DIM).astype(np.float32)
            a = np.random.randn(ACTION_DIM).astype(np.float32)
            r = float(np.random.randn())
            s_next = np.random.randn(STATE_DIM).astype(np.float32)
            dyna.buffer.add_real(s, a, r, s_next, False)

        # Run a few train steps
        for step in range(5):
            s = np.random.randn(STATE_DIM).astype(np.float32)
            a = dyna.select_action(s)
            r = float(np.random.randn())
            s_next = np.random.randn(STATE_DIM).astype(np.float32)
            metrics = dyna.train_step(s, a, r, s_next, False, "building_0", step)
            assert isinstance(metrics, dict)

    def test_mixed_buffer(self):
        """After training, buffer should contain both real and simulated data."""
        dyna = self._make_dyna()

        for _ in range(20):
            s = np.random.randn(STATE_DIM).astype(np.float32)
            a = np.random.randn(ACTION_DIM).astype(np.float32)
            dyna.buffer.add_real(s, a, 0.0, s, False)

        # One train step should add model data
        s = np.random.randn(STATE_DIM).astype(np.float32)
        a = dyna.select_action(s)
        dyna.train_step(s, a, 0.0, s, False, "building_0", 0)

        assert dyna.buffer.real_size > 0
        assert dyna.buffer.model_size > 0

    def test_save_load(self, tmp_path):
        dyna = self._make_dyna()
        path = tmp_path / "test_checkpoint.pt"
        dyna.save(path)
        assert path.exists()

        # Load into a new instance
        dyna2 = self._make_dyna()
        dyna2.load(path)
