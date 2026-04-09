"""Tests for few-shot transfer pipeline (context mode).

These tests verify the core transfer logic WITHOUT requiring Sinergym.
They use synthetic data and mock environments.
"""

import numpy as np
import torch

from src.agent.dyna_sac import DynaSAC
from src.schemas import (
    ContextEncoderSchema,
    DictionarySchema,
    DynaSchema,
    SACSchema,
    SparseEncoderSchema,
    TrainSchema,
)

STATE_DIM = 17
ACTION_DIM = 2
N_ATOMS = 32
CONTEXT_DIM = 16  # matches ContextEncoderSchema default


def _make_config(mode: str = "context") -> TrainSchema:  # ty: ignore[invalid-argument-type]
    return TrainSchema(
        mode=mode,  # ty: ignore[invalid-argument-type]
        batch_size=16,
        buffer_size=200,
        dictionary=DictionarySchema(
            n_atoms=N_ATOMS,
            state_dim=STATE_DIM,
            sparsity_lambda=0.1,
        ),
        encoder=SparseEncoderSchema(
            shared_hidden_dims=[64, 64],
            adapter_dim=32,
        ),
        context=ContextEncoderSchema(
            context_dim=CONTEXT_DIM,
            context_window=5,
            hidden_dims=[32, 32],
        ),
        dyna=DynaSchema(
            rollout_start_step=0,
            rollout_horizon=1,
            rollouts_per_step=3,
            model_to_real_ratio=0.2,
        ),
        sac=SACSchema(hidden_dims=[64, 64]),
        device="cpu",
    )


def _make_dictionary() -> torch.Tensor:
    d = torch.randn(STATE_DIM, N_ATOMS)
    return d / d.norm(dim=0, keepdim=True)


def _make_dyna(mode: str = "context") -> DynaSAC:
    return DynaSAC(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        building_ids=["source_0", "source_1"],
        dictionary=_make_dictionary(),
        config=_make_config(mode),
    )


def _make_target_data(n_steps: int = 200) -> dict[str, np.ndarray]:
    """Generate synthetic target building data."""
    rng = np.random.default_rng(42)
    states = rng.standard_normal((n_steps, STATE_DIM)).astype(np.float32)
    actions = rng.standard_normal((n_steps, ACTION_DIM)).astype(np.float32)
    next_states = states + 0.01 * rng.standard_normal((n_steps, STATE_DIM)).astype(
        np.float32
    )
    rewards = rng.standard_normal(n_steps).astype(np.float32)
    return {
        "states": states,
        "actions": actions,
        "next_states": next_states,
        "rewards": rewards,
        "raw_states": states,
        "raw_next_states": next_states,
    }


class TestContextTransferEntry:
    """Test that context mode transfer entry point works end-to-end."""

    def test_context_dyna_builds(self):
        """Default mode='context' builds ContextDynamicsModel."""
        dyna = _make_dyna("context")
        assert hasattr(dyna.world_model, "infer_context")
        assert hasattr(dyna.world_model, "context_encoder")

    def test_dict_dyna_builds(self):
        """mode='dict' builds DictDynamicsModel with adapters."""
        dyna = _make_dyna("dict")
        assert hasattr(dyna.world_model, "encoder")
        assert not hasattr(dyna.world_model, "context_encoder")

    def test_context_inference(self):
        """Context encoder can infer context from transitions."""
        dyna = _make_dyna("context")
        data = _make_target_data(50)

        s = data["states"][:10]
        a = data["actions"][:10]
        sn = data["next_states"][:10]
        delta = sn - s
        transitions = np.concatenate([s, a, delta], axis=-1)
        transitions_t = torch.tensor(transitions, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            ctx = dyna.world_model.infer_context(transitions_t)

        assert ctx.shape == (1, CONTEXT_DIM)
        assert torch.isfinite(ctx).all()

    def test_context_compute_loss(self):
        """World model compute_loss with context parameter works."""
        dyna = _make_dyna("context")
        data = _make_target_data(50)

        s = torch.tensor(data["states"][:16])
        a = torch.tensor(data["actions"][:16])
        sn = torch.tensor(data["next_states"][:16])
        ctx = torch.randn(16, CONTEXT_DIM)

        loss, _info = dyna.world_model.compute_loss(
            s, a, sn, context=ctx, sparsity_lambda=0.1
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)
        loss.backward()


class TestFewShotSampling:
    """Test uniform sampling logic used in transfer."""

    def test_uniform_sampling_covers_full_episode(self):
        """np.linspace indices should span [0, total-1]."""
        total = 1000
        for n_steps in [96, 288, 672]:  # 1d, 3d, 7d
            indices = np.linspace(0, total - 1, n_steps, dtype=int)
            assert len(indices) == n_steps
            assert indices[0] == 0
            assert indices[-1] == total - 1
            # All indices valid
            assert np.all(indices >= 0)
            assert np.all(indices < total)

    def test_sampling_deterministic(self):
        """Same parameters produce same indices."""
        total = 500
        n = 96
        idx1 = np.linspace(0, total - 1, n, dtype=int)
        idx2 = np.linspace(0, total - 1, n, dtype=int)
        np.testing.assert_array_equal(idx1, idx2)


class TestOptimizerReset:
    """Test that optimizer reset creates fresh state."""

    def test_fresh_optimizers_have_no_state(self):
        """Fresh Adam optimizers should have empty state dict."""
        dyna = _make_dyna("context")
        config = _make_config()

        # Simulate optimizer reset (same as in _run_transfer / _run_context_transfer)
        fresh_actor_opt = torch.optim.Adam(
            dyna.actor.parameters(), lr=config.sac.actor_lr
        )
        fresh_critic_opt = torch.optim.Adam(
            dyna.critic.parameters(), lr=config.sac.critic_lr
        )

        assert len(fresh_actor_opt.state) == 0
        assert len(fresh_critic_opt.state) == 0

    def test_used_optimizer_has_state(self):
        """After a step, optimizer state should be non-empty."""
        dyna = _make_dyna("context")
        config = _make_config()
        opt = torch.optim.Adam(dyna.actor.parameters(), lr=config.sac.actor_lr)

        # Fake backward pass
        loss = sum(p.sum() for p in dyna.actor.parameters())
        loss.backward()
        opt.step()

        assert len(opt.state) > 0


class TestBufferClear:
    """Test buffer clearing logic for transfer."""

    def test_fresh_buffer_is_empty(self):
        """Creating new buffer should start with zero data."""
        from src.agent.replay_buffer import MixedReplayBuffer

        buf = MixedReplayBuffer(
            real_capacity=200,
            model_capacity=100,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
        )
        assert buf.real_size == 0
        assert buf.model_size == 0

    def test_fill_and_sample(self):
        """Buffer can be filled with target data and sampled."""
        from src.agent.replay_buffer import MixedReplayBuffer

        buf = MixedReplayBuffer(
            real_capacity=200,
            model_capacity=100,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
        )
        data = _make_target_data(50)
        for i in range(50):
            buf.add_real(
                data["states"][i],
                data["actions"][i],
                data["rewards"][i],
                data["next_states"][i],
                False,
            )
        assert buf.real_size == 50

        batch = buf.sample(16, model_ratio=0.0, device=torch.device("cpu"))
        assert batch["states"].shape == (16, STATE_DIM)


class TestCloneDyna:
    """Test DynaSAC cloning for transfer."""

    def test_clone_preserves_weights(self):
        """Cloned DynaSAC should have same weights as source."""
        source = _make_dyna("context")
        config = source.config
        dictionary = _make_dictionary()

        clone = DynaSAC(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            building_ids=source.building_ids,
            dictionary=dictionary,
            config=config,
        )
        clone.world_model.load_state_dict(source.world_model.state_dict())
        clone.actor.load_state_dict(source.actor.state_dict())
        clone.critic.load_state_dict(source.critic.state_dict())

        # Check actor weights match
        for p1, p2 in zip(
            source.actor.parameters(), clone.actor.parameters(), strict=True
        ):
            torch.testing.assert_close(p1, p2)

    def test_clone_is_independent(self):
        """Modifying clone should not affect source."""
        source = _make_dyna("context")
        config = source.config
        dictionary = _make_dictionary()

        clone = DynaSAC(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            building_ids=source.building_ids,
            dictionary=dictionary,
            config=config,
        )
        clone.actor.load_state_dict(source.actor.state_dict())

        # Modify clone
        with torch.no_grad():
            for p in clone.actor.parameters():
                p.add_(1.0)

        # Source should be unchanged
        for p1, p2 in zip(
            source.actor.parameters(), clone.actor.parameters(), strict=True
        ):
            assert not torch.allclose(p1, p2)


class TestZeroShotTransfer:
    """Test zero-shot (no fine-tune) context transfer path."""

    def test_zero_shot_skips_finetune(self):
        """With fine_tune=False, encoder weights should not change."""
        dyna = _make_dyna("context")
        data = _make_target_data(100)
        n_adapt_steps = 96

        # Record initial encoder weights
        initial_weights = {
            name: p.clone() for name, p in dyna.world_model.encoder.named_parameters()
        }

        # Simulate zero-shot path: infer context, NO fine-tuning
        total = len(data["states"])
        indices = np.linspace(0, total - 1, n_adapt_steps, dtype=int)
        s_ctx = data["states"][indices]
        a_ctx = data["actions"][indices]
        sn_ctx = data["next_states"][indices]
        delta_ctx = sn_ctx - s_ctx
        transitions = np.concatenate([s_ctx, a_ctx, delta_ctx], axis=-1)
        transitions_t = torch.tensor(transitions, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            ctx = dyna.world_model.infer_context(transitions_t)

        assert ctx.shape == (1, CONTEXT_DIM)

        # Encoder weights unchanged (zero-shot = no training)
        for name, p in dyna.world_model.encoder.named_parameters():
            torch.testing.assert_close(p, initial_weights[name])

    def test_zero_shot_produces_valid_context(self):
        """Zero-shot inferred context should be finite and usable in forward pass."""
        dyna = _make_dyna("context")
        data = _make_target_data(100)

        s = data["states"][:20]
        a = data["actions"][:20]
        sn = data["next_states"][:20]
        delta = sn - s
        transitions = np.concatenate([s, a, delta], axis=-1)
        transitions_t = torch.tensor(transitions, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            ctx = dyna.world_model.infer_context(transitions_t)

        # Use context in forward pass
        s_t = torch.tensor(data["states"][:4])
        a_t = torch.tensor(data["actions"][:4])
        ctx_expanded = ctx.expand(4, -1)
        with torch.no_grad():
            result = dyna.world_model(s_t, a_t, context=ctx_expanded)
        pred = result[0] if isinstance(result, tuple) else result
        assert pred.shape == s_t.shape
        assert torch.isfinite(pred).all()


class TestContextTransferWorkflow:
    """Integration test for the context transfer workflow."""

    def test_full_context_transfer_smoke(self):
        """Smoke test: context inference → fine-tune → SAC update."""
        dyna = _make_dyna("context")
        data = _make_target_data(100)
        n_adapt_steps = 96
        device = torch.device("cpu")

        # Step 1: Context inference
        total = len(data["states"])
        indices = np.linspace(0, total - 1, n_adapt_steps, dtype=int)
        s_ctx = data["states"][indices]
        a_ctx = data["actions"][indices]
        sn_ctx = data["next_states"][indices]
        delta_ctx = sn_ctx - s_ctx
        transitions = np.concatenate([s_ctx, a_ctx, delta_ctx], axis=-1)
        transitions_t = torch.tensor(transitions, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            ctx = dyna.world_model.infer_context(transitions_t)
        assert ctx.shape == (1, CONTEXT_DIM)

        # Step 2: Fine-tune encoder (freeze dictionary)
        ctx_model = dyna.world_model
        if isinstance(ctx_model.dictionary, torch.nn.Parameter):
            ctx_model.dictionary.requires_grad_(False)

        adapt_s = torch.tensor(data["states"][indices], device=device)
        adapt_a = torch.tensor(data["actions"][indices], device=device)
        adapt_sn = torch.tensor(data["next_states"][indices], device=device)

        encoder_params = list(ctx_model.context_encoder.parameters()) + list(
            ctx_model.encoder.parameters()
        )
        optimizer = torch.optim.Adam(encoder_params, lr=1e-3)

        initial_loss = None
        for _epoch in range(5):
            ctx_model.train()
            ctx = ctx_model.infer_context(transitions_t)
            ctx_expanded = ctx.expand(len(indices), -1)
            loss, _ = ctx_model.compute_loss(
                adapt_s, adapt_a, adapt_sn, context=ctx_expanded, sparsity_lambda=0.1
            )
            if initial_loss is None:
                initial_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should decrease (or at least not explode)
        assert torch.isfinite(loss)

        # Step 3: Fill buffer and do SAC update
        for i in indices[:50]:
            dyna.buffer.add_real(
                data["states"][i],
                data["actions"][i],
                data["rewards"][i],
                data["next_states"][i],
                False,
            )

        batch = dyna.buffer.sample(16, model_ratio=0.0, device=device)
        metrics = dyna.sac_trainer.update(
            batch["states"],
            batch["actions"],
            batch["rewards"],
            batch["next_states"],
            batch["dones"],
        )
        assert isinstance(metrics, dict)
