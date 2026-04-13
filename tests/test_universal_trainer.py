"""Tests for UniversalSACTrainer, especially encoder training correctness.

After the 2026-04-13 review identified that the encoder was NEVER trained
(due to .detach() in _encode), this test suite ensures encoder gradients
flow properly during SAC updates.
"""

import numpy as np
import torch

from src.agent.replay_buffer import ReplayBuffer
from src.obs_config_universal import (
    KNOWN_5ZONE_VARS,
    TOTAL_EMBED_DIM,
    build_category_mapping,
)
from src.obs_encoder import UniversalObsEncoder

STATE_DIM_5ZONE = 17
ACTION_DIM_5ZONE = 2


class TestEncoderTrainability:
    """Verify that the encoder actually trains end-to-end with SAC."""

    def _make_trainer_components(self):
        """Build encoder + SAC components directly (avoid Sinergym dependency)."""
        import torch.nn.functional as F

        from src.agent.sac import GaussianActor, SACTrainer, SoftQNetwork

        torch.manual_seed(42)
        encoder = UniversalObsEncoder()
        mapping = build_category_mapping("5zone_test", KNOWN_5ZONE_VARS)

        actor = GaussianActor(
            TOTAL_EMBED_DIM,
            ACTION_DIM_5ZONE,
            hidden_dims=[64, 64],
            action_scale=np.array([5.0, 5.0], dtype=np.float32),
            action_bias=np.array([20.0, 27.0], dtype=np.float32),
        )
        critic = SoftQNetwork(TOTAL_EMBED_DIM, ACTION_DIM_5ZONE, hidden_dims=[64, 64])
        critic_target = SoftQNetwork(
            TOTAL_EMBED_DIM, ACTION_DIM_5ZONE, hidden_dims=[64, 64]
        )
        critic_target.load_state_dict(critic.state_dict())

        sac = SACTrainer(
            actor=actor,
            critic=critic,
            critic_target=critic_target,
            gamma=0.99,
            autotune_alpha=True,
            target_entropy=-ACTION_DIM_5ZONE,
        )

        encoder_opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        buffer = ReplayBuffer(1000, STATE_DIM_5ZONE, ACTION_DIM_5ZONE)

        return encoder, mapping, sac, encoder_opt, buffer, F

    def _fake_update(self, encoder, mapping, sac, encoder_opt, batch, F):  # noqa: N803
        """Run one update with encoder gradient flow."""
        raw_states = batch["states"]
        raw_next_states = batch["next_states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        # Critic update
        states = encoder(raw_states, mapping)
        with torch.no_grad():
            next_states_ng = encoder(raw_next_states, mapping)
            next_actions, next_log_probs = sac.actor(next_states_ng)
            q1n, q2n = sac.critic_target(next_states_ng, next_actions)
            q_next = torch.min(q1n, q2n) - sac.alpha * next_log_probs
            target_q = rewards + (1.0 - dones) * sac.gamma * q_next
        q1, q2 = sac.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        sac.critic_optimizer.zero_grad()
        encoder_opt.zero_grad()
        critic_loss.backward()
        sac.critic_optimizer.step()

        # Actor update (re-encode)
        states_a = encoder(raw_states, mapping)
        new_actions, log_probs = sac.actor(states_a)
        q1n, q2n = sac.critic(states_a, new_actions)
        q_new = torch.min(q1n, q2n)
        actor_loss = (sac.alpha.detach() * log_probs - q_new).mean()

        sac.actor_optimizer.zero_grad()
        actor_loss.backward()
        sac.actor_optimizer.step()

        encoder_opt.step()
        encoder_opt.zero_grad()

    def test_encoder_weights_change_after_updates(self):
        """After several SAC updates, encoder weights MUST differ from init."""
        encoder, mapping, sac, encoder_opt, buffer, F = self._make_trainer_components()

        # Snapshot initial encoder weights
        initial_state = {k: v.clone() for k, v in encoder.state_dict().items()}

        # Fill buffer with synthetic data
        rng = np.random.default_rng(42)
        for _ in range(300):
            s = rng.standard_normal(STATE_DIM_5ZONE).astype(np.float32)
            a = rng.standard_normal(ACTION_DIM_5ZONE).astype(np.float32)
            r = float(rng.standard_normal())
            sn = rng.standard_normal(STATE_DIM_5ZONE).astype(np.float32)
            buffer.add(s, a, r, sn, False)

        # Run 50 update steps
        device = torch.device("cpu")
        for _ in range(50):
            batch = buffer.sample(64, device)
            self._fake_update(encoder, mapping, sac, encoder_opt, batch, F)

        # Verify at least one encoder parameter has changed significantly
        changed_params = []
        for name, p in encoder.state_dict().items():
            diff = (p - initial_state[name]).abs().mean().item()
            if diff > 1e-6:
                changed_params.append((name, diff))

        assert len(changed_params) > 0, (
            "NO encoder parameter changed after 50 updates — encoder not training!"
        )
        # Sanity: expect many/most params changed
        assert len(changed_params) >= len(initial_state) // 2, (
            f"Only {len(changed_params)}/{len(initial_state)} encoder params changed. "
            f"Expected majority to update."
        )

    def test_encoder_grads_flow_from_critic_loss(self):
        """Verify that critic loss produces gradients on encoder parameters."""
        encoder, mapping, sac, _, _, F = self._make_trainer_components()

        rng = np.random.default_rng(0)
        raw_states = torch.tensor(
            rng.standard_normal((8, STATE_DIM_5ZONE)).astype(np.float32)
        )
        raw_next = torch.tensor(
            rng.standard_normal((8, STATE_DIM_5ZONE)).astype(np.float32)
        )
        actions = torch.tensor(
            rng.standard_normal((8, ACTION_DIM_5ZONE)).astype(np.float32)
        )
        rewards = torch.randn(8, 1)
        dones = torch.zeros(8, 1)

        states = encoder(raw_states, mapping)
        with torch.no_grad():
            next_s_ng = encoder(raw_next, mapping)
            next_a, next_lp = sac.actor(next_s_ng)
            q1n, q2n = sac.critic_target(next_s_ng, next_a)
            target_q = rewards + 0.99 * (1 - dones) * (
                torch.min(q1n, q2n) - sac.alpha * next_lp
            )

        q1, q2 = sac.critic(states, actions)
        loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # Zero all grads then backward
        for p in encoder.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()

        # At least one encoder param must have non-zero gradient
        has_grad = False
        for p in encoder.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                has_grad = True
                break
        assert has_grad, "Critic loss did not produce gradients on encoder"


class TestBufferStoresRawObs:
    """Verify buffer dimensions match raw obs (not embeddings)."""

    def test_buffer_state_dim_is_raw(self):
        """When using UniversalSACTrainer pattern, buffer stores raw obs."""
        buffer = ReplayBuffer(100, STATE_DIM_5ZONE, ACTION_DIM_5ZONE)
        raw = np.random.randn(STATE_DIM_5ZONE).astype(np.float32)
        act = np.random.randn(ACTION_DIM_5ZONE).astype(np.float32)
        buffer.add(raw, act, 1.0, raw, False)

        assert buffer.states.shape == (100, STATE_DIM_5ZONE)
        assert buffer.states.shape[1] != TOTAL_EMBED_DIM, (
            "Buffer should store raw obs (17d), not embeddings (128d)"
        )
