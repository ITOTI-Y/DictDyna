"""CleanRL-style pure PyTorch SAC implementation."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ObsNormLayer(nn.Module):
    """Fixed observation normalization layer (non-learnable).

    Normalizes input: (x - mean) / std, clamped to [-clip, clip].
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        obs_mean: np.ndarray | torch.Tensor,
        obs_std: np.ndarray | torch.Tensor,
        clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(obs_std, dtype=torch.float32))
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (x - self.mean) / self.std,
            -self.clip,
            self.clip,
        )


class SoftQNetwork(nn.Module):
    """Twin Q-networks for SAC.

    Args:
        state_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_dims: Hidden layer sizes.
        obs_mean: If provided, normalize state inputs.
        obs_std: If provided, normalize state inputs.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        obs_mean: np.ndarray | None = None,
        obs_std: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]

        self.obs_norm = (
            ObsNormLayer(obs_mean, obs_std)
            if obs_mean is not None and obs_std is not None
            else None
        )

        # Q1
        q1_layers: list[nn.Module] = []
        in_dim = state_dim + action_dim
        for h in hidden_dims:
            q1_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        q1_layers.append(nn.Linear(in_dim, 1))
        self.q1 = nn.Sequential(*q1_layers)

        # Q2 (identical architecture, independent weights)
        q2_layers: list[nn.Module] = []
        in_dim = state_dim + action_dim
        for h in hidden_dims:
            q2_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        q2_layers.append(nn.Linear(in_dim, 1))
        self.q2 = nn.Sequential(*q2_layers)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.obs_norm is not None:
            state = self.obs_norm(state)
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class GaussianActor(nn.Module):
    """Squashed Gaussian policy for continuous SAC.

    Args:
        state_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_dims: Hidden layer sizes.
        action_scale: Scale for tanh squashing (scalar or per-dim).
        action_bias: Bias for tanh squashing (scalar or per-dim).
        obs_mean: If provided, normalize state inputs.
        obs_std: If provided, normalize state inputs.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        action_scale: float | np.ndarray = 1.0,
        action_bias: float | np.ndarray = 0.0,
        obs_mean: np.ndarray | None = None,
        obs_std: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]

        self.obs_norm = (
            ObsNormLayer(obs_mean, obs_std)
            if obs_mean is not None and obs_std is not None
            else None
        )

        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)
        # Support per-dimension scale/bias
        self.register_buffer(
            "action_scale",
            torch.as_tensor(action_scale, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor(action_bias, dtype=torch.float32),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability."""
        if self.obs_norm is not None:
            state = self.obs_norm(state)
        h = self.trunk(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mean, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias  # ty: ignore[unsupported-operator]

        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)  # ty: ignore[unsupported-operator]
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get deterministic action (mean) for evaluation."""
        if self.obs_norm is not None:
            state = self.obs_norm(state)
        h = self.trunk(state)
        mean = self.mean_head(h)
        return torch.tanh(mean) * self.action_scale + self.action_bias  # ty: ignore[unsupported-operator]


class SACTrainer:
    """SAC training logic (CleanRL-style).

    Args:
        actor: GaussianActor policy.
        critic: SoftQNetwork twin critics.
        critic_target: Target SoftQNetwork.
        actor_lr: Actor learning rate.
        critic_lr: Critic learning rate.
        alpha_lr: Temperature learning rate.
        tau: Target network soft update rate.
        gamma: Discount factor.
        autotune_alpha: Whether to auto-tune temperature.
        initial_alpha: Initial temperature value.
        target_entropy: Target entropy (default: -action_dim).
        device: Torch device.
    """

    def __init__(
        self,
        actor: GaussianActor,
        critic: SoftQNetwork,
        critic_target: SoftQNetwork,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        tau: float = 0.005,
        gamma: float = 0.99,
        autotune_alpha: bool = True,
        initial_alpha: float = 0.2,
        target_entropy: float | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.tau = tau
        self.gamma = gamma
        self.device = device or torch.device("cpu")

        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

        self.autotune_alpha = autotune_alpha
        if autotune_alpha:
            self.log_alpha = torch.tensor(
                [float(np.log(initial_alpha))],
                requires_grad=True,
                device=self.device,
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
            self.target_entropy = target_entropy
        else:
            self.log_alpha = torch.tensor(
                [float(np.log(initial_alpha))], device=self.device
            )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        mve_target_q: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Single SAC update step.

        Args:
            mve_target_q: Pre-computed MVE Q-targets. If provided, bypasses
                standard TD target computation for the critic update.
        """
        # --- Critic update ---
        if mve_target_q is not None:
            target_q = mve_target_q
        else:
            with torch.no_grad():
                next_actions, next_log_probs = self.actor(next_states)
                q1_next, q2_next = self.critic_target(next_states, next_actions)
                q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
                target_q = rewards + (1.0 - dones) * self.gamma * q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        new_actions, log_probs = self.actor(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Alpha update ---
        alpha_loss = 0.0
        if self.autotune_alpha:
            alpha_loss_t = (
                -self.log_alpha.exp() * (log_probs.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss_t.backward()
            self.alpha_optimizer.step()
            alpha_loss = alpha_loss_t.item()

        # --- Target network soft update ---
        self._soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss,
            "alpha": self.alpha.item(),
        }

    def _soft_update(self) -> None:
        """Soft update target network parameters."""
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters(), strict=True
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
