"""Model-based rollout generation for Dyna planning."""

import numpy as np
import torch
import torch.nn as nn

from src.agent.sac import GaussianActor, SoftQNetwork
from src.agent.sparse_exploration import SparseCodeExploration
from src.world_model.reward_estimator import SinergymRewardEstimator


class ModelRollout:
    """Generate simulated rollouts using the world model.

    Optionally adds sparse-code exploration bonus to rollout rewards.
    Supports uncertainty-aware pessimistic reward for probabilistic models.

    Args:
        world_model: DictDynamicsModel or ProbabilisticDictDynamics instance.
        actor: GaussianActor policy for action selection.
        reward_estimator: Reward estimator from predicted states.
        exploration: SparseCodeExploration for intrinsic reward bonus.
        uncertainty_penalty: Coefficient for pessimistic reward (beta).
            Reward is adjusted: r_adj = r - beta * mean(pred_std).
        device: Torch device.
    """

    def __init__(
        self,
        world_model: nn.Module,
        actor: GaussianActor,
        reward_estimator: SinergymRewardEstimator,
        exploration: SparseCodeExploration | None = None,
        uncertainty_penalty: float = 0.0,
        device: torch.device | None = None,
    ) -> None:
        self.world_model = world_model
        self.actor = actor
        self.reward_estimator = reward_estimator
        self.exploration = exploration
        self.uncertainty_penalty = uncertainty_penalty
        self.device = device or torch.device("cpu")

    @torch.no_grad()
    def generate(
        self,
        start_states: np.ndarray,
        building_id: str = "0",
        horizon: int = 3,
        context: torch.Tensor | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate simulated rollouts from start states.

        Args:
            start_states: Starting states, shape (M, d).
            building_id: Building to simulate (adapter mode).
            horizon: Number of rollout steps H.
            context: Context vector z (context mode). If provided, uses
                context-conditioned forward instead of building_id routing.

        Returns:
            Dict with keys: states, actions, rewards, next_states, dones.
            Each has shape (M * H, ...).
        """
        self.world_model.eval()
        self.actor.eval()

        states_list = []
        actions_list = []
        rewards_list = []
        next_states_list = []
        dones_list = []

        current_states = torch.tensor(
            start_states, dtype=torch.float32, device=self.device
        )

        # Expand context to match batch size if provided
        if context is not None and context.shape[0] == 1:
            context = context.expand(len(start_states), -1)

        for _ in range(horizon):
            # Select actions using policy
            actions, _ = self.actor(current_states)

            # Predict next states using world model (get alpha for exploration)
            if context is not None:
                next_states, alpha = self.world_model(current_states, actions, context)
            else:
                next_states, alpha = self.world_model(
                    current_states, actions, building_id
                )

            # Estimate rewards from predicted states + exploration bonus
            rewards = self.reward_estimator.estimate(next_states)
            if self.exploration is not None:
                rewards = rewards + self.exploration.compute_bonus(alpha)

            # Uncertainty-aware pessimistic reward for probabilistic models
            if self.uncertainty_penalty > 0 and hasattr(
                self.world_model, "get_prediction_std"
            ):
                pred_std = self.world_model.get_prediction_std()  # ty: ignore[call-non-callable]
                if pred_std is not None:
                    penalty = self.uncertainty_penalty * pred_std.mean(dim=-1)
                    rewards = rewards - penalty

            states_list.append(current_states.cpu().numpy())
            actions_list.append(actions.cpu().numpy())
            rewards_list.append(rewards.cpu().numpy())
            next_states_list.append(next_states.cpu().numpy())
            dones_list.append(np.zeros(len(start_states), dtype=np.float32))

            current_states = next_states

        return {
            "states": np.concatenate(states_list, axis=0),
            "actions": np.concatenate(actions_list, axis=0),
            "rewards": np.concatenate(rewards_list, axis=0).reshape(-1, 1),
            "next_states": np.concatenate(next_states_list, axis=0),
            "dones": np.concatenate(dones_list, axis=0).reshape(-1, 1),
        }

    @torch.no_grad()
    def compute_mve_targets(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        critic_target: SoftQNetwork,
        gamma: float,
        alpha: float,
        building_id: str = "0",
        horizon: int = 3,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Model-Based Value Expansion targets.

        Uses real (s, a, r, s') as base, then extends the value estimate
        by rolling out the model for H additional steps:

        Q_mve = r + gamma * [sum_{h=0}^{H-1} gamma^h * r_hat(h)]
                + gamma^{H+1} * Q_target(s_hat_H, a_hat_H)

        Args:
            states: Real states from replay buffer, shape (batch, d).
            actions: Real actions, shape (batch, m).
            rewards: Real rewards, shape (batch, 1).
            next_states: Real next states, shape (batch, d).
            critic_target: Target Q-network for terminal value.
            gamma: Discount factor.
            alpha: SAC entropy coefficient.
            building_id: Building identifier.
            horizon: Number of model rollout steps for value expansion.
            context: Context vector for context-conditioned models.

        Returns:
            MVE Q-targets, shape (batch, 1).
        """
        self.world_model.eval()
        self.actor.eval()

        batch_size = states.shape[0]
        current = next_states  # Start from real next states

        if context is not None and context.shape[0] == 1:
            context = context.expand(batch_size, -1)

        # Accumulate discounted model rewards
        model_return = torch.zeros(batch_size, 1, device=states.device)

        for h in range(horizon):
            model_actions, _ = self.actor(current)

            if context is not None:
                model_next, _ = self.world_model(current, model_actions, context)
            else:
                model_next, _ = self.world_model(current, model_actions, building_id)

            model_rewards = self.reward_estimator.estimate(model_next)
            model_return += (gamma**h) * model_rewards.unsqueeze(-1)
            current = model_next

        # Terminal value at model horizon
        terminal_actions, terminal_log_probs = self.actor(current)
        q1_term, q2_term = critic_target(current, terminal_actions)
        q_terminal = torch.min(q1_term, q2_term) - alpha * terminal_log_probs

        # MVE target: real_r + gamma * (model_returns + gamma^H * Q_terminal)
        mve_target = rewards + gamma * (model_return + (gamma**horizon) * q_terminal)

        return mve_target
