"""Model-based rollout generation for Dyna planning."""

import numpy as np
import torch

from src.agent.sac import GaussianActor
from src.agent.sparse_exploration import SparseCodeExploration
from src.world_model.dict_dynamics import DictDynamicsModel
from src.world_model.reward_estimator import SinergymRewardEstimator


class ModelRollout:
    """Generate simulated rollouts using the world model.

    Optionally adds sparse-code exploration bonus to rollout rewards.

    Args:
        world_model: DictDynamicsModel instance.
        actor: GaussianActor policy for action selection.
        reward_estimator: Reward estimator from predicted states.
        exploration: SparseCodeExploration for intrinsic reward bonus.
        device: Torch device.
    """

    def __init__(
        self,
        world_model: DictDynamicsModel,
        actor: GaussianActor,
        reward_estimator: SinergymRewardEstimator,
        exploration: SparseCodeExploration | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.world_model = world_model
        self.actor = actor
        self.reward_estimator = reward_estimator
        self.exploration = exploration
        self.device = device or torch.device("cpu")

    @torch.no_grad()
    def generate(
        self,
        start_states: np.ndarray,
        building_id: str = "0",
        horizon: int = 3,
    ) -> dict[str, np.ndarray]:
        """Generate simulated rollouts from start states.

        Args:
            start_states: Starting states, shape (M, d).
            building_id: Building to simulate.
            horizon: Number of rollout steps H.

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

        for _ in range(horizon):
            # Select actions using policy
            actions, _ = self.actor(current_states)

            # Predict next states using world model (get alpha for exploration)
            next_states, alpha = self.world_model(
                current_states, actions, building_id
            )

            # Estimate rewards from predicted states + exploration bonus
            rewards = self.reward_estimator.estimate(next_states)
            if self.exploration is not None:
                rewards = rewards + self.exploration.compute_bonus(alpha)

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
