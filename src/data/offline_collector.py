"""Offline data collection from Sinergym environments."""

import contextlib
from pathlib import Path

import gymnasium
import numpy as np
from loguru import logger

from src.utils import sinergym_workdir

with contextlib.suppress(ImportError):
    import sinergym  # noqa: F401 — registers Sinergym envs with gymnasium


class OfflineCollector:
    """Collect offline rollout data from Sinergym for dictionary pretraining.

    Runs episodes with RBC or random policy to gather (s, a, s') transitions.
    Saves both raw transitions and computed state diffs (Δs = s' - s).

    Args:
        building_configs: List of dicts with env_name and building_id.
        policy: "random" or "rbc".
        n_episodes: Number of episodes per building.
        output_dir: Directory to save raw transitions.
        diffs_dir: Directory to save state diffs for dictionary pretraining.
    """

    def __init__(
        self,
        building_configs: list[dict],
        policy: str = "rbc",
        n_episodes: int = 2,
        output_dir: str = "data/offline_rollouts",
        diffs_dir: str = "data/processed/state_diffs",
    ) -> None:
        self.building_configs = building_configs
        self.policy = policy
        self.n_episodes = n_episodes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.diffs_dir = Path(diffs_dir)
        self.diffs_dir.mkdir(parents=True, exist_ok=True)

    def collect(self) -> dict[str, dict[str, np.ndarray]]:
        """Collect transitions from all buildings.

        Returns:
            Dict mapping building_id to {states, actions, next_states, rewards, diffs}.
        """
        all_data: dict[str, dict[str, np.ndarray]] = {}
        for cfg in self.building_configs:
            bid = cfg["building_id"]
            env_name = cfg["env_name"]
            logger.info(f"Collecting data for {bid} ({env_name})")

            data = self._collect_building(env_name)
            all_data[bid] = data

            # Save raw transitions
            np.savez(
                self.output_dir / f"{bid}_transitions.npz",
                states=data["states"],
                actions=data["actions"],
                next_states=data["next_states"],
                rewards=data["rewards"],
                dones=data["dones"],
            )
            # Save state diffs (for dictionary pretraining)
            np.save(self.diffs_dir / f"{bid}_state_diffs.npy", data["diffs"])

            logger.info(
                f"  {bid}: {len(data['states'])} transitions, "
                f"state_dim={data['states'].shape[1]}, "
                f"action_dim={data['actions'].shape[1]}"
            )

        # Print summary
        total = sum(len(d["states"]) for d in all_data.values())
        logger.info(
            f"Collection complete: {total} total transitions "
            f"from {len(all_data)} buildings"
        )
        return all_data

    def _collect_building(self, env_name: str) -> dict[str, np.ndarray]:
        """Collect transitions from a single building."""
        states, actions, next_states, rewards_list, dones_list = [], [], [], [], []

        for ep in range(self.n_episodes):
            with sinergym_workdir():
                env = gymnasium.make(env_name)
                obs, _ = env.reset(seed=ep)
                done = False
                ep_steps = 0
                ep_reward = 0.0

                while not done:
                    if self.policy == "random":
                        action = env.action_space.sample()
                    else:
                        action = self._rbc_action(obs, env)

                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    done = terminated or truncated

                    states.append(obs)
                    actions.append(action)
                    next_states.append(next_obs)
                    rewards_list.append(reward)
                    dones_list.append(float(done))

                    obs = next_obs
                    ep_steps += 1
                    ep_reward += float(reward)

                env.close()
            logger.info(
                f"    Episode {ep + 1}/{self.n_episodes}: "
                f"{ep_steps} steps, reward={ep_reward:.1f}"
            )

        states_arr = np.array(states, dtype=np.float32)
        next_states_arr = np.array(next_states, dtype=np.float32)
        return {
            "states": states_arr,
            "actions": np.array(actions, dtype=np.float32),
            "next_states": next_states_arr,
            "rewards": np.array(rewards_list, dtype=np.float32),
            "dones": np.array(dones_list, dtype=np.float32),
            "diffs": next_states_arr - states_arr,
        }

    def _rbc_action(self, obs: np.ndarray, env: gymnasium.Env) -> np.ndarray:
        """Simple rule-based control action (midpoint of action space)."""
        low = env.action_space.low  # ty: ignore[unresolved-attribute]
        high = env.action_space.high  # ty: ignore[unresolved-attribute]
        return (low + high) / 2.0
