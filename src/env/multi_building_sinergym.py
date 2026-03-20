"""Multi-building Sinergym environment management."""

import contextlib

import gymnasium
import numpy as np

with contextlib.suppress(ImportError):
    import sinergym  # noqa: F401


class MultiBuildingSinergym:
    """Manage multiple Sinergym environments as a unified multi-building interface.

    Each building = one Sinergym instance with distinct env_name (IDF + weather).

    Args:
        building_configs: List of dicts with keys:
            env_name: Sinergym registered environment name.
            building_id: Unique string identifier.
    """

    def __init__(self, building_configs: list[dict]) -> None:
        self.envs: dict[str, gymnasium.Env] = {}
        self.building_ids: list[str] = []
        for cfg in building_configs:
            env = gymnasium.make(cfg["env_name"])
            bid = cfg["building_id"]
            self.envs[bid] = env
            self.building_ids.append(bid)

    @property
    def n_buildings(self) -> int:
        return len(self.building_ids)

    @property
    def state_dim(self) -> int:
        env = next(iter(self.envs.values()))
        return env.observation_space.shape[0]  # ty: ignore[not-subscriptable]

    @property
    def action_dim(self) -> int:
        env = next(iter(self.envs.values()))
        return env.action_space.shape[0]  # ty: ignore[not-subscriptable]

    def reset_all(self, seed: int | None = None) -> dict[str, tuple]:
        """Reset all building environments.

        Returns:
            Dict mapping building_id to (obs, info).
        """
        results = {}
        for i, bid in enumerate(self.building_ids):
            s = seed + i if seed is not None else None
            obs, info = self.envs[bid].reset(seed=s)
            results[bid] = (obs, info)
        return results

    def step(self, building_id: str, action: np.ndarray) -> tuple:
        """Step a specific building environment.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        return self.envs[building_id].step(action)

    def close_all(self) -> None:
        for env in self.envs.values():
            env.close()

    def collect_offline_data(
        self,
        policy: str = "random",
        n_episodes: int = 2,
    ) -> dict[str, list[dict]]:
        """Collect offline rollout data for dictionary pretraining.

        Args:
            policy: "random" for random actions, "rbc" for rule-based.
            n_episodes: Number of episodes per building.

        Returns:
            Dict mapping building_id to list of transition dicts.
        """
        all_data: dict[str, list[dict]] = {}
        for bid in self.building_ids:
            env = self.envs[bid]
            transitions: list[dict] = []
            for ep in range(n_episodes):
                obs, _ = env.reset(seed=ep)
                done = False
                while not done:
                    if policy == "random":
                        action = env.action_space.sample()
                    else:
                        low = env.action_space.low  # ty: ignore[unresolved-attribute]
                        high = env.action_space.high  # ty: ignore[unresolved-attribute]
                        action = (low + high) / 2.0
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    transitions.append(
                        {"s": obs, "a": action, "s_next": next_obs, "r": reward}
                    )
                    obs = next_obs
                    done = terminated or truncated
            all_data[bid] = transitions
        return all_data
