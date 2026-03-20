"""CityLearn environment wrapper for DictDyna."""

import numpy as np


class CityLearnWrapper:
    """Wrapper for CityLearn environments.

    CityLearn controls battery storage systems (not HVAC thermal dynamics).
    This wrapper provides a unified interface compatible with DictDyna.

    Note: CityLearn must be installed separately due to dependency conflicts:
        uv pip install citylearn

    Args:
        n_buildings: Number of buildings in the district.
        schema_path: Path to CityLearn schema file (optional).
    """

    def __init__(
        self,
        n_buildings: int = 5,
        schema_path: str | None = None,
    ) -> None:
        self.n_buildings = n_buildings
        self.schema_path = schema_path
        self._env = None

    def _lazy_init(self) -> None:
        """Lazily initialize CityLearn environment."""
        if self._env is not None:
            return
        try:
            from citylearn.citylearn import CityLearnEnv

            if self.schema_path:
                self._env = CityLearnEnv(schema=self.schema_path)
            else:
                from citylearn.data import DataSet

                schema = DataSet.get_schema("citylearn_challenge_2022_phase_all")
                schema["count"] = self.n_buildings
                self._env = CityLearnEnv(schema=schema)
        except ImportError as e:
            raise ImportError(
                "CityLearn not installed. Install with: uv pip install citylearn"
            ) from e

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self._lazy_init()
        obs = self._env.reset()  # ty: ignore[unresolved-attribute]
        return np.array(obs, dtype=np.float32), {}

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, done, info = self._env.step(action.tolist())  # ty: ignore[unresolved-attribute]
        return (
            np.array(obs, dtype=np.float32),
            float(np.sum(reward)),
            done,
            False,
            info,
        )

    def close(self) -> None:
        if self._env is not None:
            self._env = None

    @property
    def state_dim(self) -> int:
        self._lazy_init()
        return len(self._env.observation_space[0].shape)  # ty: ignore[unresolved-attribute]

    @property
    def action_dim(self) -> int:
        self._lazy_init()
        return len(self._env.action_space)  # ty: ignore[unresolved-attribute]
