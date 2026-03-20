"""Unified multi-building interface."""

from typing import Any

import numpy as np


class MultiBuildingInterface:
    """Unified interface for multi-building environments.

    Abstracts over different backends (Sinergym, CityLearn, mock).

    Args:
        backend: The underlying multi-building environment.
        building_ids: List of building identifiers.
    """

    def __init__(
        self,
        backend: Any,
        building_ids: list[str],
    ) -> None:
        self.backend = backend
        self.building_ids = building_ids

    @property
    def n_buildings(self) -> int:
        return len(self.building_ids)

    def reset_all(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset all environments, return initial observations."""
        if hasattr(self.backend, "reset_all"):
            results = self.backend.reset_all(seed=seed)
            return {bid: obs for bid, (obs, _) in results.items()}
        raise NotImplementedError

    def step(
        self, building_id: str, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step a specific building."""
        if hasattr(self.backend, "step"):
            return self.backend.step(building_id, action)
        raise NotImplementedError

    def close(self) -> None:
        if hasattr(self.backend, "close_all"):
            self.backend.close_all()
        elif hasattr(self.backend, "close"):
            self.backend.close()
