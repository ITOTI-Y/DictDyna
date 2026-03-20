"""Running observation normalizer for RL agents."""

import numpy as np


class RunningNormalizer:
    """Online running mean/std normalizer using Welford's algorithm.

    Normalizes observations to zero mean and unit variance based on
    running statistics collected during training.

    Args:
        shape: Observation shape (e.g., (17,)).
        clip: Clip normalized values to [-clip, clip].
        epsilon: Small value to avoid division by zero.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        clip: float = 10.0,
        epsilon: float = 1e-8,
    ) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0
        self.clip = clip
        self.epsilon = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a new observation."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize an observation."""
        std = np.sqrt(self.var + self.epsilon)
        normalized = (x - self.mean) / std
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def update_and_normalize(self, x: np.ndarray) -> np.ndarray:
        """Update stats and normalize in one step."""
        self.update(x)
        return self.normalize(x)
