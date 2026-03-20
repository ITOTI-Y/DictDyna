"""Sinergym environment wrapper for DictDyna."""

import contextlib

import gymnasium
import numpy as np

with contextlib.suppress(ImportError):
    import sinergym  # noqa: F401


class SinergymWrapper(gymnasium.Wrapper):
    """Wrapper for Sinergym environments.

    Provides state normalization, action scaling, and state variable
    index mapping for DictDyna integration.

    Args:
        env: Gymnasium environment (Sinergym).
        normalize_obs: Whether to normalize observations.
        obs_mean: Pre-computed observation mean (for inference).
        obs_std: Pre-computed observation std (for inference).
    """

    def __init__(
        self,
        env: gymnasium.Env,
        normalize_obs: bool = False,
        obs_mean: np.ndarray | None = None,
        obs_std: np.ndarray | None = None,
    ) -> None:
        super().__init__(env)
        self.normalize_obs = normalize_obs
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        self._obs_buffer: list[np.ndarray] = []

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        if self.normalize_obs:
            obs = self._normalize(obs)
        return obs, info

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.normalize_obs:
            self._obs_buffer.append(obs)
            obs = self._normalize(obs)
        return obs, reward, terminated, truncated, info

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_mean is not None and self._obs_std is not None:
            std = np.maximum(self._obs_std, 1e-8)
            return (obs - self._obs_mean) / std
        return obs

    def update_normalization_stats(self) -> None:
        """Update running mean/std from buffered observations."""
        if self._obs_buffer:
            arr = np.array(self._obs_buffer)
            self._obs_mean = arr.mean(axis=0)
            self._obs_std = arr.std(axis=0)

    @property
    def state_dim(self) -> int:
        return self.observation_space.shape[0]  # ty: ignore[not-subscriptable]

    @property
    def action_dim(self) -> int:
        return self.action_space.shape[0]  # ty: ignore[not-subscriptable]
