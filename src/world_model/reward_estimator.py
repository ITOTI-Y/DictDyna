"""Reward estimation from predicted states for Sinergym environments."""

import torch


class SinergymRewardEstimator:
    """Estimate reward from predicted state in Sinergym environments.

    Sinergym rewards are typically:
        r = -w_E * E_t - (1 - w_E) * comfort_penalty(T_indoor, T_target)

    Both E_t (HVAC power) and T_indoor are state variables, so reward
    can be computed directly from world model predictions.

    When the world model operates in normalized obs space, predicted states
    must be denormalized before computing reward (temperature thresholds
    and power values are in raw units).

    Args:
        comfort_weight: Weight for comfort penalty (1 - energy_weight).
        temp_target: Target indoor temperature (C).
        temp_band: Acceptable temperature range around target.
        state_indices: Mapping of variable name to state vector index.
        obs_mean: Observation mean for denormalization (None = raw space).
        obs_std: Observation std for denormalization (None = raw space).
    """

    def __init__(
        self,
        comfort_weight: float = 0.5,
        temp_target: float = 23.0,
        temp_band: float = 2.0,
        state_indices: dict[str, int] | None = None,
        obs_mean: torch.Tensor | None = None,
        obs_std: torch.Tensor | None = None,
    ) -> None:
        self.comfort_weight = comfort_weight
        self.temp_target = temp_target
        self.temp_band = temp_band
        self.state_indices = state_indices or {}
        self.obs_mean = obs_mean
        self.obs_std = obs_std

    def estimate(self, predicted_state: torch.Tensor) -> torch.Tensor:
        """Compute reward from predicted next state.

        Args:
            predicted_state: World model output, shape (batch, d).
                May be in normalized space if obs_mean/obs_std are set.

        Returns:
            Estimated reward, shape (batch,).
        """
        # Denormalize if needed (reward thresholds are in raw units)
        if self.obs_mean is not None and self.obs_std is not None:
            state_raw = predicted_state * self.obs_std.to(
                predicted_state.device
            ) + self.obs_mean.to(predicted_state.device)
        else:
            state_raw = predicted_state

        idx_temp = self.state_indices.get("indoor_temp", 0)
        idx_power = self.state_indices.get("hvac_power", 1)

        temp = state_raw[:, idx_temp]
        power = state_raw[:, idx_power]

        # Comfort penalty: 0 inside band, linear outside
        comfort_violation = torch.relu(
            temp - (self.temp_target + self.temp_band)
        ) + torch.relu((self.temp_target - self.temp_band) - temp)

        energy_weight = 1.0 - self.comfort_weight
        reward = -(energy_weight * power + self.comfort_weight * comfort_violation)
        return reward
