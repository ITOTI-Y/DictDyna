"""Reward estimation from predicted states for Sinergym environments."""

import torch


class SinergymRewardEstimator:
    """Estimate Sinergym LinearReward from predicted state.

    Sinergym LinearReward formula:
        R = -W * lambda_E * power - (1-W) * lambda_T * comfort_violation

    Comfort range depends on season (determined by month variable):
        Summer (Jun-Sep): [23, 26], Winter: [20, 23.5]

    Args:
        energy_weight: W in the formula (default 0.5).
        lambda_energy: Scaling factor for energy (default 0.0001).
        lambda_temp: Scaling factor for temperature violation (default 1.0).
        range_comfort_summer: [T_low, T_high] for summer.
        range_comfort_winter: [T_low, T_high] for winter.
        summer_months: (start_month, end_month) inclusive.
        state_indices: Mapping of variable name to state vector index.
        obs_mean: Observation mean for denormalization (None = raw space).
        obs_std: Observation std for denormalization (None = raw space).
    """

    def __init__(
        self,
        energy_weight: float = 0.5,
        lambda_energy: float = 0.0001,
        lambda_temp: float = 1.0,
        range_comfort_summer: tuple[float, float] = (23.0, 26.0),
        range_comfort_winter: tuple[float, float] = (20.0, 23.5),
        summer_months: tuple[int, int] = (6, 9),
        state_indices: dict[str, int] | None = None,
        obs_mean: torch.Tensor | None = None,
        obs_std: torch.Tensor | None = None,
    ) -> None:
        self.W = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temp
        self.range_summer = range_comfort_summer
        self.range_winter = range_comfort_winter
        self.summer_months = summer_months
        self.state_indices = state_indices or {
            "month": 0,
            "indoor_temp": 9,
            "hvac_power": 15,
        }
        self.obs_mean = obs_mean
        self.obs_std = obs_std

    def estimate(self, predicted_state: torch.Tensor) -> torch.Tensor:
        """Compute reward from predicted next state.

        Args:
            predicted_state: World model output, shape (batch, d).

        Returns:
            Estimated reward, shape (batch,).
        """
        # Denormalize if needed, with OOD clipping
        if self.obs_mean is not None and self.obs_std is not None:
            # Clip normalized state to training range before denormalization
            clipped = torch.clamp(predicted_state, -10.0, 10.0)
            state_raw = clipped * self.obs_std.to(
                predicted_state.device
            ) + self.obs_mean.to(predicted_state.device)
        else:
            state_raw = predicted_state

        month = torch.clamp(state_raw[:, self.state_indices["month"]], 1.0, 12.0)
        temp = torch.clamp(state_raw[:, self.state_indices["indoor_temp"]], 5.0, 45.0)
        power = torch.clamp(
            state_raw[:, self.state_indices["hvac_power"]], 0.0, 20000.0
        )

        # Seasonal comfort range
        is_summer = (month >= self.summer_months[0]) & (month <= self.summer_months[1])
        t_low = torch.where(
            is_summer,
            torch.tensor(self.range_summer[0], device=temp.device),
            torch.tensor(self.range_winter[0], device=temp.device),
        )
        t_high = torch.where(
            is_summer,
            torch.tensor(self.range_summer[1], device=temp.device),
            torch.tensor(self.range_winter[1], device=temp.device),
        )

        comfort_violation = torch.relu(temp - t_high) + torch.relu(t_low - temp)

        reward = (
            -self.W * self.lambda_energy * power
            - (1 - self.W) * self.lambda_temp * comfort_violation
        )
        return reward
