"""Observation space configuration for Sinergym 5zone environments.

Provides a centralized mapping of observation variable names to indices,
replacing all hardcoded dimension indices throughout the codebase.

Source: env.unwrapped.observation_variables from Sinergym Docker.
Verified consistent across office_hot, office_mixed, office_cool.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ObsConfig:
    """Observation space layout for Sinergym 5zone buildings."""

    VARIABLE_NAMES: tuple[str, ...] = (
        "month",
        "day_of_month",
        "hour",
        "outdoor_temperature",
        "outdoor_humidity",
        "wind_speed",
        "wind_direction",
        "diffuse_solar_radiation",
        "direct_solar_radiation",
        "air_temperature",
        "air_humidity",
        "people_occupant",
        "heating_setpoint",
        "cooling_setpoint",
        "co2_emission",
        "HVAC_electricity_demand_rate",
        "total_electricity_HVAC",
    )

    # Exogenous dims: time + weather + occupant + co2 (cannot be predicted from action)
    EXOGENOUS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 14)

    # Controllable dims: affected by HVAC action
    CONTROLLABLE: tuple[int, ...] = (9, 10, 15, 16)

    # Action setpoint dims: directly determined by action, not predicted
    ACTION_SETPOINTS: tuple[int, ...] = (12, 13)

    # Named indices for reward calculation
    MONTH: int = 0
    AIR_TEMPERATURE: int = 9
    AIR_HUMIDITY: int = 10
    HVAC_POWER: int = 15
    HVAC_ENERGY: int = 16

    @property
    def total_dim(self) -> int:
        return len(self.VARIABLE_NAMES)

    @property
    def controllable_dim(self) -> int:
        return len(self.CONTROLLABLE)

    def name(self, idx: int) -> str:
        return self.VARIABLE_NAMES[idx]


OBS_CONFIG = ObsConfig()
