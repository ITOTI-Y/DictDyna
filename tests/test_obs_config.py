"""Tests for observation space configuration."""

from src.obs_config import OBS_CONFIG


class TestObsConfig:
    def test_total_dim(self):
        assert OBS_CONFIG.total_dim == 17

    def test_controllable_dim(self):
        assert OBS_CONFIG.controllable_dim == 4

    def test_all_indices_covered(self):
        """Every index 0-16 should be in exactly one group."""
        all_idx = set(
            OBS_CONFIG.EXOGENOUS + OBS_CONFIG.CONTROLLABLE + OBS_CONFIG.ACTION_SETPOINTS
        )
        assert all_idx == set(range(17))

    def test_named_indices_match(self):
        assert OBS_CONFIG.name(OBS_CONFIG.MONTH) == "month"
        assert OBS_CONFIG.name(OBS_CONFIG.AIR_TEMPERATURE) == "air_temperature"
        assert OBS_CONFIG.name(OBS_CONFIG.HVAC_POWER) == "HVAC_electricity_demand_rate"

    def test_reward_dims_are_controllable(self):
        """Reward-critical dims should be in controllable set."""
        assert OBS_CONFIG.AIR_TEMPERATURE in OBS_CONFIG.CONTROLLABLE
        assert OBS_CONFIG.HVAC_POWER in OBS_CONFIG.CONTROLLABLE
