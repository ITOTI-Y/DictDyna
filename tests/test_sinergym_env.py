"""Tests for environment wrappers (Sinergym requires EnergyPlus)."""

import pytest

# Skip all tests if sinergym is not installed
sinergym = pytest.importorskip("sinergym")


@pytest.mark.slow
class TestSinergymWrapper:
    def test_wrapper_creation(self):
        import gymnasium

        from src.env.sinergym_wrapper import SinergymWrapper

        env = gymnasium.make("Eplus-5zone-hot-continuous-v1")
        wrapped = SinergymWrapper(env)
        assert wrapped.state_dim > 0
        assert wrapped.action_dim > 0
        wrapped.close()

    def test_multi_building_creation(self):
        from src.env.multi_building_sinergym import MultiBuildingSinergym

        configs = [
            {"env_name": "Eplus-5zone-hot-continuous-v1", "building_id": "test_hot"},
        ]
        multi = MultiBuildingSinergym(configs)
        assert multi.n_buildings == 1
        multi.close_all()
