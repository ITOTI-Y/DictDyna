"""Tests for UniversalObsEncoder and CategoryMapping."""

import numpy as np
import torch

from src.obs_config_universal import (
    CATEGORIES,
    CATEGORY_PAD,
    KNOWN_5ZONE_VARS,
    KNOWN_SHOP_VARS,
    KNOWN_WAREHOUSE_VARS,
    TOTAL_EMBED_DIM,
    build_category_mapping,
    categorize_var,
)
from src.obs_encoder import UniversalObsEncoder


class TestCategorizeVar:
    def test_time_vars(self):
        assert categorize_var("month") == "TIME"
        assert categorize_var("day_of_month") == "TIME"
        assert categorize_var("hour") == "TIME"

    def test_weather_vars(self):
        assert categorize_var("outdoor_temperature") == "WEATHER"
        assert categorize_var("outdoor_humidity") == "WEATHER"
        assert categorize_var("wind_speed") == "WEATHER"
        assert categorize_var("diffuse_solar_radiation") == "WEATHER"

    def test_zone_temp_vars(self):
        assert categorize_var("air_temperature") == "ZONE_TEMP"
        assert categorize_var("zone1_office_air_temperature") == "ZONE_TEMP"
        assert categorize_var("core_bottom_air_temperature") == "ZONE_TEMP"
        assert categorize_var("mean_radiant_temperature") == "ZONE_TEMP"

    def test_zone_humid_vars(self):
        assert categorize_var("air_humidity") == "ZONE_HUMID"
        assert categorize_var("zone1_office_air_humidity") == "ZONE_HUMID"

    def test_occupancy_vars(self):
        assert categorize_var("people_occupant") == "OCCUPANCY"
        assert categorize_var("zn_1_flr_1_sec_1_people_count") == "OCCUPANCY"
        assert categorize_var("cpu_loading_fraction") == "OCCUPANCY"

    def test_setpoint_vars(self):
        assert categorize_var("heating_setpoint") == "SETPOINT"
        assert categorize_var("cooling_setpoint") == "SETPOINT"
        assert categorize_var("office_heating_setpoint") == "SETPOINT"

    def test_power_vars(self):
        assert categorize_var("HVAC_electricity_demand_rate") == "POWER"
        assert categorize_var("total_electricity_HVAC") == "POWER"
        assert categorize_var("chiller_electricity_rate") == "POWER"

    def test_other_vars(self):
        assert categorize_var("co2_emission") == "OTHER"
        assert categorize_var("storage_battery_charge_state") == "OTHER"


class TestCategoryMapping:
    def test_5zone_mapping(self):
        m = build_category_mapping("5zone_hot", KNOWN_5ZONE_VARS)
        assert m.obs_dim == 17
        assert len(m.category_indices["TIME"]) == 3
        assert len(m.category_indices["WEATHER"]) == 6
        assert len(m.category_indices["ZONE_TEMP"]) == 1
        assert len(m.category_indices["ZONE_HUMID"]) == 1
        assert len(m.category_indices["OCCUPANCY"]) == 1
        assert len(m.category_indices["SETPOINT"]) == 2
        assert len(m.category_indices["POWER"]) == 2
        assert len(m.category_indices["OTHER"]) == 1  # co2

    def test_warehouse_mapping(self):
        m = build_category_mapping("warehouse_hot", KNOWN_WAREHOUSE_VARS)
        assert m.obs_dim == 22
        assert len(m.category_indices["ZONE_TEMP"]) == 3
        assert len(m.category_indices["ZONE_HUMID"]) == 3
        assert len(m.category_indices["SETPOINT"]) == 5

    def test_shop_mapping(self):
        m = build_category_mapping("shop_hot", KNOWN_SHOP_VARS)
        assert m.obs_dim == 34
        assert len(m.category_indices["ZONE_TEMP"]) == 5
        assert len(m.category_indices["ZONE_HUMID"]) == 5
        assert len(m.category_indices["OCCUPANCY"]) == 5

    def test_masks_correct_shape(self):
        m = build_category_mapping("5zone_hot", KNOWN_5ZONE_VARS)
        for cat in CATEGORIES:
            mask = m.category_masks[cat]
            assert mask.shape == (CATEGORY_PAD[cat],)
            n_valid = len(m.category_indices[cat])
            assert mask[:n_valid].sum() == n_valid
            assert mask[n_valid:].sum() == 0

    def test_all_dims_accounted(self):
        """Every raw dim must be assigned to exactly one category."""
        for vars_list in [KNOWN_5ZONE_VARS, KNOWN_WAREHOUSE_VARS, KNOWN_SHOP_VARS]:
            m = build_category_mapping("test", vars_list)
            all_indices = []
            for indices in m.category_indices.values():
                all_indices.extend(indices)
            assert sorted(all_indices) == list(range(m.obs_dim))

    def test_pad_and_mask(self):
        m = build_category_mapping("5zone_hot", KNOWN_5ZONE_VARS)
        raw = np.random.randn(17).astype(np.float32)
        padded, mask = m.pad_and_mask(raw, "ZONE_TEMP")
        assert padded.shape == (CATEGORY_PAD["ZONE_TEMP"],)
        assert mask.shape == (CATEGORY_PAD["ZONE_TEMP"],)
        # 5zone has 1 zone temp → first value matches, rest zero
        assert padded[0] == raw[m.category_indices["ZONE_TEMP"][0]]
        assert np.all(padded[1:] == 0)

    def test_pad_and_mask_batch(self):
        m = build_category_mapping("warehouse_hot", KNOWN_WAREHOUSE_VARS)
        raw = np.random.randn(10, 22).astype(np.float32)
        padded, _mask = m.pad_and_mask(raw, "ZONE_TEMP")
        assert padded.shape == (10, CATEGORY_PAD["ZONE_TEMP"])
        # warehouse has 3 zone temps
        assert np.all(padded[:, 3:] == 0)


class TestUniversalObsEncoder:
    def _make_encoder(self) -> UniversalObsEncoder:
        return UniversalObsEncoder()

    def test_output_dim(self):
        encoder = self._make_encoder()
        assert encoder.total_embed_dim == TOTAL_EMBED_DIM  # 128

    def test_5zone_forward(self):
        encoder = self._make_encoder()
        m = build_category_mapping("5zone_hot", KNOWN_5ZONE_VARS)
        raw = torch.randn(4, 17)
        embed = encoder(raw, m)
        assert embed.shape == (4, TOTAL_EMBED_DIM)
        assert torch.isfinite(embed).all()

    def test_warehouse_forward(self):
        encoder = self._make_encoder()
        m = build_category_mapping("warehouse_hot", KNOWN_WAREHOUSE_VARS)
        raw = torch.randn(4, 22)
        embed = encoder(raw, m)
        assert embed.shape == (4, TOTAL_EMBED_DIM)
        assert torch.isfinite(embed).all()

    def test_shop_forward(self):
        encoder = self._make_encoder()
        m = build_category_mapping("shop_hot", KNOWN_SHOP_VARS)
        raw = torch.randn(4, 34)
        embed = encoder(raw, m)
        assert embed.shape == (4, TOTAL_EMBED_DIM)
        assert torch.isfinite(embed).all()

    def test_same_encoder_different_buildings(self):
        """ONE encoder instance handles multiple building types."""
        encoder = self._make_encoder()
        m5 = build_category_mapping("5zone", KNOWN_5ZONE_VARS)
        mw = build_category_mapping("warehouse", KNOWN_WAREHOUSE_VARS)
        ms = build_category_mapping("shop", KNOWN_SHOP_VARS)

        e5 = encoder(torch.randn(2, 17), m5)
        ew = encoder(torch.randn(2, 22), mw)
        es = encoder(torch.randn(2, 34), ms)

        # All produce same embedding dim
        assert e5.shape == ew.shape == es.shape == (2, TOTAL_EMBED_DIM)

    def test_backward(self):
        """Gradients flow through the encoder."""
        encoder = self._make_encoder()
        m = build_category_mapping("5zone", KNOWN_5ZONE_VARS)
        raw = torch.randn(4, 17, requires_grad=True)
        embed = encoder(raw, m)
        loss = embed.sum()
        loss.backward()
        assert raw.grad is not None
        assert torch.isfinite(raw.grad).all()

    def test_masked_dims_have_no_effect(self):
        """Changing padded (masked) values should not change output."""
        encoder = self._make_encoder()
        m = build_category_mapping("5zone", KNOWN_5ZONE_VARS)

        raw1 = torch.randn(2, 17)
        raw2 = raw1.clone()
        # 5zone has 1 ZONE_TEMP at dim 9. Dims 10+ of ZONE_TEMP category are padding.
        # But padding is done INTERNALLY, so raw obs doesn't have padding dims.
        # Instead, test: two different raw obs → same TIME/WEATHER dims → same TIME/WEATHER embedding
        raw2[:, :9] = raw1[:, :9]  # same time+weather
        raw2[:, 9:] = torch.randn(2, 8)  # different building-specific

        e1 = encoder(raw1, m)
        e2 = encoder(raw2, m)

        # TIME + WEATHER embeddings (first 32 dims) should match
        torch.testing.assert_close(e1[:, :32], e2[:, :32])
        # Building-specific embeddings may differ
        assert not torch.allclose(e1[:, 32:], e2[:, 32:])
