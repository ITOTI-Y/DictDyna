"""Universal observation config for heterogeneous Sinergym buildings.

Auto-categorizes observation variables by keyword matching on env variable
names, producing a CategoryMapping that enables the UniversalObsEncoder
to handle ANY Sinergym building with a single fixed-size architecture.

No manual per-building mapping needed — new buildings are supported by
just reading their env.observation_variables list.
"""

from dataclasses import dataclass, field
from typing import Final

import numpy as np

# Category names (order matters — determines embedding layout)
CATEGORIES: Final = (
    "TIME",
    "WEATHER",
    "ZONE_TEMP",
    "ZONE_HUMID",
    "OCCUPANCY",
    "SETPOINT",
    "POWER",
    "OTHER",
)

# Max dims per category (covers all Sinergym building types)
CATEGORY_PAD: Final[dict[str, int]] = {
    "TIME": 3,
    "WEATHER": 6,
    "ZONE_TEMP": 20,
    "ZONE_HUMID": 20,
    "OCCUPANCY": 15,
    "SETPOINT": 6,
    "POWER": 10,
    "OTHER": 8,
}

CATEGORY_EMBED_DIM: Final = 16
TOTAL_EMBED_DIM: Final = CATEGORY_EMBED_DIM * len(CATEGORIES)  # 128


def categorize_var(name: str) -> str:
    """Classify a Sinergym observation variable name into a category.

    Uses keyword matching — no manual mapping needed per building.
    """
    nl = name.lower()
    if name in ("month", "day_of_month", "hour"):
        return "TIME"
    if any(k in nl for k in ("outdoor", "wind", "solar")):
        return "WEATHER"
    if "temperature" in nl and "outdoor" not in nl:
        return "ZONE_TEMP"
    if "humidity" in nl and "outdoor" not in nl:
        return "ZONE_HUMID"
    if any(k in nl for k in ("people", "occupant", "cpu_loading")):
        return "OCCUPANCY"
    if "setpoint" in nl:
        return "SETPOINT"
    if any(k in nl for k in ("hvac", "electricity", "energy", "power", "cooling_rate")):
        return "POWER"
    return "OTHER"


@dataclass(frozen=True)
class CategoryMapping:
    """Per-building mapping from raw obs dims to padded category arrays.

    Generated automatically from env observation variable names.
    """

    building_id: str
    obs_dim: int
    category_indices: dict[str, tuple[int, ...]]
    """Maps category name → tuple of raw obs dim indices belonging to it."""

    category_masks: dict[str, np.ndarray] = field(repr=False)
    """Maps category name → bool mask of shape (pad_to,), True for valid dims."""

    @property
    def total_pad_dim(self) -> int:
        return sum(CATEGORY_PAD.values())

    def pad_and_mask(
        self, raw_obs: np.ndarray, category: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract, pad, and mask a category's values from raw obs.

        Args:
            raw_obs: Raw observation, shape (..., obs_dim).
            category: Category name.

        Returns:
            (padded_values, mask) where padded has shape (..., pad_to)
            and mask has shape (pad_to,).
        """
        indices = self.category_indices[category]
        pad_to = CATEGORY_PAD[category]
        mask = self.category_masks[category]

        if len(indices) == 0:
            shape = (*raw_obs.shape[:-1], pad_to)
            return np.zeros(shape, dtype=np.float32), mask

        values = raw_obs[..., list(indices)]
        pad_width = pad_to - len(indices)
        if pad_width > 0:
            pad_shape = [(0, 0)] * (values.ndim - 1) + [(0, pad_width)]
            values = np.pad(values, pad_shape, constant_values=0.0)
        return values.astype(np.float32), mask


def build_category_mapping(
    building_id: str,
    observation_variables: list[str] | tuple[str, ...],
) -> CategoryMapping:
    """Build CategoryMapping from Sinergym env observation variable names.

    Args:
        building_id: Identifier for the building.
        observation_variables: List of variable names from env.observation_variables.

    Returns:
        CategoryMapping with auto-detected category assignments.
    """
    cat_indices: dict[str, list[int]] = {c: [] for c in CATEGORIES}

    for i, var_name in enumerate(observation_variables):
        cat = categorize_var(var_name)
        if cat not in cat_indices:
            cat_indices[cat] = []
        cat_indices[cat].append(i)

    # Validate: no category exceeds its pad limit
    for cat, indices in cat_indices.items():
        pad_to = CATEGORY_PAD[cat]
        if len(indices) > pad_to:
            raise ValueError(
                f"Building '{building_id}': category '{cat}' has {len(indices)} vars "
                f"but pad_to={pad_to}. Increase CATEGORY_PAD['{cat}']."
            )

    # Build masks
    cat_masks: dict[str, np.ndarray] = {}
    for cat in CATEGORIES:
        pad_to = CATEGORY_PAD[cat]
        n_valid = len(cat_indices[cat])
        mask = np.zeros(pad_to, dtype=np.float32)
        mask[:n_valid] = 1.0
        cat_masks[cat] = mask

    return CategoryMapping(
        building_id=building_id,
        obs_dim=len(observation_variables),
        category_indices={c: tuple(cat_indices[c]) for c in CATEGORIES},
        category_masks=cat_masks,
    )


# Pre-defined mappings for known buildings (avoids needing Docker at import time)
KNOWN_5ZONE_VARS: Final = (
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

KNOWN_WAREHOUSE_VARS: Final = (
    "month",
    "day_of_month",
    "hour",
    "outdoor_temperature",
    "outdoor_humidity",
    "wind_speed",
    "wind_direction",
    "diffuse_solar_radiation",
    "direct_solar_radiation",
    "zone1_office_air_temperature",
    "zone2_fine_storage_air_temperature",
    "zone3_bulk_storage_air_temperature",
    "zone1_office_air_humidity",
    "zone2_fine_storage_air_humidity",
    "zone3_bulk_storage_air_humidity",
    "zone1_office_people_occupant",
    "office_heating_setpoint",
    "fine_storage_heating_setpoint",
    "bulk_storage_heating_setpoint",
    "office_cooling_setpoint",
    "fine_storage_cooling_setpoint",
    "HVAC_electricity_demand_rate",
)

KNOWN_SHOP_VARS: Final = (
    "month",
    "day_of_month",
    "hour",
    "outdoor_temperature",
    "outdoor_humidity",
    "wind_speed",
    "wind_direction",
    "diffuse_solar_radiation",
    "direct_solar_radiation",
    "zn_1_flr_1_sec_1_air_temperature",
    "zn_1_flr_1_sec_2_air_temperature",
    "zn_1_flr_1_sec_3_air_temperature",
    "zn_1_flr_1_sec_4_air_temperature",
    "zn_1_flr_1_sec_5_air_temperature",
    "zn_1_flr_1_sec_1_air_humidity",
    "zn_1_flr_1_sec_2_air_humidity",
    "zn_1_flr_1_sec_3_air_humidity",
    "zn_1_flr_1_sec_4_air_humidity",
    "zn_1_flr_1_sec_5_air_humidity",
    "zn_1_flr_1_sec_1_people_count",
    "zn_1_flr_1_sec_2_people_count",
    "zn_1_flr_1_sec_3_people_count",
    "zn_1_flr_1_sec_4_people_count",
    "zn_1_flr_1_sec_5_people_count",
    "heating_setpoint",
    "cooling_setpoint",
    "storage_battery_charge_state",
    "storage_charge_energy",
    "storage_charge_power",
    "storage_discharge_energy",
    "storage_discharge_power",
    "storage_thermal_loss_energy",
    "storage_thermal_loss_rate",
    "HVAC_electricity_demand_rate",
)
