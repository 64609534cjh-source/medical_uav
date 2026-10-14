import math

import pytest

from medical_uav.env import EnvConfig, MedicalUAVEnv
from medical_uav.env.config import day_length


def test_default_config_is_valid():
    EnvConfig().validate()
    assert day_length(EnvConfig()) == 660.0


@pytest.mark.parametrize("field,value", [("wait_max", 0.0), ("charge_rate", 0.0), ("max_boxes", 4),
                                         ("speed_km_per_min", -1.0),
                                         ("reserve", 100.0)])
def test_invalid_config_rejected(field, value):
    cfg = EnvConfig(**{field: value})
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_load_value_rejected():
    with pytest.raises(ValueError):
        EnvConfig(loads={"low": 0.04, "medium": math.inf, "high": 0.12}).validate()


def test_env_rejects_unknown_load_and_render_mode():
    with pytest.raises(ValueError):
        MedicalUAVEnv(load="extreme")
    with pytest.raises(ValueError):
        MedicalUAVEnv(render_mode="video")
    env = MedicalUAVEnv()
    with pytest.raises(ValueError):
        env.reset(options={"lam": math.nan})
    with pytest.raises(ValueError):
        env.reset(options={"load": "extreme"})


def test_default_loads():
    assert EnvConfig().loads == {"low": 0.04, "medium": 0.08, "high": 0.12}


@pytest.mark.parametrize("loads", [{"low": 0.04, "medium": 0.08}, {"low": 0.0, "high": 0.0}])
def test_loads_need_a_positive_high_level(loads):
    with pytest.raises(ValueError):
        EnvConfig(loads=loads).validate()


@pytest.mark.parametrize("field,value", [("payload_factor", math.nan), ("expiry_penalty", math.nan),
                                         ("cell_km", math.inf), ("grid_size", 16.5),
                                         ("max_boxes", 2.5), ("pool_cap", 30.0)])
def test_non_finite_or_fractional_values_rejected(field, value):
    with pytest.raises(ValueError):
        EnvConfig(**{field: value}).validate()


@pytest.mark.parametrize("field,value", [
    ("peak_multiplier", -1.0), ("offpeak_multiplier", math.nan), ("min_node_spacing_km", math.nan),
    ("c2_min_dist_from_m_km", -1.0), ("min_charge_gain", -5.0), ("map_seed", -1),
    ("n_zones_range", (3, 2)), ("n_zones_range", (-1, 2)), ("n_zones_range", (1.5, 2)),
    ("zone_size_range", (0, 3)), ("zone_size_range", (5, 2)), ("zone_size_range", (2, 40)),
    ("peak_windows", ((180.0, 60.0),)), ("peak_windows", ((60.0, math.nan),)),
    ("peak_windows", ((-10.0, 30.0),)),
])
def test_more_invalid_settings_rejected(field, value):
    with pytest.raises(ValueError):
        EnvConfig(**{field: value}).validate()


def test_zero_zones_and_short_release_window_are_allowed():
    EnvConfig(n_zones_range=(0, 0)).validate()
    EnvConfig(release_window=120.0).validate()  # peak windows past the end are simply unused


def test_named_rules_have_their_defaults():
    cfg = EnvConfig()
    assert cfg.min_charge_gain == 5.0 and cfg.zone_size_range == (2, 3)


def test_zone_size_is_irrelevant_without_zones():
    EnvConfig(grid_size=4, n_zones_range=(0, 0)).validate()


@pytest.mark.parametrize("field,value", [("peak_windows", (None,)), ("n_zones_range", None),
                                         ("zone_size_range", (2,)), ("peak_windows", ((1.0,),))])
def test_malformed_structures_raise_value_error(field, value):
    with pytest.raises(ValueError):
        EnvConfig(**{field: value}).validate()
