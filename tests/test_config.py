import pytest

from medical_uav.env import EnvConfig, MedicalUAVEnv
from medical_uav.env.config import day_length


def test_default_config_is_valid():
    EnvConfig().validate()
    assert day_length(EnvConfig()) == 660.0


def test_max_boxes_is_limited_by_the_onboard_slots():
    with pytest.raises(ValueError):
        EnvConfig(max_boxes=4).validate()


@pytest.mark.parametrize("loads", [{"low": 0.04, "medium": 0.08}, {"low": 0.0, "high": 0.0}])
def test_loads_need_a_positive_high_level(loads):
    with pytest.raises(ValueError):
        EnvConfig(loads=loads).validate()


def test_env_rejects_unknown_load_and_render_mode():
    with pytest.raises(ValueError):
        MedicalUAVEnv(load="extreme")
    with pytest.raises(ValueError):
        MedicalUAVEnv(render_mode="video")
    with pytest.raises(ValueError):
        MedicalUAVEnv().reset(options={"load": "extreme"})


def test_default_loads():
    assert EnvConfig().loads == {"low": 0.04, "medium": 0.09, "high": 0.12}


def test_named_rules_have_their_defaults():
    cfg = EnvConfig()
    assert cfg.min_charge_gain == 5.0 and cfg.zone_size_range == (2, 4)


@pytest.mark.parametrize("field,value", [
    ("wait_max", 0.0), ("charge_rate", 0.0), ("max_weight", 0.0),
    ("handling_min", 0.0), ("speed_km_per_min", -1.0), ("cell_km", 0.0),
    ("release_window", 0.0), ("energy_per_km", float("inf")),
    ("payload_factor", -0.1), ("energy_cost", -0.1),
    ("expiry_penalty", float("nan")), ("peak_multiplier", -1.0),
    ("offpeak_multiplier", float("inf")), ("reserve", -1.0), ("reserve", 100.0),
    ("min_charge_gain", 100.0), ("max_boxes", 1.5), ("grid_size", 16.5),
    ("max_map_tries", 0), ("map_seed", -1),
])
def test_invalid_numeric_settings_fail_before_simulation(field, value):
    with pytest.raises(ValueError, match=field):
        EnvConfig(**{field: value}).validate()


@pytest.mark.parametrize("field,value", [
    ("n_zones_range", (-1, 3)), ("n_zones_range", (4, 3)),
    ("zone_size_range", (0, 4)), ("zone_size_range", (2, 3.5)),
    ("peak_windows", ((60.0, 500.0),)),
    ("peak_windows", ((60.0, 60.0),)),
    ("peak_windows", ((60.0, 180.0), (150.0, 200.0))),
])
def test_invalid_ranges_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        EnvConfig(**{field: value}).validate()


@pytest.mark.parametrize("loads", [
    {"low": -0.01, "high": 0.12},
    {"low": 0.04, "high": float("nan")},
    {"medium": float("inf"), "high": 0.12},
])
def test_non_finite_or_negative_loads_are_rejected(loads):
    with pytest.raises(ValueError, match="loads"):
        EnvConfig(loads=loads).validate()


def test_zero_demand_and_costs_are_allowed():
    EnvConfig(loads={"low": 0.0, "high": 0.12}, peak_windows=(),
              payload_factor=0.0, energy_cost=0.0, expiry_penalty=0.0).validate()
