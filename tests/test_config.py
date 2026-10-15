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
