import math

import pytest

from medical_uav.env import EnvConfig, MedicalUAVEnv
from medical_uav.env.config import day_length


def test_default_config_is_valid():
    EnvConfig().validate()
    assert day_length(EnvConfig()) == 660.0


@pytest.mark.parametrize("field,value", [("wait_max", 0.0), ("charge_rate", 0.0), ("max_boxes", 4),
                                         ("n_slots", 16), ("speed_km_per_min", -1.0),
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
