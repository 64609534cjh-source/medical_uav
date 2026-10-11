"""Tests for configuration constants (spec §4.2, §4.3) and physics helpers (§4.2, §7)."""
import pytest

from medical_uav.env import physics
from medical_uav.env.config import ORDER_TYPES, EnvConfig

cfg = EnvConfig()


def test_order_types_match_spec():
    names = [o.name for o in ORDER_TYPES]
    assert names == ["plasma", "medicine", "sample", "supplies"]
    assert [o.limit for o in ORDER_TYPES] == [30, 90, 60, 75]
    assert [o.expiry for o in ORDER_TYPES] == [60, 180, 120, 150]
    assert sum(o.share for o in ORDER_TYPES) == pytest.approx(1.0)


def test_flight_energy_and_time():
    assert physics.flight_energy(1.0, 2.0, cfg) == pytest.approx(5.25)
    assert physics.flight_energy(2.0, 0.0, cfg) == pytest.approx(7.0)
    assert physics.flight_time(2.5, cfg) == pytest.approx(2.5)


def test_charge_minutes():
    assert physics.charge_minutes(40, 80, cfg) == pytest.approx(8.0)
    assert physics.charge_minutes(90, 80, cfg) == 0.0


@pytest.mark.parametrize("age,expected", [(10, 1.0), (30, 1.0), (45, 0.25), (60, -0.5)])
def test_delivery_reward_plasma(age, expected):
    assert physics.delivery_reward(age, 30, 60, cfg) == pytest.approx(expected)
