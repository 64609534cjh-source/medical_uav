import numpy as np
import pytest

from medical_uav.env.config import HOSPITALS, M, EnvConfig
from medical_uav.env.orders import demand_multiplier, generate_orders

cfg = EnvConfig()


def test_multiplier_profile():
    assert demand_multiplier(30, cfg) == 0.6
    assert demand_multiplier(60, cfg) == 2.0
    assert demand_multiplier(179.9, cfg) == 2.0
    assert demand_multiplier(200, cfg) == 0.6
    assert demand_multiplier(480, cfg) == 0.0


def test_deterministic_and_sorted():
    a, b = generate_orders(5, 0.07, cfg), generate_orders(5, 0.07, cfg)
    key = [(o.release, o.origin, o.dest, o.weight) for o in a]
    assert key == [(o.release, o.origin, o.dest, o.weight) for o in b]
    assert all(a[i].release <= a[i + 1].release for i in range(len(a) - 1))
    assert [o.id for o in a] == list(range(len(a)))


def test_expected_count_is_582_lambda():
    counts = [len(generate_orders(s, 0.07, cfg)) for s in range(400)]
    assert np.mean(counts) == pytest.approx(582 * 0.07, rel=0.05)


def test_routes_weights_limits():
    for o in generate_orders(7, 0.2, cfg):
        name = ["plasma", "medicine", "sample", "supplies"][o.type_idx]
        if name in ("plasma", "medicine"):
            assert o.origin == M and o.dest in HOSPITALS
        elif name == "sample":
            assert o.origin in HOSPITALS and o.dest == M
        else:
            assert o.origin in HOSPITALS and o.dest in HOSPITALS and o.origin != o.dest
        assert 0.2 <= o.weight <= 0.9 and 0 <= o.release < 480
        assert o.expiry_after == 2 * o.limit
