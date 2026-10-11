import pytest

from medical_uav.env.config import C2, M, EnvConfig
from tests.conftest import make_order, make_sim


def test_start_state(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 1, 0.7, 0.0)])
    assert sim.t == 0 and sim.location == M and sim.battery == 100
    assert [o.id for o in sim.pending] == [0]


def test_wait_stops_at_next_release(line_map):
    sim = make_sim(line_map, [make_order(0, 1, M, 1, 0.5, 4.0)])
    sim.do_wait()
    assert sim.t == pytest.approx(4.0) and len(sim.pending) == 1


def test_wait_max_ten_minutes(line_map):
    sim = make_sim(line_map, [make_order(0, 1, M, 1, 0.5, 100.0)])
    sim.do_wait()
    assert sim.t == pytest.approx(10.0)


def test_pending_order_expires_with_penalty(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 1, 0.7, 0.0)])  # plasma, H = 60
    sim.advance_to(60.0)
    o = sim.orders[0]
    assert o.status == "expired" and o.end_time == pytest.approx(60.0)
    assert sim.pending == [] and sim.total_reward == pytest.approx(-0.5)


def test_expiry_before_release_same_time(line_map):
    cfg = EnvConfig(pool_cap=1)
    orders = [make_order(0, 0, M, 1, 0.7, 0.0), make_order(1, 0, M, 2, 0.7, 60.0)]
    sim = make_sim(line_map, orders, cfg)
    sim.advance_to(60.0)
    assert orders[0].status == "expired" and orders[1].status == "pending"


def test_pool_cap_rejects(line_map):
    cfg = EnvConfig(pool_cap=2)
    orders = [make_order(i, 1, M, 1, 0.5, 5.0) for i in range(3)]
    sim = make_sim(line_map, orders, cfg)
    sim.advance_to(5.0)
    assert [o.status for o in orders] == ["pending", "pending", "rejected"]
    assert sim.total_reward == pytest.approx(-0.5)


def test_done_only_after_release_window(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 1, 0.7, 0.0)])
    sim.advance_to(61.0)
    assert not sim.is_done()
    sim.advance_to(480.0)
    assert sim.is_done()


def test_nearest_charger(line_map):
    sim = make_sim(line_map, [])
    assert sim.nearest_charger(5, 0.0) == (C2, pytest.approx(2.5 * 3.5))  # H5 -> C2 is 2.5 km
    assert sim.nearest_charger(1, 0.0) == (C2, pytest.approx(1.5 * 3.5))  # 1.5 km < 2 km to M
    assert sim.nearest_charger(M, 0.0) == (M, 0.0)


def test_events_log_release_and_expiry(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 1, 0.7, 5.0)])
    sim.advance_to(65.0)
    assert [(k, i) for (_, k, i) in sim.events] == [("release", 0), ("expire", 0)]


def test_unsorted_orders_rejected(line_map):
    orders = [make_order(0, 1, M, 1, 0.5, 10.0), make_order(1, 1, M, 2, 0.5, 5.0)]
    with pytest.raises(ValueError):
        make_sim(line_map, orders)
