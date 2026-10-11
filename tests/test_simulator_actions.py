import pytest

from medical_uav.env.config import C2, M
from tests.conftest import make_order, make_sim


def test_pick_and_on_time_delivery(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 1, 0.8, 0.0)])
    sim.do_pick(sim.orders[0])
    assert sim.t == pytest.approx(2.0) and sim.orders[0].status == "onboard"
    r0 = sim.total_reward
    sim.do_deliver(1)  # 2 km with 0.8 kg
    energy = 2 * 3.5 * (1 + 0.25 * 0.8)  # 8.4
    assert sim.t == pytest.approx(6.0)
    assert sim.battery == pytest.approx(100 - energy)
    assert sim.total_reward - r0 == pytest.approx(1.0 - 0.001 * energy)
    assert sim.orders[0].status == "delivered" and sim.onboard == []


def test_late_delivery_reward(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 1, 0.8, 0.0)])
    sim.do_pick(sim.orders[0])
    sim.advance_to(41.0)  # arrive 43, unload done 45 -> age 45
    r0 = sim.total_reward
    sim.do_deliver(1)
    assert sim.total_reward - r0 == pytest.approx(0.25 - 0.001 * 8.4)


def test_unload_at_exact_expiry_is_late_delivery(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 4, 0.6, 0.0)])  # plasma to H4 (5 km)
    sim.do_pick(sim.orders[0])
    sim.advance_to(53.0)  # arrive 58, unload done 60 = expiry
    sim.do_deliver(4)
    assert sim.orders[0].status == "delivered"
    assert sim.orders[0].end_time == pytest.approx(60.0)


def test_two_boxes_same_destination_unload_sequentially(line_map):
    orders = [make_order(0, 1, M, 2, 0.5, 0.0), make_order(1, 0, M, 2, 0.7, 0.0)]
    sim = make_sim(line_map, orders)
    sim.do_pick(orders[0])
    sim.do_pick(orders[1])  # t = 4, both at M
    sim.do_deliver(2)  # 3 km, arrive 7
    assert orders[1].end_time == pytest.approx(9.0)  # plasma first (earlier deadline)
    assert orders[0].end_time == pytest.approx(11.0)


def test_expired_cargo_not_disposed_at_c2(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 5, 0.6, 0.0)])
    sim.do_pick(sim.orders[0])
    sim.advance_to(60.0)  # box expires on board
    assert sim.orders[0].status == "expired" and len(sim.onboard) == 1
    sim.do_charge(C2, 100.0)
    assert len(sim.onboard) == 1  # C2 is not a facility
    sim.do_return()
    assert sim.onboard == [] and sim.orders[0].disposed_time is not None


def test_charge_to_target(line_map):
    sim = make_sim(line_map, [])
    sim.battery = 50.0
    sim.do_charge(M, 80.0)
    assert sim.battery == pytest.approx(80.0) and sim.t == pytest.approx(6.0)
    assert sim.charging_time == pytest.approx(6.0)


def test_segments_cover_timeline(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 1, 0.8, 0.0)])
    sim.do_pick(sim.orders[0])
    sim.do_deliver(1)
    assert [s.kind for s in sim.segments] == ["load", "fly", "unload"]
    assert all(a.t1 == pytest.approx(b.t0) for a, b in zip(sim.segments, sim.segments[1:]))


def test_unload_expiry_release_same_instant_with_full_pool(line_map):
    from medical_uav.env.config import EnvConfig

    cfg = EnvConfig(pool_cap=1)
    a = make_order(0, 1, M, 1, 0.5, 0.0)  # medicine to H1
    c = make_order(1, 2, 5, M, 0.3, 3.0)  # sample at H5, expires at 123, fills the pool
    d = make_order(2, 1, M, 2, 0.5, 123.0)  # released exactly when c expires
    sim = make_sim(line_map, [a, c, d], cfg)
    sim.do_pick(a)  # t = 2
    sim.advance_to(119.0)
    sim.do_deliver(1)  # arrive 121, unload done 123
    assert [(k, i) for (_, k, i) in sim.events[-3:]] == [("deliver", 0), ("expire", 1), ("release", 2)]
    assert d.status == "pending"


def test_box_expiring_during_other_unload_is_disposed(line_map):
    a = make_order(0, 1, M, 2, 0.5, 0.0)  # medicine, deadline 90
    b = make_order(1, 0, M, 2, 0.6, 61.0)  # plasma, deadline 91, expiry 121
    sim = make_sim(line_map, [a, b])
    sim.do_pick(a)
    sim.advance_to(61.0)
    sim.do_pick(b)
    sim.advance_to(117.0)
    sim.do_deliver(2)  # arrive 120: a unloaded 120-122, b expired at 121
    assert a.status == "delivered" and a.end_time == pytest.approx(122.0)
    assert b.status == "expired" and b.disposed_time == pytest.approx(124.0)
    assert sim.onboard == []
