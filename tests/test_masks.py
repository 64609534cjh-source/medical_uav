from medical_uav.env.actions import CHARGE0, DELIVER0, N_ACTIONS, RETURN, WAIT, action_name, decode
from medical_uav.env.config import C2, M
from medical_uav.env.masks import compute_action_mask
from tests.conftest import make_order, make_sim


def test_decode_table():
    assert decode(3) == ("pick", 3) and decode(DELIVER0 + 2) == ("deliver", 2)
    assert decode(CHARGE0 + 4) == ("charge", (C2, 80.0))
    assert decode(24) == ("return", None) and decode(25) == ("wait", None)
    assert action_name(CHARGE0 + 4) == "CHARGE(C2,80)"
    assert action_name(DELIVER0 + 2) == "DELIVER(H2)"


def test_wait_always_valid_and_return_masked_at_center(line_map):
    sim = make_sim(line_map, [])
    mask = compute_action_mask(sim, [])
    assert mask.shape == (N_ACTIONS,) and mask[WAIT] and not mask[RETURN]


def test_pick_valid_for_simple_order(line_map):
    o = make_order(0, 0, M, 1, 0.7, 0.0)
    sim = make_sim(line_map, [o])
    mask = compute_action_mask(sim, [o])
    assert mask[0] and not mask[1]


def test_box_limit(line_map):
    orders = [make_order(i, 1, M, 1, 0.4, 0.0) for i in range(3)] + [make_order(3, 2, 2, M, 0.3, 0.0)]
    sim = make_sim(line_map, orders)
    for o in orders[:3]:
        sim.do_pick(o)  # 3 boxes, 1.2 kg: only the box limit binds
    assert not compute_action_mask(sim, [orders[3]])[0]


def test_weight_limit(line_map):
    a, b = make_order(0, 0, M, 1, 0.9, 0.0), make_order(1, 0, M, 2, 0.9, 0.0)
    c = make_order(2, 1, M, 3, 0.6, 0.0)
    sim = make_sim(line_map, [a, b, c])
    sim.do_pick(a)
    sim.do_pick(b)  # 1.8 kg
    assert not compute_action_mask(sim, [c])[0]  # 2.4 kg > 2.0


def test_pick_masked_if_cannot_load_before_expiry(line_map):
    o = make_order(0, 2, 5, M, 0.3, 0.0)  # sample at H5 (6 km away), expires at 120
    sim = make_sim(line_map, [o])
    sim.advance_to(113.0)  # arrive 119, load done 121 > 120
    assert not compute_action_mask(sim, [o])[0]


def test_pick_masked_when_battery_too_low(line_map):
    o = make_order(0, 2, 5, M, 0.4, 0.0)
    sim = make_sim(line_map, [o])
    sim.location, sim.battery = 5, 20.0  # at H5; to C2 with 0.4 kg costs 9.625
    assert not compute_action_mask(sim, [o])[0]


def test_deliver_only_where_cargo_goes(line_map):
    o = make_order(0, 0, M, 3, 0.7, 0.0)
    sim = make_sim(line_map, [o])
    sim.do_pick(o)
    mask = compute_action_mask(sim, [])
    assert mask[DELIVER0 + 3] and mask[DELIVER0:DELIVER0 + 6].sum() == 1


def test_charge_targets(line_map):
    sim = make_sim(line_map, [])
    sim.battery = 100.0
    # at M nothing to charge; C2 arrival = 100 - 12.25 = 87.75, so only (C2, 100) is valid
    charge = list(compute_action_mask(sim, [])[CHARGE0:CHARGE0 + 6])
    assert charge == [False, False, False, False, False, True]
    sim.battery = 50.0
    m = compute_action_mask(sim, [])
    assert m[CHARGE0 + 0] and m[CHARGE0 + 1] and m[CHARGE0 + 2]  # C1 60/80/100
    assert m[CHARGE0 + 3]  # C2: arrival 37.75, 60 > 42.75


def test_expired_cargo_frees_capacity_at_origin(line_map):
    old = [make_order(i, 0, M, 1, 0.6, 0.0) for i in range(3)]  # plasma, expire at 60
    new = make_order(3, 1, M, 2, 0.5, 55.0)
    sim = make_sim(line_map, old + [new])
    for o in old:
        sim.do_pick(o)
    sim.do_charge(C2, 100.0)
    sim.advance_to(57.0)  # at C2, still 3 boxes
    assert compute_action_mask(sim, [new])[0]  # arrive at M 60.5 >= expiry 60: disposed
