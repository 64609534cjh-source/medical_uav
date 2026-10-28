import pytest

from medical_uav.baselines.insertion import insertion_policy, plan_route, route_score
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import DELIVER0, PICK0
from medical_uav.env.config import M
from tests.conftest import make_order, make_sim
from tests.test_heuristics import SimEnv


def test_route_score_counts_on_time_and_completion_times(line_map):
    a = make_order(0, 1, M, 2, 0.5, 0.0)  # medicine M -> H2 (3 km), deadline 90
    sim = make_sim(line_map, [a])
    # load 0-2, fly to H2 2-5, unload 5-7
    assert route_score(sim, [("pick", a), ("deliver", 2)]) == (1, 1, -7.0)
    assert route_score(sim, [("deliver", 2)]) is None  # nothing on board for H2


def test_route_score_rejects_a_fourth_box(line_map):
    orders = [make_order(i, 1, M, 1 + i, 0.3, 0.0) for i in range(4)]
    sim = make_sim(line_map, orders)
    assert route_score(sim, [("pick", o) for o in orders]) is None


def test_insertion_batches_an_order_for_the_same_destination(line_map):
    a, c = make_order(0, 1, M, 2, 0.5, 0.0), make_order(1, 1, M, 2, 0.5, 0.0)
    sim = make_sim(line_map, [a, c])
    sim.do_pick(a)  # at M with a box for H2; c waits at M for H2
    env = SimEnv(sim)
    assert insertion_policy(env) == PICK0 + env.slots.index(c)


def test_insertion_delivers_when_the_drone_is_full(line_map):
    boxes = [make_order(i, 1, M, 1 + i, 0.4, 0.0) for i in range(3)]  # to H1, H2, H3
    extra = make_order(3, 1, M, 4, 0.4, 0.0)
    sim = make_sim(line_map, boxes + [extra])
    for o in boxes:
        sim.do_pick(o)
    assert insertion_policy(SimEnv(sim)) == DELIVER0 + 1  # nearest drop frees a place first


@pytest.mark.parametrize("load", ["low", "medium", "high"])
def test_insertion_only_chooses_valid_actions(load):
    env = MedicalUAVEnv(load=load)
    _, info = env.reset(options={"demand_seed": 1})
    done = False
    while not done:
        a = insertion_policy(env)
        assert info["action_mask"][a]
        _, _, done, _, info = env.step(a)


def test_orders_that_can_only_be_late_go_after_the_on_time_ones(line_map):
    late = make_order(0, 0, M, 1, 0.7, 0.0)  # plasma, deadline 30, expiry 60: late at t=45
    fresh = make_order(1, 1, M, 2, 0.5, 40.0)  # medicine, deadline 130
    sim = make_sim(line_map, [late, fresh])
    sim.advance_to(45.0)
    env = SimEnv(sim)
    # appended after delivering fresh, late would be back at M at 55 and unloaded after expiry
    assert plan_route(env) == [("pick", fresh), ("deliver", 2)]
    assert ("pick", late) in plan_route(env, late="insert")
    assert insertion_policy(env) == PICK0 + env.slots.index(fresh)


def test_a_late_order_that_can_still_be_served_is_appended(line_map):
    late = make_order(0, 1, M, 1, 0.5, 0.0)  # medicine, deadline 90, expiry 180: late at t=95
    fresh = make_order(1, 1, M, 2, 0.5, 90.0)  # medicine, deadline 180
    sim = make_sim(line_map, [late, fresh])
    sim.advance_to(95.0)
    env = SimEnv(sim)
    on_time = [("pick", fresh), ("deliver", 2)]
    assert plan_route(env) == on_time + [("pick", late), ("deliver", 1)]  # unloaded at 111
    assert plan_route(env, late="skip") == on_time
