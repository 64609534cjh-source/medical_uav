import numpy as np
import pytest

from medical_uav.env.config import M
from medical_uav.env.masks import compute_action_mask
from medical_uav.env.observation import (GLOBAL_DIM, OBS_DIM, ONBOARD_DIM, PENDING_DIM,
                                         build_observation, rank_pending)
from tests.conftest import make_order, make_sim


def observe(sim):
    slots = rank_pending(sim)
    return build_observation(sim, slots, compute_action_mask(sim, slots))


def test_two_tier_ranking_puts_on_time_feasible_first(line_map):
    late = make_order(0, 0, M, 1, 0.7, 0.0)  # plasma: deadline 30, expiry 60
    fresh = make_order(1, 1, M, 2, 0.5, 40.0)  # medicine: deadline 130
    sim = make_sim(line_map, [late, fresh])
    sim.advance_to(50.0)
    assert [o.id for o in rank_pending(sim)] == [1, 0]


def test_at_most_twelve_slots(line_map):
    orders = [make_order(i, 1, M, 1 + i % 5, 0.4, 0.0) for i in range(20)]
    sim = make_sim(line_map, orders)
    assert len(rank_pending(sim)) == 12


def test_observation_shape_and_slot_encoding(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 1, 0.7, 0.0)])
    obs = observe(sim)
    assert OBS_DIM == 403
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32 and np.isfinite(obs).all()
    s0 = obs[GLOBAL_DIM:GLOBAL_DIM + PENDING_DIM]
    assert s0[0] == 1.0 and s0[1] == 1.0  # valid, type plasma
    assert s0[26] == 1.0  # pick feasible
    assert obs[GLOBAL_DIM + PENDING_DIM] == 0.0  # slot 1 is padding


def test_late_order_has_negative_time_to_deadline(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 1, 0.7, 0.0)])
    sim.advance_to(45.0)
    obs = observe(sim)
    ttd = obs[GLOBAL_DIM + 1 + 4 + 6 + 6 + 1]
    assert ttd == pytest.approx(-0.5)


def test_global_block(line_map):
    sim = make_sim(line_map, [make_order(0, 0, M, 1, 0.7, 0.0)])
    obs = observe(sim)
    assert obs[6] == pytest.approx(1.0)  # battery / 100
    assert obs[9] == 1.0 and obs[10:16].sum() == 0  # located at M
    assert obs[24] == pytest.approx(1 / 30)  # one pending order


def test_onboard_block(line_map):
    o = make_order(0, 0, M, 3, 0.7, 0.0)
    sim = make_sim(line_map, [o])
    sim.do_pick(o)
    obs = observe(sim)
    start = OBS_DIM - 3 * ONBOARD_DIM
    b0 = obs[start:start + ONBOARD_DIM]
    assert b0[0] == 1.0 and b0[1] == 1.0  # valid, plasma
    assert b0[5 + 3] == 1.0  # destination H3
    assert b0[15] == pytest.approx((4 + 2) / 30)  # fly 4 km + unload
