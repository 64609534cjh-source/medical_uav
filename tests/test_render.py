import numpy as np
import pytest

from medical_uav.env import MedicalUAVEnv
from medical_uav.env.config import M
from medical_uav.env.render import Renderer, drone_state_at
from tests.conftest import make_order, make_sim


def test_position_interpolates_along_flight(line_map):
    sim = make_sim(line_map, [make_order(0, 1, M, 2, 0.5, 0.0)])
    sim.do_pick(sim.orders[0])  # t = 2 at M
    sim.do_deliver(2)  # fly 2 -> 5 (3 km)
    s = drone_state_at(sim, 3.5)
    assert s["xy"] == pytest.approx(np.array([1.5, 0.0]))
    assert s["activity"] == "fly" and len(s["cargo"]) == 1
    assert s["battery"] == pytest.approx(100 - 1.5 * 3.5 * 1.125)


def test_cargo_empty_after_delivery(line_map):
    sim = make_sim(line_map, [make_order(0, 1, M, 2, 0.5, 0.0)])
    sim.do_pick(sim.orders[0])
    sim.do_deliver(2)
    assert drone_state_at(sim, sim.t)["cargo"] == []


def test_pending_visible_only_after_release(line_map):
    sim = make_sim(line_map, [make_order(0, 1, M, 2, 0.5, 5.0)])
    sim.advance_to(10.0)
    assert drone_state_at(sim, 4.0)["pending"] == []
    assert [o.id for o in drone_state_at(sim, 6.0)["pending"]] == [0]


def test_draw_returns_rgb_frame(line_map):
    sim = make_sim(line_map, [make_order(0, 1, M, 2, 0.5, 0.0)])
    frame = Renderer(line_map, sim.cfg).draw(sim, 0.0)
    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3


def test_env_render_rgb_array():
    env = MedicalUAVEnv(render_mode="rgb_array")
    env.reset(seed=0)
    env.step(25)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3
