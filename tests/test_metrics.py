import math

import pytest

from medical_uav.baselines.heuristics import run_episode, slack_aware_policy
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.config import M
from medical_uav.metrics import episode_metrics
from tests.conftest import make_order, make_sim


def test_metrics_by_type_and_delivered():
    m = run_episode(MedicalUAVEnv(), slack_aware_policy, 0)
    assert m["delivered"] == round(m["served_rate"] * m["n_orders"])
    assert set(m["on_time_by_type"]) == {"plasma", "medicine", "sample", "supplies"}


def test_on_time_by_type_values(line_map):
    plasma, medicine = make_order(0, 0, M, 1, 0.7, 0.0), make_order(1, 1, M, 2, 0.5, 0.0)
    sim = make_sim(line_map, [plasma, medicine])
    sim.do_pick(plasma)
    sim.do_deliver(1)  # plasma on time; medicine never served
    sim.advance_to(480.0)
    by_type = episode_metrics(sim)["on_time_by_type"]
    assert by_type["plasma"] == 1.0 and by_type["medicine"] == 0.0
    assert math.isnan(by_type["sample"]) and math.isnan(by_type["supplies"])


def test_partial_day_counts_only_released_orders(line_map):
    early, late = make_order(0, 1, M, 1, 0.5, 0.0), make_order(1, 1, M, 2, 0.5, 100.0)
    sim = make_sim(line_map, [early, late])
    sim.do_pick(early)
    sim.do_deliver(1)
    m = episode_metrics(sim)  # t = 6: the second order is not released yet
    assert m["n_orders"] == 1 and m["on_time_rate"] == pytest.approx(1.0)
