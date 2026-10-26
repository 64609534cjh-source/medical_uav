import math

import pytest

from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.config import M
from medical_uav.metrics import episode_metrics
from tests.conftest import make_order, make_sim, run_episode


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


def test_behaviour_metrics_on_a_built_day(line_map):
    from medical_uav.metrics import behaviour_metrics

    a = make_order(0, 1, M, 2, 0.5, 0.0)
    b = make_order(1, 1, M, 2, 0.5, 70.0)  # released in the 60-180 peak
    sim = make_sim(line_map, [a, b])
    sim.advance_to(70.0)
    sim.do_pick(a)
    sim.do_pick(b)
    sim.do_deliver(2)  # one stop, two boxes unloaded
    sim.battery = 50.0
    sim.do_charge(M, 80.0)  # one charge starting at about 50 - flight back
    log = [(0.0, "PICK medicine M->H2", 0.0), (2.0, "PICK medicine M->H2", 0.0),
           (4.0, "DELIVER(H2)", 2.0), (9.0, "CHARGE(C1,80)", 0.0), (20.0, "WAIT", 0.0)]
    m = behaviour_metrics(sim, log)
    assert m["boxes_per_stop"] == 2.0 and m["charges"] == 1
    assert m["battery_at_charge"] == pytest.approx(sim.segments[-1].battery0)
    assert m["wait_share"] == pytest.approx(0.2) and m["return_share"] == 0.0
    assert m["on_time_peak"] == 1.0 and m["on_time_offpeak"] == 1.0
    assert m["charge_full_share"] == 0.0  # the one charge went to 80 %
    assert m["peak_charge_share"] == 1.0  # all charging minutes inside the 60-180 peak
    assert m["battery_at_peak"] == 100.0  # full when the first peak began; the second not reached
    assert m["released_peak"] == 1 and m["on_time_peak_count"] == 1  # order b, released at 70
    charge = sim.segments[-1]
    assert m["charge_minutes"] == pytest.approx(charge.t1 - charge.t0)
    assert m["peak_charge_minutes"] == pytest.approx(charge.t1 - charge.t0)


def test_analyze_behaviour_script(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(repo / "scripts" / "analyze_behaviour.py"),
                             "--checkpoints", str(repo / "checkpoints" / "bc_set.pt"),
                             "--methods", "slack", "--loads", "medium", "--days", "1",
                             "--out", str(tmp_path / "beh")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    names = {p.name for p in (tmp_path / "beh").iterdir()}
    assert {"behaviour.csv", "battery_day.png", "on_time_by_hour.png"} <= names


def test_analyze_behaviour_runs_the_energy_conditions(tmp_path):
    import csv
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(repo / "scripts" / "analyze_behaviour.py"),
                             "--checkpoints", str(repo / "checkpoints" / "bc_set.pt"),
                             "--methods", "slack100,prepeak", "--mask-full-charge",
                             "--loads", "medium", "--seeds", "select", "--days", "1",
                             "--out", str(tmp_path / "beh")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(open(tmp_path / "beh" / "behaviour.csv")))
    assert {r["method"] for r in rows} == {"slack100", "prepeak", "bc_set", "bc_set/no100"}
    assert all(r["seed"] == "100040" for r in rows)
    prepeak = next(r for r in rows if r["method"] == "prepeak")
    assert prepeak["overrides"] in ("0", "1", "2")


def test_boxes_per_stop_counts_only_successful_deliveries(line_map):
    from medical_uav.metrics import behaviour_metrics

    a = make_order(0, 0, M, 5, 0.6, 0.0)  # plasma to H5 (6 km), expires at 60
    b = make_order(1, 1, M, 5, 0.5, 0.0)  # medicine to H5
    sim = make_sim(line_map, [a, b])
    sim.do_pick(a)
    sim.do_pick(b)
    sim.advance_to(52.5)  # arrive 58.5: a is unloaded first and expires during it (at 60)
    sim.do_deliver(5)
    m = behaviour_metrics(sim, [(0.0, "PICK", 0.0)])
    assert a.status == "expired" and b.status == "delivered"
    assert m["boxes_per_stop"] == 1.0  # one stop, one box actually delivered
