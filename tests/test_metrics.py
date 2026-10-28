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
    assert m["battery_at_peak_1"] == 100.0 and math.isnan(m["battery_at_peak_2"])
    assert m["released_peak"] == 1 and m["on_time_peak_count"] == 1  # order b, released at 70
    charge = sim.segments[-1]
    assert m["charge_minutes"] == pytest.approx(charge.t1 - charge.t0)
    # charged at M, nothing after
    assert m["charge_at_center"] == 1.0 and m["charge_then_pickup"] == 0.0
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
    assert float(prepeak["km_to_charger"]) > 0  # distance breakdown in every row
    assert 0 <= float(prepeak["on_time_plasma"]) <= 1  # per-type service
    assert 0 <= float(prepeak["overflow_share"]) <= 1  # decisions with orders beyond the slots


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


def test_distance_breakdown_on_a_built_day(line_map):
    from medical_uav.metrics import distance_breakdown

    a = make_order(0, 1, M, 2, 0.5, 0.0)
    b = make_order(1, 1, M, 2, 0.5, 0.0)
    sim = make_sim(line_map, [a, b])
    sim.do_pick(a)
    sim.do_pick(b)  # both at M: no flight
    sim.do_deliver(2)  # loaded flight M -> H2
    sim.battery = 50.0
    sim.do_charge(M, 80.0)  # empty flight H2 -> M, then charge
    d = distance_breakdown(sim)
    km = line_map.dist_km[M, 2]
    assert d["km_to_delivery"] == pytest.approx(km) and d["km_to_charger"] == pytest.approx(km)
    assert d["km_to_pickup"] == 0.0 and d["km_reposition"] == 0.0
    assert d["km_empty"] == pytest.approx(km) and d["futile_pickups"] == 0
    assert d["pickups_in_place"] == 1.0  # both boxes loaded at M without a flight first
    assert sum(d[k] for k in ("km_to_pickup", "km_to_delivery", "km_to_charger",
                              "km_reposition")) == pytest.approx(sim.km_flown)


def test_distance_breakdown_sums_to_the_distance_flown():
    from medical_uav.metrics import distance_breakdown

    env = MedicalUAVEnv(load="high")
    env.reset(options={"demand_seed": 100_000})
    done = False
    while not done:
        _, _, done, _, _ = env.step(slack_aware_policy(env))
    d = distance_breakdown(env.sim)
    parts = [d[k] for k in ("km_to_pickup", "km_to_delivery", "km_to_charger", "km_reposition")]
    assert sum(parts) == pytest.approx(env.sim.km_flown) and min(parts[:3]) > 0
    assert 0 < d["km_empty"] < env.sim.km_flown


def test_pickup_metrics_skip_unloading_expired_boxes():
    from medical_uav.env.simulator import Segment
    from medical_uav.metrics import behaviour_metrics, distance_breakdown

    env = MedicalUAVEnv(load="low")
    env.reset(options={"demand_seed": 100_000})
    sim = env.sim
    seg = lambda kind, a, b: Segment(kind, 0.0, 1.0, a, b, 50.0, 50.0)
    # a flight, an expired box unloaded on arrival, then a pickup: the pickup needed the flight
    sim.segments = [seg("fly", M, 2), seg("dispose", 2, 2), seg("load", 2, 2)]
    assert distance_breakdown(sim)["pickups_in_place"] == 0.0
    # a charge at M, an expired box unloaded, then a pickup at M: charge then pickup
    sim.segments = [seg("charge", M, M), seg("dispose", M, M), seg("load", M, M)]
    assert behaviour_metrics(sim, [])["charge_then_pickup"] == 1.0


def test_failure_analysis_script(tmp_path):
    import csv
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(repo / "scripts" / "failure_analysis.py"),
                             "--methods", "slack", "--loads", "medium", "--days", "4",
                             "--processes", "2", "--out", str(tmp_path / "f")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    orders = list(csv.DictReader(open(tmp_path / "f" / "orders.csv")))
    assert {o["outcome"] for o in orders} <= {"on time", "late", "expired before pickup",
                                              "expired on board"}
    assert "snowball check" in result.stdout and "workload" in result.stdout


def test_empty_flight_and_backlog(line_map):
    sample = make_order(0, 2, 1, M, 0.3, 0.0)  # waiting at H1 (2 km) for M
    medicine = make_order(1, 1, M, 2, 0.5, 0.0)  # never picked; expires at 180
    sim = make_sim(line_map, [sample, medicine])
    sim.do_pick(sample)  # empty flight M -> H1, loaded at t = 4
    sim.do_deliver(M)  # loaded flight back
    sim.advance_to(200.0)
    m = episode_metrics(sim)
    assert m["empty_km"] == pytest.approx(2.0) and m["km_flown"] == pytest.approx(4.0)
    assert m["mean_backlog"] == pytest.approx((4.0 + 180.0) / 200.0)  # waiting minutes per minute
