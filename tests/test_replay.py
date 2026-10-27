import json
import random
import shutil
import subprocess

import numpy as np
import pytest

from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.config import EPS


def snapshot(env) -> dict:
    """What the simulator holds at a decision: the player must show the same."""
    sim = env.sim
    released = [o for o in sim.orders if o.status != "future"]
    delivered = [o for o in released if o.status == "delivered"]
    return {"t": sim.t, "xy": tuple(env.city_map.node_xy[sim.location]), "battery": sim.battery,
            "cargo": sorted(o.id for o in sim.onboard), "waiting": sorted(o.id for o in sim.pending),
            "on_time": sum(o.end_time - o.release <= o.limit + EPS for o in delivered),
            "late": sum(o.end_time - o.release > o.limit + EPS for o in delivered),
            "expired": sum(o.status == "expired" for o in released)}


@pytest.fixture(scope="module")
def high_day():
    from medical_uav.env.replay import day_record, decision_record

    env = MedicalUAVEnv(load="high")
    env.reset(options={"demand_seed": 100_003})
    snaps, decisions = [], []
    done = False
    while not done:
        a = slack_aware_policy(env)
        snaps.append(snapshot(env))
        decisions.append(decision_record(env, a))
        _, _, done, _, _ = env.step(a)
    snaps.append(snapshot(env))
    return env, snaps, day_record(env, decisions, "Slack-aware rule")


def test_player_state_matches_the_simulator_at_every_decision(high_day):
    from medical_uav.env.replay import state_at

    env, snaps, record = high_day
    assert any(s["late"] for s in snaps) and any(s["expired"] for s in snaps)
    for snap in snaps:
        s = state_at(record, snap["t"])
        assert s["xy"] == pytest.approx(snap["xy"]) and s["battery"] == pytest.approx(snap["battery"])
        assert (s["cargo"], s["waiting"]) == (snap["cargo"], snap["waiting"])
        assert (s["on_time"], s["late"], s["expired"]) == \
            (snap["on_time"], snap["late"], snap["expired"])


def test_record_holds_what_the_player_needs(high_day):
    env, snaps, record = high_day
    assert record["meta"]["policy"] == "Slack-aware rule"
    assert len(record["decisions"]) == len(env.decision_log)
    flights = {(s[3], s[4]) for s in record["segments"] if s[0] == "fly"}
    assert all(f"{a}-{b}" in record["map"]["routes"] for a, b in flights)
    d = record["decisions"][0]
    assert {"t", "end", "text", "short", "facts", "top"} <= set(d) and d["top"] == []
    json.dumps(record)  # plain JSON


def test_flight_positions_follow_the_route(high_day):
    from medical_uav.env.replay import state_at

    env, _, record = high_day
    kind, t0, t1, a, b = next(s for s in record["segments"] if s[0] == "fly")[:5]
    path = np.array(record["map"]["routes"][f"{a}-{b}"])
    assert state_at(record, t0)["xy"] == pytest.approx(tuple(path[0]))
    assert state_at(record, t1 - 1e-6)["xy"] == pytest.approx(tuple(path[-1]), abs=1e-4)
    mid = state_at(record, (t0 + t1) / 2)["xy"]
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert min(np.linalg.norm(path - np.array(mid), axis=1)) <= lengths.sum() / 2 + 1e-9


@pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the player's script")
def test_the_player_script_agrees_with_python(high_day, tmp_path):
    from medical_uav.env.replay import TEMPLATE, state_at

    _, _, record = high_day
    html = TEMPLATE.read_text()
    core = html[html.index("// core:start"):html.index("// core:end")]
    rng = random.Random(0)
    times = [s[1] for s in record["segments"]] + [s[2] for s in record["segments"]]
    times += [rng.uniform(0, record["meta"]["end"]) for _ in range(300)]
    script = tmp_path / "check.js"
    script.write_text(core + "\nconst R = " + json.dumps(record) + ";\nconst T = " + json.dumps(times)
                      + ";\nconsole.log(JSON.stringify(T.map((t) => stateAt(R, t))));\n")
    out = json.loads(subprocess.run(["node", str(script)], capture_output=True, text=True,
                                    check=True).stdout)
    for t, js in zip(times, out):
        py = state_at(record, t)
        assert js["xy"] == pytest.approx(list(py["xy"]), abs=1e-9)
        assert js["battery"] == pytest.approx(py["battery"], abs=1e-9)
        for key in ("cargo", "waiting", "on_time", "late", "expired", "decision"):
            assert js[key] == py[key], (t, key)


def test_write_html_is_a_single_offline_file(high_day, tmp_path):
    from medical_uav.env.replay import write_html

    _, _, record = high_day
    record["decisions"][0]["text"] = "</script><b>"  # must not end the data block early
    path = write_html(record, tmp_path / "day.html")
    html = path.read_text()
    assert "/*DATA*/" not in html and "</script><b>" not in html
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in html and "<link" not in html and "src=" not in html


def test_opening_decision_matches_the_player_rule():
    from medical_uav.env.replay import opening_decision

    decs = [{"t": 0.0, "text": "Wait for the next order", "top": []},
            {"t": 70.0, "text": "Pick up sample at H2 for M", "top": [["a", 0.5], ["b", 0.45]]},
            {"t": 80.0, "text": "Deliver 1 sample box to M", "top": [["Deliver at M", 0.7], ["Charge C1 to 60%", 0.3]]},
            {"t": 90.0, "text": "Deliver 2 medicine boxes to H3", "top": [["Deliver at H3", 0.9], ["Wait", 0.1]]}]
    record = {"decisions": decs, "meta": {"peaks": [[60.0, 180.0]]}}
    assert opening_decision(record) == 2  # deliver-or-charge beats a closer pick call
    decs[2]["top"][1][0] = "Wait"
    assert opening_decision(record) == 1  # otherwise the closest call
    for d in decs:
        d["top"] = []
    assert opening_decision(record) == 1  # a rule: the first decision in the first peak
