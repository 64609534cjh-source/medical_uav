import numpy as np
import pytest

from medical_uav.baselines.heuristics import fcfs_policy, slack_aware_policy
from medical_uav.env.actions import PICK0
from medical_uav.evaluation import evaluate, heuristic, paired_ci


def test_evaluate_rows_and_columns():
    rows = evaluate({"slack": heuristic(slack_aware_policy), "fcfs": heuristic(fcfs_policy)},
                    ["low"], range(100_000, 100_002))
    assert len(rows) == 4
    keys = {"method", "load", "seed", "env_version", "on_time", "n_orders", "on_time_rate",
            "served_rate", "expiry_rate", "return", "charge_share", "on_time_plasma",
            "decision_ms_mean", "decision_ms_p95"}
    assert keys <= set(rows[0])
    assert {r["method"] for r in rows} == {"slack", "fcfs"}


def test_evaluate_rejects_invalid_action():
    def bad(env, obs, mask):
        return int(np.flatnonzero(~mask)[0]) if (~mask).any() else PICK0
    with pytest.raises(AssertionError):
        evaluate({"bad": bad}, ["low"], [100_000])


def test_paired_ci_constant():
    assert paired_ci(np.full(20, 0.1)) == pytest.approx((0.1, 0.1, 0.1))


def test_checkpoint_labels_are_unique():
    from medical_uav.evaluation import checkpoint_labels

    labels = checkpoint_labels(["runs/a/ckpt_01.pt", "runs/b/ckpt_01.pt", "checkpoints/bc_set.pt"])
    assert labels == ["a/ckpt_01", "b/ckpt_01", "bc_set"]
    # a run's checkpoint keeps its run name even when its file name is unique
    labels = checkpoint_labels(["runs/a/ckpt_01.pt", "runs/b/ckpt_02.pt", "checkpoints/ppo_set.pt"])
    assert labels == ["a/ckpt_01", "b/ckpt_02", "ppo_set"]


def test_rows_have_per_type_counts():
    from medical_uav.evaluation import run_day

    row = run_day(heuristic(slack_aware_policy), "medium", 100_000)
    types = ["plasma", "medicine", "sample", "supplies"]
    assert sum(row[f"released_{t}"] for t in types) == row["n_orders"]
    assert sum(row[f"on_time_count_{t}"] for t in types) == row["on_time"]


TYPES = ["plasma", "medicine", "sample", "supplies"]


def fake(low, medium, high, types=None):
    """Selection-day results of one checkpoint: on-time rates per load and pooled per type."""
    types = types or {}
    out = {}
    for load, rate in (("low", low), ("medium", medium), ("high", high)):
        out[load] = {"on_time_rate": rate,
                     **{f"on_time_{t}": types.get((load, t), rate) for t in TYPES}}
    return out


SLACK = fake(0.99, 0.81, 0.66)


def test_select_prefers_highest_eligible():
    from medical_uav.evaluation import select_checkpoint

    results = {"run/ckpt_1.pt": fake(0.99, 0.90, 0.80), "run/ckpt_2.pt": fake(0.95, 0.95, 0.90),
               "run/ckpt_3.pt": fake(0.99, 0.91, 0.82)}
    chosen, rows = select_checkpoint(results, SLACK)
    assert chosen == "run/ckpt_3.pt"
    assert not next(r for r in rows if r["path"] == "run/ckpt_2.pt")["eligible"]


def test_select_type_gate():
    from medical_uav.evaluation import select_checkpoint

    results = {"run/ckpt_1.pt": fake(0.99, 0.90, 0.85, {("high", "plasma"): 0.61})}
    chosen, rows = select_checkpoint(results, SLACK)
    assert chosen is None and "plasma" in rows[0]["reason"]


def test_select_none_eligible():
    from medical_uav.evaluation import select_checkpoint

    results = {"a.pt": fake(0.9, 0.9, 0.9), "b.pt": fake(0.8, 0.9, 0.9)}
    chosen, rows = select_checkpoint(results, SLACK)
    assert chosen is None and all(r["reason"] for r in rows)


def test_select_tie_goes_to_earlier():
    from medical_uav.evaluation import select_checkpoint

    results = {"run/ckpt_2.pt": fake(0.99, 0.90, 0.80), "run/ckpt_1.pt": fake(0.99, 0.90, 0.80)}
    assert select_checkpoint(results, SLACK)[0] == "run/ckpt_1.pt"


def test_select_checkpoints_script(tmp_path):
    import json
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    run = tmp_path / "run"
    run.mkdir()
    shutil.copy(repo / "checkpoints" / "bc_set.pt", run / "ckpt_00000001.pt")
    result = subprocess.run([sys.executable, str(repo / "scripts" / "select_checkpoints.py"),
                             str(run), "--days", "2", "--processes", "2",
                             "--out", str(tmp_path / "sel.json")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "sel.json").read_text())
    rows = report["runs"][str(run)]["checkpoints"]
    assert len(rows) == 1 and "eligible" in rows[0] and "medium" in report["slack"]
    result = subprocess.run([sys.executable, str(repo / "scripts" / "select_checkpoints.py"),
                             str(run), "--days", "1", "--maps", "select", "--processes", "4",
                             "--out", str(tmp_path / "sel2.json")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report2 = json.loads((tmp_path / "sel2.json").read_text())
    assert report2["metadata"]["maps"] == "select"
    # one selection day on each of the ten selection maps differs from two days on map 0
    assert report2["slack"]["medium"]["on_time_rate"] != report["slack"]["medium"]["on_time_rate"]


def test_evaluate_script_in_parallel_with_the_planner(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "eval.csv"
    result = subprocess.run([sys.executable, str(repo / "scripts" / "evaluate.py"), "--methods",
                             "slack,rollout", "--loads", "low", "--days", "2", "--processes", "2",
                             "--out", str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert out.read_text().count("\n") == 5 and "rollout" in result.stdout


def test_evaluate_script_runs_the_scenario_planner_day_by_day(tmp_path):
    import csv
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "eval.csv"
    result = subprocess.run([sys.executable, str(repo / "scripts" / "evaluate.py"), "--methods",
                             "slack,scenario4", "--loads", "low", "--days", "2", "--processes",
                             "4", "--out", str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(open(out)))
    assert [(r["method"], r["seed"]) for r in rows] == [
        ("slack", "100000"), ("slack", "100001"), ("scenario4", "100000"), ("scenario4", "100001")]


def test_parallel_evaluation_job_uses_one_thread():
    import importlib.util
    from pathlib import Path

    import torch

    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    before = torch.get_num_threads()
    try:
        script.run_job(("slack", None, "slack", "low", [100_000]))
        assert torch.get_num_threads() == 1  # parallel workers must not each use every core
    finally:
        torch.set_num_threads(before)


def test_select_rejects_missing_rates():
    from medical_uav.evaluation import select_checkpoint

    chosen, rows = select_checkpoint({"a.pt": fake(0.99, 0.9, float("nan"))}, SLACK)
    assert chosen is None and "not finite" in rows[0]["reason"]


def test_first_sustained_crossing():
    from medical_uav.evaluation import first_sustained_crossing

    decisions = [1, 2, 3, 4, 5]
    assert first_sustained_crossing(decisions, [.5, .9, .7, .9, .95], .8) == 4  # 2 dips back
    # the last evaluation cannot be confirmed
    assert first_sustained_crossing(decisions, [.5, .6, .7, .8, .9], .85) is None
    assert first_sustained_crossing(decisions, [.9, .9, .9, .9, .9], .8) == 1


def test_curve_mean_over_a_window():
    from medical_uav.evaluation import curve_mean

    # trapezoid over [1, 3] of a straight line from .5 to .9, divided by the width: .7
    assert curve_mean([0, 1, 2, 3], [.1, .5, .7, .9], 1, 3) == pytest.approx(.7)


def test_plot_training_runs_on_logged_curves(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    for name, scores in [("a_s11", (.80, .85, .86)), ("b_s11", (.84, .86, .87))]:
        (tmp_path / name).mkdir()
        lines = ["decisions,seconds,on_time_medium,on_time_high"]
        lines += [f"{d},0,{s},{s}" for d, s in zip((500, 1000, 1500), scores)]
        (tmp_path / name / "evals.csv").write_text("\n".join(lines))
        (tmp_path / name / "run.json").write_text('{"maps": null}')
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(repo / "scripts" / "plot_training.py"),
                             f"A:{tmp_path / 'a_s11'}", f"B:{tmp_path / 'b_s11'}",
                             "--extra-cost", "B=300", "--eval-days", "2",
                             "--out", str(tmp_path / "curves.png")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "curves.png").exists()
    # target .95 * .86 = .817: A reaches it at 1000, B at 500 (+300 extra cost = 800)
    assert "a_s11" in result.stdout and "b_s11" in result.stdout
    assert "B: saving vs A, median 200 decisions (with extra cost; without 500), faster in 1/1" \
        in result.stdout
    # an extra cost of the reference group counts for it: 1000 + 100 - 800
    result = subprocess.run([sys.executable, str(repo / "scripts" / "plot_training.py"),
                             f"A:{tmp_path / 'a_s11'}", f"B:{tmp_path / 'b_s11'}",
                             "--extra-cost", "B=300", "--extra-cost", "A=100", "--eval-days", "1",
                             "--out", str(tmp_path / "curves.png")], capture_output=True, text=True)
    assert "median 300 decisions" in result.stdout
    # Slack-aware is drawn on one set of maps, so single- and multi-map runs do not mix
    (tmp_path / "b_s11" / "run.json").write_text('{"maps": "train"}')
    result = subprocess.run([sys.executable, str(repo / "scripts" / "plot_training.py"),
                             f"A:{tmp_path / 'a_s11'}", f"B:{tmp_path / 'b_s11'}",
                             "--out", str(tmp_path / "curves.png")], capture_output=True, text=True)
    assert result.returncode != 0 and "different maps" in result.stderr


def test_compare_variants_pairs_runs_by_seed(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    lines = ["method,load,seed,on_time_rate,served_rate,on_time_count_plasma,released_plasma"]
    for variant, rate in [("base", 0.80), ("new", 0.85)]:
        for load in ("medium", "high"):
            for day in (1, 2, 3):
                plasma = 8 if variant == "new" else 6
                lines.append(f"{variant}_s11/ckpt_1,{load},{day},{rate},0.9,{plasma},10")
    lines.append("slack,medium,1,0.7,0.9,5,10")  # not a run: ignored
    (tmp_path / "eval.csv").write_text("\n".join(lines))
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(repo / "scripts" / "compare_variants.py"),
                             str(tmp_path / "eval.csv"), "--reference", "base"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "new (seeds 11)" in result.stdout
    assert "score    +5.0 [ +5.0,  +5.0]" in result.stdout
    assert "plasma  +20.0 [+20.0, +20.0]" in result.stdout


def test_latency_script_keeps_every_decision_time(tmp_path):
    import csv
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "latency.csv"
    result = subprocess.run([sys.executable, str(repo / "scripts" / "latency.py"), "--methods",
                             "slack", "--loads", "low", "--decisions", "100", "--passes", "2",
                             "--out", str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(open(out)))
    assert {r["pass"] for r in rows} == {"1", "2"}
    per_pass = [sum(r["pass"] == p for r in rows) for p in ("1", "2")]
    assert per_pass[0] == per_pass[1] >= 100  # whole days until at least 100 decisions
    assert "p95" in result.stdout


def test_run_day_on_another_map():
    from medical_uav.env import EnvConfig
    from medical_uav.evaluation import run_day

    policy = heuristic(slack_aware_policy)
    on_map0 = run_day(policy, "low", 100_000)
    on_map3 = run_day(policy, "low", 100_000, EnvConfig(map_seed=3))
    assert on_map0["n_orders"] == on_map3["n_orders"]  # the same orders
    assert on_map0["km_flown"] != on_map3["km_flown"]  # flown on another map


def test_map_stress_script(tmp_path):
    import csv
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "maps.csv"
    result = subprocess.run([sys.executable, str(repo / "scripts" / "map_stress.py"),
                             "--methods", "slack", "--checkpoints",
                             str(repo / "checkpoints" / "bc_set.pt"), "--maps", "1-2",
                             "--loads", "medium", "--days", "1", "--processes", "2",
                             "--out", str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(open(out)))
    assert {(r["method"], r["map"]) for r in rows} == {
        ("slack", "1"), ("slack", "2"), ("bc_set", "1"), ("bc_set", "2")}
    assert "networks - slack" in result.stdout


def test_run_day_details_add_behaviour_and_every_decision_time():
    from medical_uav.baselines.heuristics import slack_aware_policy
    from medical_uav.evaluation import heuristic, run_day

    row = run_day(heuristic(slack_aware_policy), "medium", 100_040, details=True)
    assert len(row["decision_ms"]) == row["decisions"]
    assert {"battery_at_peak", "peak_charge_minutes", "boxes_per_stop", "km_to_pickup"} <= set(row)
    assert row["on_time_rate"] == run_day(heuristic(slack_aware_policy), "medium", 100_040)["on_time_rate"]
