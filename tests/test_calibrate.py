import importlib.util
from pathlib import Path

from medical_uav.env import MedicalUAVEnv

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calibrate.py"


def load_calibrate():
    spec = importlib.util.spec_from_file_location("calibrate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_theta_is_tuned_for_every_heuristic():
    calibrate = load_calibrate()
    results = calibrate.tune_thetas(MedicalUAVEnv(), seeds=[100_000], lam=0.08, thetas=[20.0, 40.0])
    assert set(results) == {"edf", "slack"}
    for by_theta in results.values():
        assert set(by_theta) == {20.0, 40.0}
        assert all(0.0 <= rate <= 1.0 for rate in by_theta.values())


def test_select_thetas_picks_the_best_per_heuristic():
    calibrate = load_calibrate()
    results = {"edf": {20.0: 0.5, 40.0: 0.6}, "slack": {20.0: 0.8, 30.0: 0.7}}
    assert calibrate.select_thetas(results) == {"edf": 40.0, "slack": 20.0}


def test_random_rows_do_not_depend_on_run_order():
    calibrate = load_calibrate()
    env = MedicalUAVEnv()
    thetas = {"edf": 20.0, "slack": 20.0}
    first = calibrate.run_day(env, calibrate.make_policy("random", 100_000, thetas), 100_000, 0.08)
    calibrate.run_day(env, calibrate.make_policy("random", 100_001, thetas), 100_001, 0.08)
    again = calibrate.run_day(env, calibrate.make_policy("random", 100_000, thetas), 100_000, 0.08)
    assert first["on_time_rate"] == again["on_time_rate"]
    assert first["energy_used"] == again["energy_used"]
