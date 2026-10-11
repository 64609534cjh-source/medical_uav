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
