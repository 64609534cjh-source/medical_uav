import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_demo_offers_policy_load_and_seed():
    result = subprocess.run([sys.executable, "scripts/demo.py", "--help"],
                            capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0
    assert all(option in result.stdout for option in ("--policy", "--load", "--seed"))


def test_demo_network_runs_from_any_directory(tmp_path):
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "demo_network.py"), "--quick"],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    for word in ("forward", "backward", "logits", "value", "on time"):
        assert word in result.stdout
