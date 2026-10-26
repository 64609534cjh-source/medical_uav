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
    for word in ("forward", "backward", "logits", "value estimate", "on time"):
        assert word in result.stdout
    assert "count_" not in result.stdout  # per-type rates only, not the order counts


def test_demo_writes_explanation_and_day_videos(tmp_path):
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "demo.py"), "--load", "low",
                             "--explain-count", "2", "--day-speed", "60"],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    names = {p.name for p in (tmp_path / "outputs").iterdir()}
    assert "demo_slack_low_seed0_explained.mp4" in names and "demo_slack_low_seed0_day.mp4" in names


def test_demo_runs_the_trained_agent(tmp_path):
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "demo.py"), "--policy", "ppo",
                             "--load", "low", "--explain-count", "1", "--day-speed", "60"],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "PPO set network" in result.stdout
    assert (tmp_path / "outputs" / "demo_ppo_low_seed0_day.mp4").exists()
