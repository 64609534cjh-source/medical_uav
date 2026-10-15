import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_demo_offers_policy_load_and_seed():
    result = subprocess.run([sys.executable, "scripts/demo.py", "--help"],
                            capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0
    assert all(option in result.stdout for option in ("--policy", "--load", "--seed"))
