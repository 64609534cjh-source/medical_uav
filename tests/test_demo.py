import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_demo_rejects_a_frame_step_that_is_not_positive():
    result = subprocess.run([sys.executable, "scripts/demo.py", "--no-video",
                             "--minutes-per-frame", "-1"], capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 2 and "error:" in result.stderr
