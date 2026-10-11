import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("args", [["--fps", "0"], ["--minutes-per-frame", "-1"],
                                  ["--out", "day.png"]])
def test_demo_rejects_bad_arguments(args):
    result = subprocess.run([sys.executable, "scripts/demo.py", "--no-video", *args],
                            capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 2 and "error:" in result.stderr
