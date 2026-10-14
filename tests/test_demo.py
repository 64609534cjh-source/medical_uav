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


@pytest.mark.parametrize("args", [["--minutes-per-frame", "nan"], ["--snapshots", "-5"],
                                  ["--snapshots", "nan"]])
def test_demo_rejects_non_finite_or_negative_times(args):
    result = subprocess.run([sys.executable, "scripts/demo.py", "--no-video", *args],
                            capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 2 and "error:" in result.stderr


def test_example_runs_and_saves_a_frame(tmp_path):
    frame = tmp_path / "frame.png"
    code = ("import sys; sys.path.insert(0, 'scripts'); import example; "
            f"example.main({str(frame)!r})")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stderr
    assert frame.stat().st_size > 0 and "full day:" in result.stdout
