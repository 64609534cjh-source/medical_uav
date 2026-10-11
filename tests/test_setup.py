"""Smoke test: the package and its third-party dependencies can be imported."""


def test_package_imports():
    import imageio  # noqa: F401
    import gymnasium  # noqa: F401
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401

    import medical_uav  # noqa: F401


def _pins(path):
    pins = set()
    for line in path.read_text().splitlines():
        line = line.strip().lstrip("-").strip()
        if "==" in line and not line.startswith("#"):
            pins.add(line.lower())
    return pins


def test_requirements_match_environment_yml():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    requirements = _pins(repo / "requirements.txt")
    assert requirements and requirements == _pins(repo / "environment.yml")
