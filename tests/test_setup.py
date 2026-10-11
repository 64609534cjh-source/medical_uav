"""Smoke test: the package and its third-party dependencies can be imported."""


def test_package_imports():
    import imageio  # noqa: F401
    import gymnasium  # noqa: F401
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401

    import medical_uav  # noqa: F401
