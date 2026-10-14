"""Tests for grid A* (spec: 8-connected, diagonal cost sqrt(2), no corner cutting)."""
import math

import numpy as np
import pytest

from medical_uav.env.astar import astar, path_cost_cells


def test_open_grid_diagonal():
    blocked = np.zeros((5, 5), dtype=bool)
    path = astar(blocked, (0, 0), (4, 4))
    assert path[0] == (0, 0) and path[-1] == (4, 4)
    assert path_cost_cells(path) == pytest.approx(4 * math.sqrt(2))


def test_detour_around_wall():
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[0:4, 2] = True  # wall with a gap at row 4
    path = astar(blocked, (0, 0), (0, 4))
    assert all(not blocked[r, c] for r, c in path)
    assert (4, 2) in path
    assert path_cost_cells(path) > 4


def test_no_corner_cutting():
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[0, 1] = True
    path = astar(blocked, (0, 0), (1, 1))
    assert path == [(0, 0), (1, 0), (1, 1)]


def test_unreachable_returns_none():
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[:, 1] = True
    assert astar(blocked, (0, 0), (0, 2)) is None
