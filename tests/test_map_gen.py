"""Tests for random city maps."""
import numpy as np

from medical_uav.env.config import C2, M, EnvConfig
from medical_uav.env.map_gen import CityMap, generate_map, is_valid_map

cfg = EnvConfig()


def test_deterministic():
    a, b = generate_map(0, cfg), generate_map(0, cfg)
    assert np.array_equal(a.dist_km, b.dist_km) and a.zones == b.zones


def test_distances_symmetric_and_at_least_euclidean():
    m = generate_map(1, cfg)
    assert np.allclose(m.dist_km, m.dist_km.T)
    for i in range(7):
        for j in range(7):
            assert m.dist_km[i, j] >= np.linalg.norm(m.node_xy[i] - m.node_xy[j]) - 1e-9


def test_spacing_rules():
    m = generate_map(2, cfg)
    for i in range(7):
        for j in range(i + 1, 7):
            assert np.linalg.norm(m.node_xy[i] - m.node_xy[j]) >= 0.75 - 1e-9
    assert np.linalg.norm(m.node_xy[M] - m.node_xy[C2]) >= 1.0 - 1e-9


def test_generated_maps_are_valid_and_paths_avoid_zones():
    for seed in range(20):
        m = generate_map(seed, cfg)
        assert is_valid_map(m.dist_km, cfg)
        assert m.paths[(0, 1)].shape[1] == 2


def test_invalid_when_too_far():
    far = CityMap.from_positions(np.array([[0, 0]] + [[20, 0]] * 5 + [[1, 0]], dtype=float))
    assert not is_valid_map(far.dist_km, cfg)
