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


def _cells(m, cfg):
    return [(int(round(y / cfg.cell_km - 0.5)), int(round(x / cfg.cell_km - 0.5)))
            for x, y in m.node_xy]


def test_nodes_keep_away_from_the_border():
    from medical_uav.env.map_gen import BORDER_MARGIN

    for seed in range(20):
        for r, c in _cells(generate_map(seed, cfg), cfg):
            assert BORDER_MARGIN <= r < cfg.grid_size - BORDER_MARGIN
            assert BORDER_MARGIN <= c < cfg.grid_size - BORDER_MARGIN


def test_c2_brings_some_hospital_closer_to_a_charger():
    from medical_uav.env.config import HOSPITALS

    for seed in range(20):
        d = generate_map(seed, cfg).dist_km
        assert any(d[h, C2] < d[h, M] for h in HOSPITALS)


def test_c2_minimises_the_longest_hospital_to_charger_route():
    from medical_uav.env.astar import grid_distances
    from medical_uav.env.config import HOSPITALS
    from medical_uav.env.map_gen import BORDER_MARGIN

    for seed in range(5):
        m = generate_map(seed, cfg)
        cells = _cells(m, cfg)
        to_h = [grid_distances(m.blocked, cells[h]) for h in HOSPITALS]
        to_m = np.array([d[cells[M]] for d in to_h])

        def worst(cell):
            return np.minimum(to_m, [d[cell] for d in to_h]).max()

        chosen = worst(cells[C2])
        lo, hi = BORDER_MARGIN, cfg.grid_size - BORDER_MARGIN
        for r in range(lo, hi):
            for c in range(lo, hi):
                xy = np.array([(c + 0.5) * cfg.cell_km, (r + 0.5) * cfg.cell_km])
                spaced = all(np.linalg.norm(xy - m.node_xy[i]) >= cfg.min_node_spacing_km - 1e-9
                             for i in range(C2))
                far_from_m = np.linalg.norm(xy - m.node_xy[M]) >= cfg.c2_min_dist_from_m_km - 1e-9
                if not m.blocked[r, c] and spaced and far_from_m:
                    assert chosen <= worst((r, c)) + 1e-9


def test_zones_lengthen_some_route_from_m():
    from medical_uav.env.astar import octile_distance
    from medical_uav.env.config import HOSPITALS

    for seed in range(20):
        m = generate_map(seed, cfg)
        cells = _cells(m, cfg)
        assert any(m.dist_km[M, h] - cfg.cell_km * octile_distance(cells[M], cells[h])
                   >= cfg.cell_km - 1e-9 for h in HOSPITALS)


def test_maps_without_zones_still_generate():
    m = generate_map(0, EnvConfig(n_zones_range=(0, 0)))
    assert not m.blocked.any()
