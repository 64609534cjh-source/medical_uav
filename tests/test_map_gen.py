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


def test_one_facility_per_district():
    from medical_uav.env.map_gen import _district_of

    for seed in range(20):
        cells = _cells(generate_map(seed, cfg), cfg)
        districts = [_district_of(cells[i], cfg) for i in range(C2)]
        assert len(set(districts)) == 6


def test_m_keeps_away_from_the_edge_and_moves_between_districts():
    from medical_uav.env.map_gen import _district_of

    m_districts = set()
    for seed in range(20):
        m = generate_map(seed, cfg)
        x, y = m.node_xy[M]
        assert min(x, y, 4.0 - x, 4.0 - y) >= cfg.m_min_dist_from_edge_km - 1e-9
        m_districts.add(_district_of(_cells(m, cfg)[M], cfg))
    assert len({r for r, _ in m_districts}) == 3 and len({c for _, c in m_districts}) == 2


def test_c2_is_far_from_m_and_serves_two_hospitals():
    from medical_uav.env.config import HOSPITALS

    for seed in range(20):
        m = generate_map(seed, cfg)
        assert np.linalg.norm(m.node_xy[C2] - m.node_xy[M]) >= cfg.c2_min_dist_from_m_km - 1e-9
        saved = [m.dist_km[h, M] - m.dist_km[h, C2] for h in HOSPITALS]
        assert sum(s >= cfg.cell_km - 1e-9 for s in saved) >= 2


def test_c2_minimises_the_summed_hospital_to_charger_distance():
    from medical_uav.env.astar import octile_distance
    from medical_uav.env.config import HOSPITALS
    from medical_uav.env.map_gen import BORDER_MARGIN

    def total(cells, c2):
        return sum(min(octile_distance(cells[h], cells[M]), octile_distance(cells[h], c2))
                   for h in HOSPITALS)

    for seed in range(5):
        m = generate_map(seed, cfg)
        cells = _cells(m, cfg)
        chosen = total(cells, cells[C2])
        lo, hi = BORDER_MARGIN, cfg.grid_size - BORDER_MARGIN
        for r in range(lo, hi):
            for c in range(lo, hi):
                xy = np.array([(c + 0.5) * cfg.cell_km, (r + 0.5) * cfg.cell_km])
                spaced = all(np.linalg.norm(xy - m.node_xy[i]) >= cfg.min_node_spacing_km - 1e-9
                             for i in range(C2))
                far = np.linalg.norm(xy - m.node_xy[M]) >= cfg.c2_min_dist_from_m_km - 1e-9
                helps = sum(octile_distance(cells[h], cells[M]) - octile_distance(cells[h], (r, c))
                            >= 1 - 1e-9 for h in HOSPITALS)
                if spaced and far and helps >= 2:
                    assert chosen <= total(cells, (r, c)) + 1e-9


def test_zones_keep_clear_of_nodes_each_other_and_the_border():
    from medical_uav.env.map_gen import ZONE_CLEARANCE

    for seed in range(20):
        m = generate_map(seed, cfg)
        g, k = cfg.grid_size, ZONE_CLEARANCE
        for (r, c, h, w) in m.zones:
            assert r >= k and c >= k and r + h <= g - k and c + w <= g - k
            for (nr, nc) in _cells(m, cfg):
                assert not (r - k <= nr < r + h + k and c - k <= nc < c + w + k)
        for i, (r1, c1, h1, w1) in enumerate(m.zones):
            for (r2, c2, h2, w2) in m.zones[i + 1:]:
                apart_rows = r1 + h1 + k <= r2 or r2 + h2 + k <= r1
                apart_cols = c1 + w1 + k <= c2 or c2 + w2 + k <= c1
                assert apart_rows or apart_cols


def test_zones_lengthen_some_route_between_facilities():
    from medical_uav.env.astar import octile_distance
    from medical_uav.env.config import HOSPITALS

    facilities = (M,) + HOSPITALS
    for seed in range(20):
        m = generate_map(seed, cfg)
        cells = _cells(m, cfg)
        assert any(m.dist_km[a, b] - cfg.cell_km * octile_distance(cells[a], cells[b])
                   >= cfg.cell_km - 1e-9 for a in facilities for b in facilities if a < b)


def test_maps_without_zones_still_generate():
    m = generate_map(0, EnvConfig(n_zones_range=(0, 0)))
    assert not m.blocked.any()


def test_layout_is_redrawn_when_no_zone_draw_suits_it():
    m = generate_map(79, cfg)  # its first layout leaves almost no route for a zone to block
    assert m.tries > 1
