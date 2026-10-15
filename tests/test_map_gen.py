"""Tests for random city maps."""
import numpy as np

from medical_uav.env.config import C2, M, EnvConfig
from medical_uav.env.map_gen import generate_map, is_valid_map
from tests.conftest import straight_map

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
    far = straight_map([[0, 0]] + [[20, 0]] * 5 + [[1, 0]])
    assert not is_valid_map(far.dist_km, cfg)


def _cells(m, cfg):
    return [(int(round(y / cfg.cell_km - 0.5)), int(round(x / cfg.cell_km - 0.5)))
            for x, y in m.node_xy]


def test_nodes_keep_away_from_the_border():
    from medical_uav.env.map_gen import BORDER_MARGIN, FACILITY_MARGIN

    for seed in range(20):
        for i, (r, c) in enumerate(_cells(generate_map(seed, cfg), cfg)):
            margin = BORDER_MARGIN if i == C2 else FACILITY_MARGIN
            assert margin <= r < cfg.grid_size - margin
            assert margin <= c < cfg.grid_size - margin


def test_facilities_are_spread_out():
    for seed in range(20):
        xy = generate_map(seed, cfg).node_xy[:C2]
        for i in range(len(xy)):
            for j in range(i + 1, len(xy)):
                assert np.linalg.norm(xy[i] - xy[j]) >= cfg.facility_spacing_km - 1e-9


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


def test_every_zone_lengthens_some_route():
    from medical_uav.env.map_gen import _route_all_pairs

    for seed in range(10):
        m = generate_map(seed, cfg)
        cells = _cells(m, cfg)
        for (r, c, h, w) in m.zones:
            without = m.blocked.copy()
            without[r:r + h, c:c + w] = False
            dist, _ = _route_all_pairs(without, cells, cfg)
            assert (m.dist_km - dist >= cfg.cell_km - 1e-9).any()


def test_zones_lie_in_different_districts_and_apart():
    from medical_uav.env.map_gen import MIN_ZONE_GAP, _district_of

    for seed in range(20):
        zones = generate_map(seed, cfg).zones
        centres = [(r + h / 2, c + w / 2) for r, c, h, w in zones]
        districts = [_district_of(centre, cfg) for centre in centres]
        assert len(set(districts)) == len(zones)
        for i, a in enumerate(centres):
            for b in centres[i + 1:]:
                assert np.hypot(a[0] - b[0], a[1] - b[1]) >= MIN_ZONE_GAP - 1e-9


def test_maps_without_zones_still_generate():
    m = generate_map(0, EnvConfig(n_zones_range=(0, 0)))
    assert not m.blocked.any()
