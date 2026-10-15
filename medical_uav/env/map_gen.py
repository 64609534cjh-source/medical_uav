"""Random city map generation: node layout first, then small no-fly zones and the routes.

M and the hospitals are placed at random but well apart, so they spread over the city; C2 is
placed where it most shortens the hospitals' trips to a charger. No-fly zones are added last, at
most one per district, and each must lengthen some route.
"""
from dataclasses import dataclass, field

import numpy as np

from medical_uav.env.astar import astar, octile_distance, path_cost_cells
from medical_uav.env.config import (C2, CHARGERS, EPS, FULL_BATTERY, HOSPITALS, M, N_NODES,
                                    EnvConfig)
from medical_uav.env.physics import flight_energy

MAX_PLACEMENT_TRIES = 500  # random cells tried per node or zone before the attempt fails
BORDER_MARGIN = 1  # free cells between every node and the edge of the map
FACILITY_MARGIN = 2  # free cells between M or a hospital and the edge of the map
DISTRICT_ROWS, DISTRICT_COLS = 3, 2  # districts the city is split into; at most one zone each
ZONE_CLEARANCE = 1  # free cells around every zone: to nodes, to other zones and to the edge
MIN_ZONE_GAP = 4  # cells between the centres of any two zones
MIN_C2_BENEFICIARIES = 2  # hospitals C2 must bring at least one cell closer to a charger
ZONE_TRIES_PER_LAYOUT = 20  # zone draws tried on one node layout before a new layout is drawn

@dataclass
class CityMap:
    """A city: node positions, routes between nodes and no-fly zones."""

    node_xy: np.ndarray  # node positions in km, shape (N_NODES, 2)
    dist_km: np.ndarray  # route length between each pair of nodes
    paths: dict  # route polyline in km for each node pair (i, j), used for drawing
    blocked: np.ndarray | None = None  # True for grid cells inside a no-fly zone
    zones: list = field(default_factory=list)  # (row, col, height, width) of each zone
    seed: int = -1  # seed the map was generated from
    tries: int = 0  # node layouts drawn, including the one kept


def is_valid_map(dist_km: np.ndarray, cfg: EnvConfig) -> bool:
    """Return True if a fully loaded drone can fly from any charger to any node i, then to any
    node j, then to the nearest charger, without using the reserve."""
    load = cfg.max_weight
    for c in CHARGERS:
        for i in range(N_NODES):
            for j in range(N_NODES):
                back = min(flight_energy(dist_km[j, k], load, cfg) for k in CHARGERS)
                used = (flight_energy(dist_km[c, i], load, cfg)
                        + flight_energy(dist_km[i, j], load, cfg) + back)
                if FULL_BATTERY - used < cfg.reserve:
                    return False
    return True


def _cell_center_km(cell, cfg: EnvConfig) -> np.ndarray:
    row, col = cell
    return np.array([(col + 0.5) * cfg.cell_km, (row + 0.5) * cfg.cell_km])


def _too_close(cell, others, cfg: EnvConfig) -> bool:
    xy = _cell_center_km(cell, cfg)
    return any(np.linalg.norm(xy - _cell_center_km(o, cfg)) < cfg.min_node_spacing_km - EPS
               for o in others)


def _edges(n_parts: int, cfg: EnvConfig) -> np.ndarray:
    """Split the grid into n_parts nearly equal bands and return their boundaries in cells."""
    return np.linspace(0, cfg.grid_size, n_parts + 1).round().astype(int)


def _district_of(cell, cfg: EnvConfig) -> tuple[int, int]:
    """Return the (row, col) of the district that contains a cell or a point in cell units."""
    rows, cols = _edges(DISTRICT_ROWS, cfg), _edges(DISTRICT_COLS, cfg)
    return (int(np.searchsorted(rows, cell[0], side="right")) - 1,
            int(np.searchsorted(cols, cell[1], side="right")) - 1)


def _place_nodes(rng, cfg: EnvConfig):
    """Place M and the hospitals at random cells, FACILITY_MARGIN cells from the edge and at least
    facility_spacing_km apart. Returns None if a node does not fit."""
    lo, hi = FACILITY_MARGIN, cfg.grid_size - FACILITY_MARGIN
    cells = []  # (row, col) of M, H1..H5 in node-index order
    for _ in range(len(HOSPITALS) + 1):
        for _ in range(MAX_PLACEMENT_TRIES):
            cell = (int(rng.integers(lo, hi)), int(rng.integers(lo, hi)))
            xy = _cell_center_km(cell, cfg)
            if all(np.linalg.norm(xy - _cell_center_km(other, cfg))
                   >= cfg.facility_spacing_km - EPS for other in cells):
                cells.append(cell)
                break
        else:
            return None
    return cells


def _place_c2(rng, cells, cfg: EnvConfig):
    """Choose the C2 cell with the smallest summed hospital-to-nearest-charger distance, ignoring
    zones (ties: smallest worst distance, then random). Return None if no cell qualifies."""
    to_m = np.array([octile_distance(cells[h], cells[M]) for h in HOSPITALS])
    m_xy = _cell_center_km(cells[M], cfg)
    lo, hi = BORDER_MARGIN, cfg.grid_size - BORDER_MARGIN
    min_from_m = cfg.c2_min_dist_from_m_km - EPS
    best_key, best = None, []
    for r in range(lo, hi):
        for c in range(lo, hi):
            near_m = np.linalg.norm(_cell_center_km((r, c), cfg) - m_xy) < min_from_m
            if near_m or _too_close((r, c), cells, cfg):
                continue
            to_c2 = np.array([octile_distance(cells[h], (r, c)) for h in HOSPITALS])
            if np.sum(to_m - to_c2 >= 1 - EPS) < MIN_C2_BENEFICIARIES:
                continue
            nearest = np.minimum(to_m, to_c2)  # each hospital's distance to its nearest charger
            key = (round(nearest.sum(), 9), round(nearest.max(), 9))
            if best_key is None or key < best_key:
                best_key, best = key, [(r, c)]
            elif key == best_key:
                best.append((r, c))
    if not best:
        return None
    return best[int(rng.integers(len(best)))]


def _place_zones(rng, cells, cfg: EnvConfig):
    """Add no-fly zones one at a time at random places; keep a zone only if it follows the zone
    rules and lengthens some route. Return (blocked, zones), or None if a zone does not fit."""
    g, k = cfg.grid_size, ZONE_CLEARANCE
    blocked = np.zeros((g, g), dtype=bool)
    nodes = np.zeros((g, g), dtype=bool)
    for cell in cells:
        nodes[cell] = True
    zones, centres, districts = [], [], set()
    low, high = cfg.zone_size_range
    for _ in range(int(rng.integers(cfg.n_zones_range[0], cfg.n_zones_range[1] + 1))):
        dist, paths = _route_all_pairs(blocked, cells, cfg)
        on_route = _route_cells(paths, cfg)
        for _ in range(MAX_PLACEMENT_TRIES):
            height, width = int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))
            if g - 2 * k < max(height, width):
                return None
            row = int(rng.integers(k, g - k - height + 1))
            col = int(rng.integers(k, g - k - width + 1))
            centre = (row + height / 2, col + width / 2)  # in cell units from the top-left corner
            district = _district_of(centre, cfg)
            around = (slice(row - k, row + height + k), slice(col - k, col + width + k))
            if (district in districts or nodes[around].any() or blocked[around].any()
                    or any(np.hypot(centre[0] - r, centre[1] - c) < MIN_ZONE_GAP - EPS
                           for r, c in centres)):
                continue
            # a zone away from every route cannot lengthen one, so skip routing it
            if not on_route[around].any():
                continue
            trial = blocked.copy()
            trial[row:row + height, col:col + width] = True
            routed = _route_all_pairs(trial, cells, cfg)
            if routed is None or not (routed[0] - dist >= cfg.cell_km - EPS).any():
                continue
            blocked = trial
            zones.append((row, col, height, width))
            centres.append(centre)
            districts.add(district)
            break
        else:
            return None
    return blocked, zones


def _route_cells(paths: dict, cfg: EnvConfig) -> np.ndarray:
    """Return a grid that is True for every cell some route passes through."""
    on_route = np.zeros((cfg.grid_size, cfg.grid_size), dtype=bool)
    for points in paths.values():
        for x, y in points:
            on_route[int(round(y / cfg.cell_km - 0.5)), int(round(x / cfg.cell_km - 0.5))] = True
    return on_route


def _every_zone_adds_a_detour(blocked: np.ndarray, zones, cells, dist_km: np.ndarray,
                              cfg: EnvConfig) -> bool:
    """Return True if removing any one zone would shorten some route by a cell or more."""
    for row, col, height, width in zones:
        without = blocked.copy()
        without[row:row + height, col:col + width] = False
        shorter, _ = _route_all_pairs(without, cells, cfg)
        if not (dist_km - shorter >= cfg.cell_km - EPS).any():
            return False
    return True


def _c2_beneficiaries(dist_km: np.ndarray, cfg: EnvConfig) -> int:
    """Count the hospitals that C2 brings at least one cell closer to a charger."""
    return sum(dist_km[h, M] - dist_km[h, C2] >= cfg.cell_km - EPS for h in HOSPITALS)


def _route_all_pairs(blocked: np.ndarray, cells, cfg: EnvConfig):
    """Route every pair of nodes with A*, or return None if a pair is unreachable."""
    dist = np.zeros((N_NODES, N_NODES))
    paths = {}
    for i in range(N_NODES):
        paths[(i, i)] = np.array([_cell_center_km(cells[i], cfg)])
        for j in range(i + 1, N_NODES):
            cell_path = astar(blocked, cells[i], cells[j])
            if cell_path is None:
                return None
            dist[i, j] = dist[j, i] = path_cost_cells(cell_path) * cfg.cell_km
            points = np.array([_cell_center_km(cell, cfg) for cell in cell_path])
            paths[(i, j)] = points
            paths[(j, i)] = points[::-1].copy()
    return dist, paths


def generate_map(seed: int, cfg: EnvConfig) -> CityMap:
    """Generate a valid map; the same seed always gives the same map. Zones are redrawn first,
    then the node layout if no zone draw suits it."""
    layout_rng, zone_rng = (np.random.default_rng(s)
                            for s in np.random.SeedSequence(seed).spawn(2))
    for layout_try in range(1, cfg.max_map_tries + 1):
        cells = _place_nodes(layout_rng, cfg)
        c2 = None if cells is None else _place_c2(layout_rng, cells, cfg)
        if c2 is None:
            continue
        cells.append(c2)
        free = np.array([[octile_distance(a, b) * cfg.cell_km for b in cells] for a in cells])
        if not is_valid_map(free, cfg):
            continue

        for _ in range(ZONE_TRIES_PER_LAYOUT):
            placed = _place_zones(zone_rng, cells, cfg)
            if placed is None:
                continue
            blocked, zones = placed
            dist, paths = _route_all_pairs(blocked, cells, cfg)
            if not _every_zone_adds_a_detour(blocked, zones, cells, dist, cfg):
                continue
            if _c2_beneficiaries(dist, cfg) < MIN_C2_BENEFICIARIES or not is_valid_map(dist, cfg):
                continue
            node_xy = np.array([_cell_center_km(cell, cfg) for cell in cells])
            return CityMap(node_xy=node_xy, dist_km=dist, paths=paths, blocked=blocked,
                           zones=zones, seed=seed, tries=layout_try)
    raise RuntimeError(f"no valid map for seed {seed} after {cfg.max_map_tries} node layouts")
