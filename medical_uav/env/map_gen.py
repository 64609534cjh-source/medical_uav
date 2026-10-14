"""Random city map generation: node layout first, then small no-fly zones and the routes.

The city is split into 3 x 2 districts. M goes into a middle-row district and one hospital into
each of the others; C2 is placed where it most shortens the hospitals' trips to a charger. The
no-fly zones are added last, so they do not shape the layout.
"""
from dataclasses import dataclass, field

import numpy as np

from medical_uav.env.astar import astar, octile_distance, path_cost_cells
from medical_uav.env.config import (C2, CHARGERS, EPS, FULL_BATTERY, HOSPITALS, M, N_NODES,
                                    EnvConfig)
from medical_uav.env.physics import flight_energy

MAX_PLACEMENT_TRIES = 500  # random cells tried per node or zone before the attempt fails
BORDER_MARGIN = 1  # free cells between every node and the edge of the map
DISTRICT_ROWS, DISTRICT_COLS = 3, 2  # the city is split into this many districts
ZONE_CLEARANCE = 1  # free cells around every zone: to nodes, to other zones and to the edge
MIN_C2_BENEFICIARIES = 2  # hospitals C2 must bring at least one cell closer to a charger

# one district for M and one for each hospital
assert DISTRICT_ROWS * DISTRICT_COLS == len(HOSPITALS) + 1


@dataclass
class CityMap:
    """A city: node positions, routes between nodes and no-fly zones."""

    node_xy: np.ndarray  # node positions in km, shape (N_NODES, 2)
    dist_km: np.ndarray  # route length between each pair of nodes
    paths: dict  # route polyline in km for each node pair (i, j), used for drawing
    blocked: np.ndarray | None = None  # True for grid cells inside a no-fly zone
    zones: list = field(default_factory=list)  # (row, col, height, width) of each zone
    seed: int = -1  # seed the map was generated from
    tries: int = 0  # attempts generation needed

    @classmethod
    def from_positions(cls, node_xy: np.ndarray) -> "CityMap":
        """Build an obstacle-free map with straight routes (used in tests)."""
        node_xy = np.asarray(node_xy, dtype=float)
        dist = np.linalg.norm(node_xy[:, None, :] - node_xy[None, :, :], axis=2)
        paths = {(i, j): np.array([node_xy[i], node_xy[j]])
                 for i in range(N_NODES) for j in range(N_NODES)}
        return cls(node_xy=node_xy, dist_km=dist, paths=paths)


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
    """Split the cells inside the border margin into n_parts nearly equal bands."""
    edges = np.linspace(BORDER_MARGIN, cfg.grid_size - BORDER_MARGIN, n_parts + 1)
    return edges.round().astype(int)


def _district_of(cell, cfg: EnvConfig) -> tuple[int, int]:
    """Return the (row, col) of the district that contains cell."""
    rows, cols = _edges(DISTRICT_ROWS, cfg), _edges(DISTRICT_COLS, cfg)
    return (int(np.searchsorted(rows, cell[0], side="right")) - 1,
            int(np.searchsorted(cols, cell[1], side="right")) - 1)


def _place_nodes(rng, cfg: EnvConfig):
    """Place M in a middle-row district and one hospital in each other district, or return None."""
    rows, cols = _edges(DISTRICT_ROWS, cfg), _edges(DISTRICT_COLS, cfg)
    m_district = (DISTRICT_ROWS // 2, int(rng.integers(DISTRICT_COLS)))
    others = [(r, c) for r in range(DISTRICT_ROWS) for c in range(DISTRICT_COLS)
              if (r, c) != m_district]
    rng.shuffle(others)  # which hospital goes to which district
    cells = []  # (row, col) of M, H1..H5 in node-index order
    for dr, dc in [m_district] + others:
        for _ in range(MAX_PLACEMENT_TRIES):
            cell = (int(rng.integers(rows[dr], rows[dr + 1])),
                    int(rng.integers(cols[dc], cols[dc + 1])))
            if not _too_close(cell, cells, cfg):
                cells.append(cell)
                break
        else:
            return None
    return cells


def _place_c2(rng, cells, cfg: EnvConfig):
    """Choose the cell for C2 that minimises the summed hospital-to-nearest-charger distance.

    Distances ignore no-fly zones, which are added later. C2 must be at least
    c2_min_dist_from_m_km from M and bring MIN_C2_BENEFICIARIES hospitals a cell or more closer
    to a charger. Ties are broken by the longest such distance, then at random. Returns None
    if no cell qualifies.
    """
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
    """Add rectangular no-fly zones that keep ZONE_CLEARANCE free cells to the nodes, to each other
    and to the edge. Returns (blocked, zones), or None if a zone does not fit."""
    g, k = cfg.grid_size, ZONE_CLEARANCE
    blocked = np.zeros((g, g), dtype=bool)
    nodes = np.zeros((g, g), dtype=bool)
    for cell in cells:
        nodes[cell] = True
    zones = []
    low, high = cfg.zone_size_range
    for _ in range(int(rng.integers(cfg.n_zones_range[0], cfg.n_zones_range[1] + 1))):
        for _ in range(MAX_PLACEMENT_TRIES):
            height, width = int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))
            if g - 2 * k < max(height, width):
                return None
            row = int(rng.integers(k, g - k - height + 1))
            col = int(rng.integers(k, g - k - width + 1))
            around = (slice(row - k, row + height + k), slice(col - k, col + width + k))
            if not nodes[around].any() and not blocked[around].any():
                blocked[row:row + height, col:col + width] = True
                zones.append((row, col, height, width))
                break
        else:
            return None
    return blocked, zones


def _zones_add_a_detour(dist_km: np.ndarray, cells, cfg: EnvConfig) -> bool:
    """Return True if the zones lengthen a route between two facilities by a cell or more."""
    facilities = (M,) + HOSPITALS
    return any(dist_km[a, b] - cfg.cell_km * octile_distance(cells[a], cells[b])
               >= cfg.cell_km - EPS for i, a in enumerate(facilities) for b in facilities[i + 1:])


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
    """Generate a valid map; the same seed always produces the same map.

    The node layout is drawn first and kept; only the no-fly zones are redrawn until at least
    one route has to go around them and the other rules still hold.
    """
    layout_rng, zone_rng = (np.random.default_rng(s)
                            for s in np.random.SeedSequence(seed).spawn(2))
    for layout_try in range(1, cfg.max_map_tries + 1):
        cells = _place_nodes(layout_rng, cfg)
        c2 = None if cells is None else _place_c2(layout_rng, cells, cfg)
        if c2 is None:
            continue
        cells.append(c2)
        free = np.array([[octile_distance(a, b) * cfg.cell_km for b in cells] for a in cells])
        if is_valid_map(free, cfg):
            break
    else:
        raise RuntimeError(f"no valid node layout for seed {seed} after {cfg.max_map_tries} tries")

    for zone_try in range(1, cfg.max_map_tries + 1):
        placed = _place_zones(zone_rng, cells, cfg)
        if placed is None:
            continue
        blocked, zones = placed
        routed = _route_all_pairs(blocked, cells, cfg)
        if routed is None:
            continue
        dist, paths = routed
        if zones and not _zones_add_a_detour(dist, cells, cfg):
            continue
        if _c2_beneficiaries(dist, cfg) < MIN_C2_BENEFICIARIES or not is_valid_map(dist, cfg):
            continue
        node_xy = np.array([_cell_center_km(cell, cfg) for cell in cells])
        return CityMap(node_xy=node_xy, dist_km=dist, paths=paths, blocked=blocked,
                       zones=zones, seed=seed, tries=layout_try + zone_try - 1)
    raise RuntimeError(f"no valid no-fly zones for seed {seed} after {cfg.max_map_tries} tries")
