"""Random city map generation: no-fly zones, node positions and routes between nodes."""
from dataclasses import dataclass, field

import numpy as np

from medical_uav.env.astar import astar, grid_distances, octile_distance, path_cost_cells
from medical_uav.env.config import (C2, CHARGERS, EPS, FULL_BATTERY, HOSPITALS, M, N_NODES,
                                    EnvConfig)
from medical_uav.env.physics import flight_energy

MAX_PLACEMENT_TRIES = 500  # random cells tried per node before the attempt fails
BORDER_MARGIN = 1  # free cells between every node and the edge of the map


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


def _sample_zones(rng, cfg: EnvConfig):
    g = cfg.grid_size
    blocked = np.zeros((g, g), dtype=bool)
    zones = []
    n_zones = int(rng.integers(cfg.n_zones_range[0], cfg.n_zones_range[1] + 1))
    for _ in range(n_zones):
        low, high = cfg.zone_size_range
        height = int(rng.integers(low, high + 1))
        width = int(rng.integers(low, high + 1))
        row = int(rng.integers(0, g - height + 1))
        col = int(rng.integers(0, g - width + 1))
        blocked[row:row + height, col:col + width] = True
        zones.append((row, col, height, width))
    return blocked, zones


def _too_close(cell, others, cfg: EnvConfig) -> bool:
    xy = _cell_center_km(cell, cfg)
    return any(np.linalg.norm(xy - _cell_center_km(o, cfg)) < cfg.min_node_spacing_km - EPS
               for o in others)


def _place_nodes(rng, blocked: np.ndarray, cfg: EnvConfig):
    """Choose random free cells for M and H1..H5 that keep the spacing rule, or return None."""
    lo, hi = BORDER_MARGIN, cfg.grid_size - BORDER_MARGIN
    cells = []  # (row, col) of each node, in node-index order
    for _ in range(C2):  # M, H1..H5 in node-index order; C2 is placed separately
        for _ in range(MAX_PLACEMENT_TRIES):
            cell = (int(rng.integers(lo, hi)), int(rng.integers(lo, hi)))
            if not blocked[cell] and not _too_close(cell, cells, cfg):
                cells.append(cell)
                break
        else:
            return None
    return cells


def _place_c2(rng, blocked: np.ndarray, cells, cfg: EnvConfig):
    """Choose the cell for C2 that makes the longest hospital-to-charger route shortest.

    Ties are broken by the total of those routes, then at random. C2 must bring at least one
    hospital closer to a charger. Returns None if no free cell qualifies.
    """
    to_hospital = [grid_distances(blocked, cells[h]) for h in HOSPITALS]
    to_m = np.array([d[cells[M]] for d in to_hospital])
    m_xy = _cell_center_km(cells[M], cfg)
    lo, hi = BORDER_MARGIN, cfg.grid_size - BORDER_MARGIN
    min_from_m = cfg.c2_min_dist_from_m_km - EPS
    best_key, best = None, []
    for r in range(lo, hi):
        for c in range(lo, hi):
            near_m = np.linalg.norm(_cell_center_km((r, c), cfg) - m_xy) < min_from_m
            if blocked[r, c] or near_m or _too_close((r, c), cells, cfg):
                continue
            to_c2 = np.array([d[r, c] for d in to_hospital])
            if not np.any(to_c2 < to_m - EPS):
                continue
            nearest = np.minimum(to_m, to_c2)  # each hospital's route to its nearest charger
            key = (round(nearest.max(), 9), round(nearest.sum(), 9))
            if best_key is None or key < best_key:
                best_key, best = key, [(r, c)]
            elif key == best_key:
                best.append((r, c))
    if not best:
        return None
    return best[int(rng.integers(len(best)))]


def _zones_add_a_detour(dist_km: np.ndarray, cells, cfg: EnvConfig) -> bool:
    """Return True if no-fly zones lengthen some route from M to a hospital by a cell or more."""
    return any(dist_km[M, h] - cfg.cell_km * octile_distance(cells[M], cells[h])
               >= cfg.cell_km - EPS for h in HOSPITALS)


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

    No-fly zones, M and the hospitals are placed at random; C2 is then placed where it best
    covers the hospitals (see _place_c2).
    """
    rng = np.random.default_rng(seed)
    for attempt in range(1, cfg.max_map_tries + 1):
        blocked, zones = _sample_zones(rng, cfg)
        cells = _place_nodes(rng, blocked, cfg)
        c2 = None if cells is None else _place_c2(rng, blocked, cells, cfg)
        if c2 is None:
            continue
        cells.append(c2)
        routed = _route_all_pairs(blocked, cells, cfg)
        if routed is None:
            continue
        dist, paths = routed
        if zones and not _zones_add_a_detour(dist, cells, cfg):
            continue
        if not is_valid_map(dist, cfg):
            continue
        node_xy = np.array([_cell_center_km(cell, cfg) for cell in cells])
        return CityMap(node_xy=node_xy, dist_km=dist, paths=paths, blocked=blocked,
                       zones=zones, seed=seed, tries=attempt)
    raise RuntimeError(f"no valid map for seed {seed} after {cfg.max_map_tries} attempts")
