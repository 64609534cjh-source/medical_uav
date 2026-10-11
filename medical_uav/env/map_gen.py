"""Random city map: grid, no-fly zones, 7 nodes and A* routes (spec §4.1).

A map is generated once from a seed. The environment only needs the 7x7
matrix of flight distances between nodes; the cell paths are kept so that the
renderer can draw the real routes around the no-fly zones.
"""
from dataclasses import dataclass, field

import numpy as np

from medical_uav.env.astar import astar, path_cost_cells
from medical_uav.env.config import C2, CHARGERS, M, N_NODES, EnvConfig
from medical_uav.env.physics import flight_energy

MAX_PLACEMENT_TRIES = 500  # random cells tried per attempt before giving up on node placement


@dataclass
class CityMap:
    """Everything the simulator and the renderer need to know about a map."""

    node_xy: np.ndarray  # (7, 2) node positions in km, x to the right, y downwards
    dist_km: np.ndarray  # (7, 7) flight distance along the A* route
    paths: dict  # (i, j) -> (k, 2) array of km points along the route from node i to node j
    blocked: np.ndarray | None = None  # (G, G) occupancy grid, True inside no-fly zones
    zones: list = field(default_factory=list)  # no-fly rectangles as (row, col, height, width)
    seed: int = -1
    tries: int = 0  # number of generation attempts that were needed

    @classmethod
    def from_positions(cls, node_xy: np.ndarray) -> "CityMap":
        """A map without obstacles: straight-line routes. Used by tests."""
        node_xy = np.asarray(node_xy, dtype=float)
        dist = np.linalg.norm(node_xy[:, None, :] - node_xy[None, :, :], axis=2)
        paths = {(i, j): np.array([node_xy[i], node_xy[j]]) for i in range(N_NODES) for j in range(N_NODES)}
        return cls(node_xy=node_xy, dist_km=dist, paths=paths)


def is_valid_map(dist_km: np.ndarray, cfg: EnvConfig) -> bool:
    """Check that every trip is possible with a full battery and full load (§4.1).

    From 100 % at either charger, flying with 2 kg to any node i, then to any
    node j, then to the nearest charger must leave at least the reserve.
    """
    full_load = cfg.max_weight
    for c in CHARGERS:
        for i in range(N_NODES):
            for j in range(N_NODES):
                to_charger = min(flight_energy(dist_km[j, k], full_load, cfg) for k in CHARGERS)
                used = (flight_energy(dist_km[c, i], full_load, cfg)
                        + flight_energy(dist_km[i, j], full_load, cfg)
                        + to_charger)
                if 100.0 - used < cfg.reserve:
                    return False
    return True


def _cell_center_km(cell: tuple[int, int], cfg: EnvConfig) -> np.ndarray:
    """Centre of a grid cell in km as (x, y) = (column, row)."""
    row, col = cell
    return np.array([(col + 0.5) * cfg.cell_km, (row + 0.5) * cfg.cell_km])


def _sample_zones(rng: np.random.Generator, cfg: EnvConfig) -> tuple[np.ndarray, list]:
    """Place a few random rectangular no-fly zones on the grid."""
    g = cfg.grid_size
    blocked = np.zeros((g, g), dtype=bool)
    zones = []
    n_zones = int(rng.integers(cfg.n_zones_range[0], cfg.n_zones_range[1] + 1))
    for _ in range(n_zones):
        height = int(rng.integers(2, 6))
        width = int(rng.integers(2, 6))
        row = int(rng.integers(0, g - height + 1))
        col = int(rng.integers(0, g - width + 1))
        blocked[row:row + height, col:col + width] = True
        zones.append((row, col, height, width))
    return blocked, zones


def _place_nodes(rng: np.random.Generator, blocked: np.ndarray, cfg: EnvConfig) -> list | None:
    """Pick free cells for M, H1..H5 and C2 respecting the spacing rules.

    Returns 7 cells in node order, or None if the rules could not be met.
    """
    g = cfg.grid_size
    cells = []
    for node in range(N_NODES):
        for _ in range(MAX_PLACEMENT_TRIES):
            cell = (int(rng.integers(0, g)), int(rng.integers(0, g)))
            if blocked[cell]:
                continue
            xy = _cell_center_km(cell, cfg)
            too_close = any(np.linalg.norm(xy - _cell_center_km(other, cfg)) < cfg.min_node_spacing_km - 1e-9
                            for other in cells)
            if too_close:
                continue
            if node == C2 and np.linalg.norm(xy - _cell_center_km(cells[M], cfg)) < cfg.c2_min_dist_from_m_km - 1e-9:
                continue
            cells.append(cell)
            break
        else:
            return None  # no acceptable cell found for this node
    return cells


def _route_all_pairs(blocked: np.ndarray, cells: list, cfg: EnvConfig) -> tuple[np.ndarray, dict] | None:
    """Run A* between every pair of nodes. Returns (distance matrix, paths) or None."""
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
    """Generate a valid random map. The same seed always gives the same map.

    Each attempt samples zones, places nodes and routes all pairs; failed
    attempts continue with the same random generator, so the result is still
    deterministic.
    """
    rng = np.random.default_rng(seed)
    for attempt in range(1, cfg.max_map_tries + 1):
        blocked, zones = _sample_zones(rng, cfg)
        cells = _place_nodes(rng, blocked, cfg)
        if cells is None:
            continue
        routed = _route_all_pairs(blocked, cells, cfg)
        if routed is None:
            continue
        dist, paths = routed
        if not is_valid_map(dist, cfg):
            continue
        node_xy = np.array([_cell_center_km(cell, cfg) for cell in cells])
        return CityMap(node_xy=node_xy, dist_km=dist, paths=paths, blocked=blocked,
                       zones=zones, seed=seed, tries=attempt)
    raise RuntimeError(f"no valid map found for seed {seed} in {cfg.max_map_tries} attempts")
