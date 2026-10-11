"""Random city map: no-fly zones, 7 nodes and A* routes between them."""
from dataclasses import dataclass, field

import numpy as np

from medical_uav.env.astar import astar, path_cost_cells
from medical_uav.env.config import C2, CHARGERS, M, N_NODES, EnvConfig
from medical_uav.env.physics import flight_energy

MAX_PLACEMENT_TRIES = 500


@dataclass
class CityMap:
    node_xy: np.ndarray  # (7, 2) km
    dist_km: np.ndarray  # (7, 7) A* route lengths
    paths: dict  # (i, j) -> (k, 2) km polyline, for rendering
    blocked: np.ndarray | None = None  # (G, G) occupancy grid
    zones: list = field(default_factory=list)  # (row, col, height, width)
    seed: int = -1
    tries: int = 0

    @classmethod
    def from_positions(cls, node_xy: np.ndarray) -> "CityMap":
        """Obstacle-free map with straight routes (used in tests)."""
        node_xy = np.asarray(node_xy, dtype=float)
        dist = np.linalg.norm(node_xy[:, None, :] - node_xy[None, :, :], axis=2)
        paths = {(i, j): np.array([node_xy[i], node_xy[j]])
                 for i in range(N_NODES) for j in range(N_NODES)}
        return cls(node_xy=node_xy, dist_km=dist, paths=paths)


def is_valid_map(dist_km: np.ndarray, cfg: EnvConfig) -> bool:
    """Charger -> i -> j -> nearest charger with a full load must keep the reserve."""
    load = cfg.max_weight
    for c in CHARGERS:
        for i in range(N_NODES):
            for j in range(N_NODES):
                back = min(flight_energy(dist_km[j, k], load, cfg) for k in CHARGERS)
                used = (flight_energy(dist_km[c, i], load, cfg)
                        + flight_energy(dist_km[i, j], load, cfg) + back)
                if 100.0 - used < cfg.reserve:
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
        height = int(rng.integers(2, 6))
        width = int(rng.integers(2, 6))
        row = int(rng.integers(0, g - height + 1))
        col = int(rng.integers(0, g - width + 1))
        blocked[row:row + height, col:col + width] = True
        zones.append((row, col, height, width))
    return blocked, zones


def _place_nodes(rng, blocked: np.ndarray, cfg: EnvConfig):
    """Free cells for M, H1..H5, C2 with the spacing rules, or None."""
    g = cfg.grid_size
    cells = []
    for node in range(N_NODES):
        for _ in range(MAX_PLACEMENT_TRIES):
            cell = (int(rng.integers(0, g)), int(rng.integers(0, g)))
            if blocked[cell]:
                continue
            xy = _cell_center_km(cell, cfg)
            if any(np.linalg.norm(xy - _cell_center_km(other, cfg)) < cfg.min_node_spacing_km - 1e-9
                   for other in cells):
                continue
            if node == C2:
                d_m = np.linalg.norm(xy - _cell_center_km(cells[M], cfg))
                if d_m < cfg.c2_min_dist_from_m_km - 1e-9:
                    continue
            cells.append(cell)
            break
        else:
            return None
    return cells


def _route_all_pairs(blocked: np.ndarray, cells, cfg: EnvConfig):
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
    """Same seed -> same map. Failed attempts retry with the same rng."""
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
    raise RuntimeError(f"no valid map for seed {seed} after {cfg.max_map_tries} attempts")
