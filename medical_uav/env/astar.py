"""8-connected A* on the occupancy grid, no corner cutting."""
import heapq
import math

import numpy as np

Cell = tuple[int, int]  # (row, col)

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def octile_distance(a: Cell, b: Cell) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)


def astar(blocked: np.ndarray, start: Cell, goal: Cell) -> list[Cell] | None:
    """Shortest cell path from start to goal (inclusive), or None if unreachable."""
    n_rows, n_cols = blocked.shape
    open_heap = [(octile_distance(start, goal), 0.0, start)]
    best_cost = {start: 0.0}
    came_from = {}

    while open_heap:
        _, cost, cell = heapq.heappop(open_heap)
        if cell == goal:
            break
        if cost > best_cost[cell]:
            continue  # stale entry

        for dr, dc in MOVES:
            r, c = cell[0] + dr, cell[1] + dc
            if not (0 <= r < n_rows and 0 <= c < n_cols) or blocked[r, c]:
                continue
            diagonal = dr != 0 and dc != 0
            if diagonal and (blocked[cell[0] + dr, cell[1]] or blocked[cell[0], cell[1] + dc]):
                continue  # don't cut corners
            new_cost = cost + (math.sqrt(2) if diagonal else 1.0)
            if new_cost < best_cost.get((r, c), math.inf):
                best_cost[(r, c)] = new_cost
                came_from[(r, c)] = cell
                priority = new_cost + octile_distance((r, c), goal)
                heapq.heappush(open_heap, (priority, new_cost, (r, c)))

    if goal not in best_cost:
        return None

    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def path_cost_cells(path: list[Cell]) -> float:
    total = 0.0
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        total += math.sqrt(2) if (r0 != r1 and c0 != c1) else 1.0
    return total
