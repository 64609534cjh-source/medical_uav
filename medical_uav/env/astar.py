"""A* path finding on the occupancy grid (spec §4.1).

The drone flies on a 2D grid. Cells inside no-fly zones are `blocked`.
Moves go to the 8 neighbouring cells: straight moves cost 1, diagonal moves
cost sqrt(2). A diagonal move is only allowed if both straight cells it passes
are free, so a path never cuts the corner of a no-fly zone.
"""
import heapq
import math

import numpy as np

Cell = tuple[int, int]  # (row, col)

# The 8 moves: (row change, col change)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def octile_distance(a: Cell, b: Cell) -> float:
    """Exact grid distance between two cells when there are no obstacles.

    Used as the A* heuristic: it never overestimates the true cost.
    """
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)


def astar(blocked: np.ndarray, start: Cell, goal: Cell) -> list[Cell] | None:
    """Shortest path from `start` to `goal` on the grid.

    Returns the list of cells including start and goal, or None if the goal
    cannot be reached.
    """
    n_rows, n_cols = blocked.shape

    # Priority queue of (estimated total cost, cost so far, cell).
    open_heap = [(octile_distance(start, goal), 0.0, start)]
    best_cost = {start: 0.0}  # cheapest known cost to reach each cell
    came_from = {}  # cell -> previous cell on the best path

    while open_heap:
        _, cost, cell = heapq.heappop(open_heap)
        if cell == goal:
            break
        if cost > best_cost[cell]:
            continue  # an outdated queue entry; a cheaper one was processed

        for dr, dc in MOVES:
            r, c = cell[0] + dr, cell[1] + dc
            if not (0 <= r < n_rows and 0 <= c < n_cols) or blocked[r, c]:
                continue
            is_diagonal = dr != 0 and dc != 0
            if is_diagonal and (blocked[cell[0] + dr, cell[1]] or blocked[cell[0], cell[1] + dc]):
                continue  # would cut the corner of a blocked cell
            new_cost = cost + (math.sqrt(2) if is_diagonal else 1.0)
            if new_cost < best_cost.get((r, c), math.inf):
                best_cost[(r, c)] = new_cost
                came_from[(r, c)] = cell
                heapq.heappush(open_heap, (new_cost + octile_distance((r, c), goal), new_cost, (r, c)))

    if goal not in best_cost:
        return None

    # Walk back from the goal to rebuild the path.
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def path_cost_cells(path: list[Cell]) -> float:
    """Length of a cell path in cell units (1 per straight step, sqrt(2) per diagonal)."""
    total = 0.0
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        total += math.sqrt(2) if (r0 != r1 and c0 != c1) else 1.0
    return total
