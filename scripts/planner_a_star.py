import time
from typing import List, Tuple, Optional
import heapq
import numpy as np
from scipy.ndimage import binary_dilation

Coord = Tuple[int, int]


def inflate_obstacles(occ: np.ndarray, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return occ
    structure = np.ones((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
    return binary_dilation(occ.astype(bool), structure=structure)


def astar(
    occ: np.ndarray,
    cost: Optional[np.ndarray],
    start: Coord,
    goal: Coord,
    allow_diagonal: bool = True,
) -> Tuple[List[Coord], dict]:
    """
    Simple/classic A* on a 2D occupancy grid.
    - occ: boolean grid (True=obstacle)
    - cost: optional continuous cost grid [0,1] for path cost accumulation (may be None)
    - start, goal: (row, col)
    Returns (path, info) where path is a list of (r,c) from start..goal inclusive.
    """
    H, W = occ.shape
    if occ[start] or occ[goal]:
        return [], {"success": False, "reason": "start_or_goal_in_collision"}

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    cost_step = [1.0] * 4
    if allow_diagonal:
        moves += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        cost_step += [np.sqrt(2.0)] * 4

    def in_bounds(r, c):
        return 0 <= r < H and 0 <= c < W

    def heuristic(a: Coord, b: Coord) -> float:
        return np.hypot(a[0] - b[0], a[1] - b[1])

    g = {start: 0.0}
    parent = {start: None}
    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))

    t0 = time.perf_counter()
    while open_heap:
        f, g_now, cur = heapq.heappop(open_heap)
        if cur == goal:
            # reconstruct
            path = []
            node = cur
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            t_ms = (time.perf_counter() - t0) * 1000.0
            # compute metrics
            path_len = 0.0
            path_cost = 0.0
            for i in range(1, len(path)):
                dr = path[i][0] - path[i - 1][0]
                dc = path[i][1] - path[i - 1][1]
                path_len += np.hypot(dr, dc)
            if cost is not None and len(path) > 0:
                path_cost = float(np.sum(cost[tuple(np.array(path).T)]))
            return path, {
                "success": True,
                "time_ms": t_ms,
                "path_length_cells": path_len,
                "path_cost_sum": path_cost,
            }
        # expand
        r, c = cur
        for (mv, step) in zip(moves, cost_step):
            nr, nc = r + mv[0], c + mv[1]
            if not in_bounds(nr, nc) or occ[nr, nc]:
                continue
            new_g = g_now + step
            if (nr, nc) not in g or new_g < g[(nr, nc)]:
                g[(nr, nc)] = new_g
                parent[(nr, nc)] = cur
                f_new = new_g + heuristic((nr, nc), goal)
                heapq.heappush(open_heap, (f_new, new_g, (nr, nc)))

    t_ms = (time.perf_counter() - t0) * 1000.0
    return [], {"success": False, "reason": "no_path", "time_ms": t_ms}


def plan_with_inflation(
    costmap: np.ndarray,
    start: Coord,
    goal: Coord,
    threshold: float = 0.5,
    inflation_radius: int = 1,
    allow_diagonal: bool = True,
) -> Tuple[List[Coord], dict]:
    occ = costmap >= threshold
    occ = inflate_obstacles(occ, inflation_radius)
    path, info = astar(occ, costmap, start, goal, allow_diagonal=allow_diagonal)
    return path, info