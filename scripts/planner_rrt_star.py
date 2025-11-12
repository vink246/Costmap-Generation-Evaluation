import time
from typing import Tuple, List, Optional
import numpy as np
from scipy.ndimage import binary_dilation

Coord = Tuple[float, float]


def inflate_obstacles(occ: np.ndarray, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return occ
    structure = np.ones((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
    return binary_dilation(occ.astype(bool), structure=structure)


def line_collision_free(occ: np.ndarray, p0: Coord, p1: Coord) -> bool:
    # sample along the line in grid coords
    r0, c0 = p0
    r1, c1 = p1
    dist = np.hypot(r1 - r0, c1 - c0)
    if dist == 0:
        return not occ[int(round(r0)), int(round(c0))]
    steps = int(np.ceil(dist * 2))  # 0.5 cell resolution
    rr = np.linspace(r0, r1, steps)
    cc = np.linspace(c0, c1, steps)
    rr = np.clip(np.round(rr).astype(int), 0, occ.shape[0] - 1)
    cc = np.clip(np.round(cc).astype(int), 0, occ.shape[1] - 1)
    return not np.any(occ[rr, cc])


def rrt_star(
    occ: np.ndarray,
    start: Coord,
    goal: Coord,
    *,
    cost: Optional[np.ndarray] = None,
    max_iter: int = 1500,
    step_size: float = 3.0,
    goal_radius: float = 3.0,
    neighbor_radius: float = 6.0,
    goal_sample_rate: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Coord], dict]:
    """
    Simple RRT* on an occupancy grid, in grid coordinates.
    """
    H, W = occ.shape
    rng = rng or np.random.default_rng(0)

    nodes: List[Coord] = [start]
    parent = {0: None}
    node_cost = {0: 0.0}

    def sample() -> Coord:
        if rng.random() < goal_sample_rate:
            return goal
        return (rng.uniform(0, H - 1), rng.uniform(0, W - 1))

    def nearest(p: Coord) -> int:
        pts = np.array(nodes)
        d = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
        return int(np.argmin(d))

    def steer(p_from: Coord, p_to: Coord, step: float) -> Coord:
        vec = np.array(p_to) - np.array(p_from)
        dist = np.hypot(vec[0], vec[1])
        if dist <= step:
            return tuple(p_to)
        return tuple(np.array(p_from) + vec / dist * step)

    def neighbors(idx: int, radius: float) -> List[int]:
        p = nodes[idx]
        pts = np.array(nodes)
        d = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
        return [i for i in range(len(nodes)) if d[i] <= radius]

    t0 = time.perf_counter()
    goal_idx = None
    hit_timeout = True

    for it in range(max_iter):
        q_rand = sample()
        idx_near = nearest(q_rand)
        q_new = steer(nodes[idx_near], q_rand, step_size)
        if not line_collision_free(occ, nodes[idx_near], q_new):
            continue
        nodes.append(q_new)
        new_idx = len(nodes) - 1
        parent[new_idx] = idx_near
        node_cost[new_idx] = node_cost[idx_near] + np.hypot(q_new[0] - nodes[idx_near][0], q_new[1] - nodes[idx_near][1])

        # rewire
        for nb in neighbors(new_idx, neighbor_radius):
            if nb == new_idx:
                continue
            if not line_collision_free(occ, nodes[nb], q_new):
                continue
            new_cost_val = node_cost[new_idx] + np.hypot(nodes[nb][0] - q_new[0], nodes[nb][1] - q_new[1])
            if new_cost_val < node_cost[nb]:
                parent[nb] = new_idx
                node_cost[nb] = new_cost_val

        # check goal
        if np.hypot(q_new[0] - goal[0], q_new[1] - goal[1]) <= goal_radius and line_collision_free(occ, q_new, goal):
            # add goal as a node and connect
            nodes.append(goal)
            goal_idx = len(nodes) - 1
            parent[goal_idx] = new_idx
            node_cost[goal_idx] = node_cost[new_idx] + np.hypot(goal[0] - q_new[0], goal[1] - q_new[1])
            hit_timeout = False
            break

    t_ms = (time.perf_counter() - t0) * 1000.0

    if goal_idx is None:
        reason = "timeout_rrtstar" if hit_timeout else "no_path"
        return [], {"success": False, "reason": reason, "time_ms": t_ms}

    # reconstruct
    path: List[Coord] = []
    cur = goal_idx
    while cur is not None:
        path.append(nodes[cur])
        cur = parent[cur]
    path.reverse()
    path_len = 0.0
    path_cost_sum = 0.0
    for i in range(1, len(path)):
        path_len += np.hypot(path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
    if cost is not None and len(path) > 0:
        rr = np.clip(np.round(np.array([p[0] for p in path])).astype(int), 0, occ.shape[0]-1)
        cc = np.clip(np.round(np.array([p[1] for p in path])).astype(int), 0, occ.shape[1]-1)
        path_cost_sum = float(np.sum(cost[rr, cc]))
    return path, {"success": True, "time_ms": t_ms, "path_length_cells": path_len, "path_cost_sum": path_cost_sum}


def plan_with_inflation(
    costmap: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    threshold: float = 0.5,
    inflation_radius: int = 1,
    rng: Optional[np.random.Generator] = None,
    # optional RRT* tuning parameters
    max_iter: int = 1500,
    step_size: float = 3.0,
    goal_radius: float = 3.0,
    neighbor_radius: float = 6.0,
    goal_sample_rate: float = 0.05,
):
    occ = costmap >= threshold
    occ = inflate_obstacles(occ, inflation_radius)
    # convert integer grid cells to float coords (center of cells)
    s = (float(start[0]), float(start[1]))
    g = (float(goal[0]), float(goal[1]))
    return rrt_star(
        occ,
        s,
        g,
        cost=costmap,
        max_iter=max_iter,
        step_size=step_size,
        goal_radius=goal_radius,
        neighbor_radius=neighbor_radius,
        goal_sample_rate=goal_sample_rate,
        rng=rng,
    )