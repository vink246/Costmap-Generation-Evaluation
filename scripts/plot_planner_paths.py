import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

from planner_a_star import plan_with_inflation as astar_plan
from planner_rrt_star import plan_with_inflation as rrt_plan

Coord = Tuple[int, int]


def format_path_len(info: dict) -> str:
    length = info.get("path_length_cells")
    return f"{length:.1f}" if isinstance(length, (int, float)) else "—"


def load_costmap(processed_root: Path, dataset: str, split: str, stem: str) -> np.ndarray:
    npz = processed_root / dataset / split / f"{stem}.npz"
    data = np.load(npz, allow_pickle=True)
    for key in ("costmap", "cost", "target", "y", "label"):
        if key in data:
            arr = data[key].astype(np.float32)
            if arr.ndim == 3:
                arr = arr.squeeze()
            return arr
    raise KeyError(f"No costmap-like key found in {npz}")


def load_prediction(pred_dir: Path, stem: str, target_shape: Tuple[int, int]) -> np.ndarray:
    path = pred_dir / f"{stem}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction: {path}")
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze()
    if arr.shape != target_shape:
        scale = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
        arr = zoom(arr, scale, order=1)
    return arr


def choose_free_cell(cost: np.ndarray, threshold: float, rng: np.random.Generator) -> Coord:
    occ = cost >= threshold
    H, W = cost.shape
    for _ in range(4000):
        r = int(rng.integers(0, H))
        c = int(rng.integers(0, W))
        if not occ[r, c]:
            return r, c
    return H // 2, W // 2


def sample_paths(
    label_cost: np.ndarray,
    pred_cost: np.ndarray,
    threshold: float,
    inflation: int,
    rng: np.random.Generator,
    max_trials: int,
    mode: str,
) -> Tuple[Coord, Coord, List[Coord], List[Coord], dict, dict]:
    for _ in range(max_trials):
        start = choose_free_cell(label_cost, threshold, rng)
        goal = choose_free_cell(label_cost, threshold, rng)
        if start == goal:
            continue
        path_label, info_label = astar_plan(
            label_cost,
            start,
            goal,
            threshold=threshold,
            inflation_radius=inflation,
        )
        path_pred, info_pred = astar_plan(
            pred_cost,
            start,
            goal,
            threshold=threshold,
            inflation_radius=inflation,
        )
        label_success = info_label.get("success")
        pred_success = info_pred.get("success")
        if mode == "success" and label_success and pred_success:
            return start, goal, path_label, path_pred, info_label, info_pred
        if mode == "failure" and label_success and not pred_success:
            return start, goal, path_label, path_pred, info_label, info_pred
    raise RuntimeError(
        "Failed to sample a start/goal meeting the requested condition within max_trials"
    )


def run_rrt_with_retries(
    cost: np.ndarray,
    start: Coord,
    goal: Coord,
    threshold: float,
    inflation: int,
    args,
    rng: np.random.Generator,
):
    for attempt in range(args.rrt_retries):
        local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        path, info = rrt_plan(
            cost,
            start,
            goal,
            threshold=threshold,
            inflation_radius=inflation,
            rng=local_rng,
            max_iter=args.rrt_max_iter,
            step_size=args.rrt_step_size,
            goal_radius=args.rrt_goal_radius,
            neighbor_radius=args.rrt_neighbor_radius,
            goal_sample_rate=args.rrt_goal_sample_rate,
            return_tree=True,
        )
        if info.get("success"):
            return path, info
    return path, info


def plot_paths(
    dataset: str,
    stem: str,
    tag: str,
    label_cost: np.ndarray,
    pred_cost: np.ndarray,
    start: Coord,
    goal: Coord,
    path_label: List[Coord],
    path_pred: List[Coord],
    info_label: dict,
    info_pred: dict,
    rrt_info: dict,
    out_path: Path,
):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))

    # A* comparison panel
    ax0 = axs[0]
    ax0.imshow(pred_cost, cmap="viridis", vmin=0, vmax=1)
    ax0.set_title(f"A* Paths · {dataset.upper()} {stem}")
    for path, color, label in (
        (path_label, "#ffd700", "Label A*"),
        (path_pred, "#ff1493", "Pred A*"),
    ):
        if path:
            rr = [p[0] for p in path]
            cc = [p[1] for p in path]
            ax0.plot(cc, rr, color=color, linewidth=2.0, label=label)
    ax0.scatter([start[1]], [start[0]], color="#00ff90", s=35, marker="o", label="Start")
    ax0.scatter([goal[1]], [goal[0]], color="#ff4444", s=45, marker="x", label="Goal")
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.legend(loc="upper right", fontsize=8, frameon=False)
    label_len = format_path_len(info_label)
    pred_len = format_path_len(info_pred)
    pred_reason = info_pred.get("reason")
    text_lines = [
        f"Label success={info_label.get('success', False)} len={label_len}",
        f"Pred success={info_pred.get('success', False)} len={pred_len}",
    ]
    if pred_reason and not info_pred.get("success", False):
        text_lines.append(f"Pred reason: {pred_reason}")
    ax0.text(
        0.02,
        0.02,
        "\n".join(text_lines),
        transform=ax0.transAxes,
        fontsize=8,
        color="white",
        bbox=dict(facecolor="black", alpha=0.35, pad=2, edgecolor="none"),
    )

    # RRT* tree panel
    ax1 = axs[1]
    ax1.imshow(pred_cost, cmap="viridis", vmin=0, vmax=1)
    ax1.set_title(f"RRT* Tree — {tag}")
    nodes = np.array(rrt_info.get("tree_nodes", []))
    if nodes.size:
        edges = rrt_info.get("tree_edges", [])
        for parent_idx, child_idx in edges:
            p = nodes[parent_idx]
            c = nodes[child_idx]
            ax1.plot([p[1], c[1]], [p[0], c[0]], color="white", alpha=0.25, linewidth=0.7)
        ax1.scatter(nodes[:, 1], nodes[:, 0], s=6, color="white", alpha=0.4)
    rrt_path = rrt_info.get("path", None)
    if rrt_path:
        rr = [p[0] for p in rrt_path]
        cc = [p[1] for p in rrt_path]
        ax1.plot(cc, rr, color="#ffa500", linewidth=2.2, label="RRT* Path")
    else:
        # fallback to highlight best-effort path (if returned)
        final_path = rrt_info.get("path_nodes") or []
        if final_path:
            rr = [p[0] for p in final_path]
            cc = [p[1] for p in final_path]
            ax1.plot(cc, rr, color="#ffa500", linewidth=2.2)
    ax1.scatter([start[1]], [start[0]], color="#00ff90", s=35, marker="o")
    ax1.scatter([goal[1]], [goal[0]], color="#ff4444", s=45, marker="x")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(
        0.02,
        0.02,
        f"Success={rrt_info.get('success', False)}  t={rrt_info.get('time_ms', 0):.1f} ms",
        transform=ax1.transAxes,
        fontsize=8,
        color="white",
        bbox=dict(facecolor="black", alpha=0.35, pad=2, edgecolor="none"),
    )

    fig.suptitle(f"Planner Overlays · {dataset.upper()} {tag}")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(description="Generate planner overlay figures (A* + RRT*)")
    ap.add_argument("--dataset", choices=["nyu", "kitti"], required=True)
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--tag", type=str, required=True, help="Short identifier used in titles/filenames")
    ap.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--inflation", type=int, default=2)
    ap.add_argument(
        "--sample-mode",
        choices=["success", "failure"],
        default="success",
        help="Select start/goal pairs where both planners succeed or where the prediction fails",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-trials", type=int, default=500)
    ap.add_argument("--rrt-max-iter", type=int, default=1500)
    ap.add_argument("--rrt-step-size", type=float, default=3.0)
    ap.add_argument("--rrt-goal-radius", type=float, default=3.0)
    ap.add_argument("--rrt-neighbor-radius", type=float, default=6.0)
    ap.add_argument("--rrt-goal-sample-rate", type=float, default=0.05)
    ap.add_argument("--rrt-retries", type=int, default=3)
    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stem in args.stems:
        label_cost = load_costmap(args.processed_root, args.dataset, args.split, stem)
        pred_cost = load_prediction(args.pred_dir, stem, label_cost.shape)
        start, goal, path_label, path_pred, info_label, info_pred = sample_paths(
            label_cost,
            pred_cost,
            args.threshold,
            args.inflation,
            rng,
            args.max_trials,
            args.sample_mode,
        )
        rrt_path, rrt_info = run_rrt_with_retries(
            pred_cost,
            start,
            goal,
            args.threshold,
            args.inflation,
            args,
            rng,
        )
        # store final path for visualization convenience
        if rrt_info.get("success"):
            rrt_info.setdefault("path", rrt_path)
        out_path = args.out_dir / f"planner_{args.dataset}_{args.tag}_{stem}.png"
        plot_paths(
            args.dataset,
            stem,
            args.tag,
            label_cost,
            pred_cost,
            start,
            goal,
            path_label,
            path_pred,
            info_label,
            info_pred,
            rrt_info,
            out_path,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
