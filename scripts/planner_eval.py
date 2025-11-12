import argparse
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np

from planner_a_star import plan_with_inflation as astar_plan
from planner_rrt_star import plan_with_inflation as rrtstar_plan


def choose_free_cell(cost: np.ndarray, threshold: float, rng: np.random.Generator) -> Tuple[int, int]:
    occ = cost >= threshold
    H, W = cost.shape
    for _ in range(2000):
        r = rng.integers(0, H)
        c = rng.integers(0, W)
        if not occ[r, c]:
            return int(r), int(c)
    # fallback: just return center
    return H // 2, W // 2


def load_processed_paths(processed_root: Path, dataset: str, split: str) -> List[Path]:
    ds_dir = processed_root / dataset / split
    if not ds_dir.exists():
        raise FileNotFoundError(f"Processed dir not found: {ds_dir}")
    return sorted([p for p in ds_dir.glob("*.npz")])


def load_costmap_from_npz(npz_path: Path) -> Optional[np.ndarray]:
    try:
        data = np.load(npz_path)
        for key in ("cost", "costmap", "target", "y", "label"):
            if key in data:
                arr = data[key]
                # ensure 2D map
                if arr.ndim == 3:
                    arr = arr.squeeze()
                arr = arr.astype(np.float32)
                return arr
    except Exception:
        return None
    return None


def load_pred_costmap(pred_dir: Optional[Path], stem: str) -> Optional[np.ndarray]:
    if pred_dir is None:
        return None
    for ext in (".npy", ".npz"):
        p = pred_dir / f"{stem}{ext}"
        if p.exists():
            if ext == ".npy":
                arr = np.load(p).astype(np.float32)
                return arr.squeeze()
            else:
                data = np.load(p)
                key = list(data.keys())[0]
                arr = data[key].astype(np.float32)
                return arr.squeeze()
    return None


def eval_scene(costmap: np.ndarray, planner: str, threshold: float, inflation_radius: int, rng: np.random.Generator, rrt_params: Optional[dict] = None):
    H, W = costmap.shape
    start = choose_free_cell(costmap, threshold, rng)
    goal = choose_free_cell(costmap, threshold, rng)
    # avoid start==goal
    if start == goal:
        goal = ((goal[0] + H // 3) % H, (goal[1] + W // 3) % W)

    if planner == "astar":
        path, info = astar_plan(costmap, start, goal, threshold=threshold, inflation_radius=inflation_radius)
    elif planner == "rrtstar":
        rkw = rrt_params or {}
        path, info = rrtstar_plan(costmap, start, goal, threshold=threshold, inflation_radius=inflation_radius, rng=rng, **rkw)
    else:  # both
        path_a, info_a = astar_plan(costmap, start, goal, threshold=threshold, inflation_radius=inflation_radius)
        rkw = rrt_params or {}
        path_r, info_r = rrtstar_plan(costmap, start, goal, threshold=threshold, inflation_radius=inflation_radius, rng=rng, **rkw)
        return {
            "planner": "both",
            "success_astar": info_a.get("success", False),
            "success_rrtstar": info_r.get("success", False),
            "astar_time_ms": info_a.get("time_ms"),
            "rrtstar_time_ms": info_r.get("time_ms"),
            "astar_path_len_cells": info_a.get("path_length_cells"),
            "rrtstar_path_len_cells": info_r.get("path_length_cells"),
            "astar_cost_sum": info_a.get("path_cost_sum"),
            "rrtstar_cost_sum": info_r.get("path_cost_sum"),
            "reason_astar": info_a.get("reason"),
            "reason_rrtstar": info_r.get("reason"),
        }
    # normalize to single planner dict
    return {
        "planner": planner,
        "success": info.get("success", False),
        "time_ms": info.get("time_ms"),
        "path_len_cells": info.get("path_length_cells"),
        "path_cost_sum": info.get("path_cost_sum"),
        "reason": info.get("reason"),
    }


def main():
    ap = argparse.ArgumentParser(description="Planner-in-the-loop evaluation on costmaps.")
    ap.add_argument("--dataset", choices=["kitti", "nyu"], required=True)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--pred-dir", type=Path, default=None, help="Directory of predicted costmaps (basename matches processed files)")
    ap.add_argument("--planner", choices=["astar", "rrtstar", "both"], default="astar")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--inflation-radius", type=int, default=2)
    # Optional RRT* tuning flags (no effect on A*)
    ap.add_argument("--rrt-max-iter", type=int, default=1500)
    ap.add_argument("--rrt-step-size", type=float, default=3.0)
    ap.add_argument("--rrt-goal-radius", type=float, default=3.0)
    ap.add_argument("--rrt-neighbor-radius", type=float, default=6.0)
    ap.add_argument("--rrt-goal-sample-rate", type=float, default=0.05)
    # Optional run tag to disambiguate multiple prediction runs (e.g., model/loss)
    ap.add_argument("--run-tag", type=str, default="", help="Short identifier added to filenames and CSV (e.g., unet_rgb, vit_l1)")
    ap.add_argument("--num-scenes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("outputs/planner_eval"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    files = load_processed_paths(args.processed_root, args.dataset, args.split)
    if args.num_scenes > 0:
        files = files[: args.num_scenes]

    stats = []
    t0 = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)

    for f in files:
        stem = f.stem
        cm = load_pred_costmap(args.pred_dir, stem)
        if cm is None:
            cm = load_costmap_from_npz(f)
        if cm is None:
            print(f"[WARN] Could not load costmap for {f}, skipping")
            continue
        if cm.ndim != 2:
            cm = cm.squeeze()
        rrt_params = {
            "max_iter": args.rrt_max_iter,
            "step_size": args.rrt_step_size,
            "goal_radius": args.rrt_goal_radius,
            "neighbor_radius": args.rrt_neighbor_radius,
            "goal_sample_rate": args.rrt_goal_sample_rate,
        }
        rec = eval_scene(cm.astype(np.float32), args.planner, args.threshold, args.inflation_radius, rng, rrt_params if args.planner in ("rrtstar", "both") else None)
        rec.update({"file": str(f), "stem": stem})
        stats.append(rec)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # aggregate
    def mean_safe(vals):
        arr = np.array([v for v in vals if v is not None], dtype=float)
        return float(arr.mean()) if arr.size else None

    if args.planner == "both":
        succ_a = np.mean([r.get("success_astar", False) for r in stats]) if stats else 0.0
        succ_r = np.mean([r.get("success_rrtstar", False) for r in stats]) if stats else 0.0
        agg = {
            "num_scenes": len(stats),
            "success_rate_astar": float(succ_a),
            "success_rate_rrtstar": float(succ_r),
            "mean_time_ms_astar": mean_safe([r.get("astar_time_ms") for r in stats]),
            "mean_time_ms_rrtstar": mean_safe([r.get("rrtstar_time_ms") for r in stats]),
            "mean_path_len_cells_astar": mean_safe([r.get("astar_path_len_cells") for r in stats]),
            "mean_path_len_cells_rrtstar": mean_safe([r.get("rrtstar_path_len_cells") for r in stats]),
            "mean_cost_sum_astar": mean_safe([r.get("astar_cost_sum") for r in stats]),
            "elapsed_ms_total": elapsed_ms,
        }
    else:
        succ = np.mean([r.get("success", False) for r in stats]) if stats else 0.0
        agg = {
            "num_scenes": len(stats),
            "success_rate": float(succ),
            "mean_time_ms": mean_safe([r.get("time_ms") for r in stats]),
            "mean_path_len_cells": mean_safe([r.get("path_len_cells") for r in stats]),
            "mean_cost_sum": mean_safe([r.get("path_cost_sum") for r in stats]),
            "elapsed_ms_total": elapsed_ms,
        }

    # save
    # build tag + metadata
    src_tag = "pred" if args.pred_dir is not None else "labels"
    extra = f"_{args.run_tag}" if (args.run_tag and src_tag == "pred") else ""
    tag = f"{args.dataset}_{args.split}_{args.planner}_thr{args.threshold}_infl{args.inflation_radius}_{src_tag}{extra}_n{len(stats)}"
    # ensure argparse Namespace with Path fields is JSON-serializable
    args_serializable = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    # reproducibility metadata
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = None
    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "git_commit": git_hash,
        "source": src_tag,
        "seed": args.seed,
        "run_tag": args.run_tag or None,
    }
    with open(args.out / f"summary_{tag}.json", "w") as f:
        json.dump({"args": args_serializable, "meta": meta, "aggregate": agg, "per_scene": stats}, f, indent=2)

    # CSV with key fields
    import csv
    csv_path = args.out / f"per_scene_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()) if stats else ["file", "success"])
        writer.writeheader()
        for r in stats:
            writer.writerow(r)

    print("Saved:", args.out / f"summary_{tag}.json")


if __name__ == "__main__":
    main()
