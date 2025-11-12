"""
High-level orchestration script to reproduce key artifacts.
Usage (examples):
  - python scripts/run_all.py --stage pr_curves --dataset nyu --split val
  - python scripts/run_all.py --stage planner --dataset kitti --split val --planner both --pred-dir runs/kitti_hybrid/preds
The script avoids hardcoding training; hook in your trainer commands if desired.
"""

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]):
    print("[RUN]", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pr_curves", "planner"], required=True)
    ap.add_argument("--dataset", choices=["nyu", "kitti"], default="nyu")
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--pred-dir", type=Path, default=None)
    ap.add_argument("--planner", choices=["astar", "rrtstar", "both"], default="astar")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--inflation-radius", type=int, default=2)
    ap.add_argument("--num-scenes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.stage == "pr_curves":
        # Placeholder: integrate with PR plotting utility if present
        print("Please run PR curve script with --pred-dir to produce curves.")
    elif args.stage == "planner":
        run([
            "python",
            "scripts/planner_eval.py",
            "--dataset",
            args.dataset,
            "--split",
            args.split,
            "--planner",
            args.planner,
            "--threshold",
            str(args.threshold),
            "--inflation-radius",
            str(args.inflation_radius),
            "--num-scenes",
            str(args.num_scenes),
            "--seed",
            str(args.seed),
        ] + (["--pred-dir", str(args.pred_dir)] if args.pred_dir else []))


if __name__ == "__main__":
    main()
