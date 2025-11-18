"""Orchestrate full evaluation pipeline: PR curves, planner eval sweeps, aggregation, LaTeX tables.

Windows-friendly (no bash features). Designed to reproduce planner-in-the-loop tables
for labels and all prediction runs under predictions/<dataset>/*/<split>.

Example usage (PowerShell):
  python scripts/run_all_evals.py --dataset nyu --split val --thresholds 0.4 0.5 0.6
  python scripts/run_all_evals.py --dataset kitti --split val --thresholds 0.4 0.5 0.6 --num-scenes 0

Optional: limit scenes for a quick smoke test via --num-scenes 25.

Steps:
  1. Discover processed stems for dataset/split.
  2. Enumerate prediction directories.
  3. For each prediction dir: compute PR curve JSON.
  4. Run planner_eval for labels at each threshold (both planners).
  5. Run planner_eval for each prediction dir at each threshold.
  6. Aggregate summaries -> results/planner_sweeps.csv.
  7. Generate LaTeX planner tables -> docs/tables/planner_tables.tex.

All commands are executed sequentially via subprocess; failures abort.
"""

from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
import sys


def run(cmd: list[str]):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def discover_prediction_dirs(dataset: str, split: str) -> list[Path]:
    root = Path("predictions") / dataset
    if not root.exists():
        return []
    dirs = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        candidate = model_dir / split
        if candidate.is_dir():
            # require at least one .npy file
            if any(candidate.glob("*.npy")):
                dirs.append(candidate)
    return dirs


def main():
    ap = argparse.ArgumentParser(description="Run full evaluation pipeline")
    ap.add_argument("--dataset", choices=["nyu", "kitti"], required=True)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.4, 0.5, 0.6])
    ap.add_argument("--inflation-radius", type=int, default=2)
    ap.add_argument("--num-scenes", type=int, default=0, help="Limit scenes (0=all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-pr", action="store_true", help="Skip PR curve generation")
    ap.add_argument("--skip-planner", action="store_true", help="Skip planner eval runs")
    ap.add_argument("--dry-run", action="store_true", help="Print planned commands only")
    args = ap.parse_args()

    pred_dirs = discover_prediction_dirs(args.dataset, args.split)
    if pred_dirs:
        print(f"[INFO] Found {len(pred_dirs)} prediction dirs:")
        for d in pred_dirs:
            print("       -", d)
    else:
        print("[WARN] No prediction dirs found; will run labels-only planner eval if not skipped.")

    cmds: list[list[str]] = []

    # PR curves per prediction dir
    if not args.skip_pr:
        for pd in pred_dirs:
            cmds.append([
                sys.executable, "scripts/pr_curves.py",
                "--dataset", args.dataset,
                "--split", args.split,
                "--pred-dir", str(pd),
            ])

    # Planner eval: labels (source=labels, pred-dir omitted)
    if not args.skip_planner:
        for thr in args.thresholds:
            cmds.append([
                sys.executable, "scripts/planner_eval.py",
                "--dataset", args.dataset,
                "--split", args.split,
                "--planner", "both",
                "--threshold", str(thr),
                "--inflation-radius", str(args.inflation_radius),
                "--num-scenes", str(args.num_scenes),
                "--seed", str(args.seed),
            ])
        # Planner eval for each prediction dir
        for pd in pred_dirs:
            run_tag = pd.parent.name  # folder name acts as tag
            for thr in args.thresholds:
                cmds.append([
                    sys.executable, "scripts/planner_eval.py",
                    "--dataset", args.dataset,
                    "--split", args.split,
                    "--planner", "both",
                    "--threshold", str(thr),
                    "--inflation-radius", str(args.inflation_radius),
                    "--num-scenes", str(args.num_scenes),
                    "--seed", str(args.seed),
                    "--pred-dir", str(pd),
                    "--run-tag", run_tag,
                ])

    # Aggregation + LaTeX generation
    cmds.append([sys.executable, "scripts/aggregate_planner_summaries.py"])
    cmds.append([sys.executable, "scripts/make_latex_planner_tables.py"])

    if args.dry_run:
        print("[DRY-RUN] Planned commands:")
        for c in cmds:
            print("  ", " ".join(c))
        return

    for c in cmds:
        run(c)

    print("[DONE] Evaluation pipeline complete.")


if __name__ == "__main__":
    main()
