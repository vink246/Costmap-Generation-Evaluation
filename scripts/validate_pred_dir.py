"""Validate a prediction directory for basic sanity.

Reports per-sample stats (mean, std, min, max, unique count) and flags degenerate
maps (std < 1e-6 or unique == 1). Writes a JSON summary to results/validate_<dataset>_<split>_<tag>.json.

Usage:
  python scripts/validate_pred_dir.py --pred-dir predictions/kitti/unet_rgbd/val --dataset kitti --split val --num 50
"""
import argparse
import json
from pathlib import Path
import numpy as np


def is_degenerate(arr: np.ndarray) -> bool:
    return arr.size == 0 or arr.std() < 1e-6 or float(arr.min()) == float(arr.max())


def collect_stats(path: Path):
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze()
    return {
        'stem': path.stem,
        'exists': True,
        'shape': list(arr.shape),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'unique': int(len(np.unique(arr))),
        'degenerate': bool(is_degenerate(arr)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-dir', type=Path, required=True)
    ap.add_argument('--dataset', choices=['nyu','kitti'], required=True)
    ap.add_argument('--split', choices=['train','val'], default='val')
    ap.add_argument('--num', type=int, default=0, help='Number of samples to check (0=all)')
    ap.add_argument('--out-root', type=Path, default=Path('results'))
    args = ap.parse_args()

    if not args.pred_dir.exists():
        raise FileNotFoundError(args.pred_dir)

    npy_files = sorted(args.pred_dir.glob('*.npy'))
    if args.num:
        npy_files = npy_files[:args.num]

    samples = [collect_stats(p) for p in npy_files]
    degenerate_rate = float(sum(s['degenerate'] for s in samples)) / (len(samples) + 1e-6)
    mean_of_means = float(np.mean([s['mean'] for s in samples])) if samples else 0.0
    mean_of_stds = float(np.mean([s['std'] for s in samples])) if samples else 0.0
    aggregate = {
        'dir': str(args.pred_dir),
        'count': len(samples),
        'exists_rate': 1.0 if samples else 0.0,
        'degenerate_rate': degenerate_rate,
        'mean_of_means': mean_of_means,
        'mean_of_stds': mean_of_stds,
    }

    tag = args.pred_dir.parent.name if args.pred_dir.parent.name not in ('kitti','nyu') else 'pred'
    out_name = f'validate_{args.dataset}_{args.split}_{tag}.json'
    args.out_root.mkdir(parents=True, exist_ok=True)
    out_path = args.out_root / out_name
    with open(out_path, 'w') as f:
        json.dump({'aggregate': aggregate, 'samples': samples}, f, indent=2)
    print('Wrote validation JSON:', out_path)


if __name__ == '__main__':
    main()
