"""Inspect (un-normalized) depth channel statistics for a dataset.

Loads processed pairs (.npz), inverts normalization using configs/data.yaml, and prints
per-sample and aggregate min/mean/max/std. Useful for diagnosing saturation.

Usage:
  python scripts/inspect_depth_channel.py --dataset kitti --split val --num 10
"""
import argparse
from pathlib import Path
import numpy as np
import yaml


def load_mean_std(cfg_path: Path):
    with open(cfg_path, 'r') as f:
        d = yaml.safe_load(f)
    mean = np.array(d['image']['normalize']['mean'], dtype=np.float32)
    std = np.array(d['image']['normalize']['std'], dtype=np.float32)
    return float(mean[3]), float(std[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--processed-root', type=Path, default=Path('data/processed'))
    ap.add_argument('--dataset', choices=['nyu','kitti'], required=True)
    ap.add_argument('--split', choices=['train','val'], default='val')
    ap.add_argument('--num', type=int, default=10)
    ap.add_argument('--data-config', type=Path, default=Path('configs/data.yaml'))
    args = ap.parse_args()

    depth_mean, depth_std = load_mean_std(args.data_config)
    ds_dir = args.processed_root / args.dataset / args.split
    stems = [p.stem for p in sorted(ds_dir.glob('*.npz'))]
    if args.num:
        stems = stems[:args.num]
    stats = []
    for stem in stems:
        data = np.load(ds_dir / f'{stem}.npz', allow_pickle=True)
        img = data['image'].astype(np.float32)
        depth_norm = img[...,3]
        depth_m = depth_norm * depth_std + depth_mean
        stats.append({'stem': stem, 'min': float(depth_m.min()), 'mean': float(depth_m.mean()), 'max': float(depth_m.max()), 'std': float(depth_m.std())})
    if not stats:
        print('No samples found')
        return
    agg = {
        'num_samples': len(stats),
        'mean_of_means': float(np.mean([s['mean'] for s in stats])),
        'mean_of_stds': float(np.mean([s['std'] for s in stats])),
        'global_min': float(min(s['min'] for s in stats)),
        'global_max': float(max(s['max'] for s in stats)),
    }
    print('[inspect_depth_channel] aggregate:', agg)
    for s in stats:
        print('[sample]', s)


if __name__ == '__main__':
    main()
