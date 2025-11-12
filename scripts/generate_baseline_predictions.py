import argparse
from pathlib import Path
import numpy as np
from scipy import ndimage
import yaml


def baseline_cost_from_depth(depth_m: np.ndarray, near_thresh: float = 3.0, dilation_radius: int = 2):
    """Simple heuristic: cells nearer than near_thresh meters are obstacles.

    depth_m is expected to be in meters (unnormalized).
    """
    occ = (depth_m <= near_thresh).astype(np.float32)
    if dilation_radius > 0:
        struct = ndimage.generate_binary_structure(2, 1)
        struct = ndimage.iterate_structure(struct, dilation_radius)
        occ = ndimage.binary_dilation(occ > 0.5, structure=struct).astype(np.float32)
    free = occ < 0.5
    dist = ndimage.distance_transform_edt(free)
    dist /= (dist.max() + 1e-6)
    return (1.0 - dist).astype(np.float32)


def load_mean_std(cfg_path: Path):
    with open(cfg_path, 'r') as f:
        d = yaml.safe_load(f)
    mean = np.array(d['image']['normalize']['mean'], dtype=np.float32)
    std = np.array(d['image']['normalize']['std'], dtype=np.float32)
    return mean, std


def main():
    ap = argparse.ArgumentParser(description='Generate classical baseline predictions as .npy files')
    ap.add_argument('--processed-root', type=Path, default=Path('data/processed'))
    ap.add_argument('--dataset', choices=['nyu','kitti'], required=True)
    ap.add_argument('--split', choices=['train','val'], default='val')
    ap.add_argument('--near-thresh', type=float, default=3.0, help='Obstacle threshold in METERS after un-normalization')
    ap.add_argument('--dilation-radius', type=int, default=2)
    ap.add_argument('--out-root', type=Path, default=Path('predictions/baseline/heuristic'))
    ap.add_argument('--num', type=int, default=0)
    ap.add_argument('--data-config', type=Path, default=Path('configs/data.yaml'))
    args = ap.parse_args()

    ds_dir = args.processed_root / args.dataset / args.split
    stems = [p.stem for p in sorted(ds_dir.glob('*.npz'))]
    if args.num:
        stems = stems[:args.num]

    out_dir = args.out_root / args.dataset / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Un-normalization constants for depth channel
    mean, std = load_mean_std(args.data_config)
    depth_mean = float(mean[3])
    depth_std = float(std[3])

    count = 0
    for stem in stems:
        npz = np.load(ds_dir / f'{stem}.npz', allow_pickle=True)
        img = npz['image'].astype(np.float32)
        # Image in npz is normalized -> invert normalization for depth
        depth_norm = img[...,3]
        depth_m = depth_norm * depth_std + depth_mean
        pred = baseline_cost_from_depth(depth_m, near_thresh=args.near_thresh, dilation_radius=args.dilation_radius)
        np.save(out_dir / f'{stem}.npy', pred)
        count += 1
    print('Wrote', count, 'predictions to', out_dir, 'near_thresh(m)=', args.near_thresh)


if __name__ == '__main__':
    main()
