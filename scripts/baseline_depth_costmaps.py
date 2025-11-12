"""Generate classical depth->cost baseline metrics for NYU and KITTI.

Procedure per sample:
 1. Load processed pair .npz (depth must be reconstructed from image's 4th channel or meta if available).
 2. Reconstruct baseline occupancy via simple threshold on depth (near = obstacle) then morphology (binary dilation) and distance transform to continuous cost.
 3. Compute MAE, IoU, Precision, Recall, F1 at tau=0.5 versus ground-truth costmap label.
Aggregates written to results/baseline_depth_to_cost_<dataset>.json.
"""
import argparse
import numpy as np
from pathlib import Path
import json
from scipy import ndimage
import subprocess
from datetime import datetime
import yaml

try:
    from src.train.metrics import mae, iou_binary, precision_recall_f1
except ModuleNotFoundError:
    # Fallback: np implementations mirroring torch versions for baseline purposes
    def mae(pred, target):
        return float(np.mean(np.abs(pred - target)))
    def iou_binary(pred, target, thresh=0.5, eps=1e-6):
        pb = (pred > thresh).astype(np.float32)
        tb = (target > thresh).astype(np.float32)
        inter = float((pb * tb).sum())
        union = float(np.clip(pb + tb, 0, 1).sum())
        return inter / (union + eps)
    def precision_recall_f1(pred, target, thresh=0.5, eps=1e-6):
        pb = (pred > thresh).astype(np.float32)
        tb = (target > thresh).astype(np.float32)
        tp = float((pb * tb).sum())
        fp = float(((pb == 1) & (tb == 0)).sum())
        fn = float(((pb == 0) & (tb == 1)).sum())
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        return prec, rec, f1


def depth_channel(rgbd: np.ndarray):
    # Expect (H,W,4)
    if rgbd.shape[-1] == 4:
        return rgbd[..., 3]
    raise ValueError("RGBD image missing depth channel")


def baseline_cost_from_depth(depth_m: np.ndarray, near_thresh: float = 3.0, dilation_radius: int = 2):
    # Depths below near_thresh meters mark obstacles (heuristic)
    occ = (depth_m <= near_thresh).astype(np.float32)
    if dilation_radius > 0:
        struct = ndimage.generate_binary_structure(2, 1)
        struct = ndimage.iterate_structure(struct, dilation_radius)
        occ = ndimage.binary_dilation(occ > 0.5, structure=struct).astype(np.float32)
    # Continuous cost via distance from free
    free = occ < 0.5
    dist = ndimage.distance_transform_edt(free)
    dist /= (dist.max() + 1e-6)
    cont = 1.0 - dist
    return cont.astype(np.float32)


def iter_npz(processed_root: Path, dataset: str, split: str = 'val'):
    ds_dir = processed_root / dataset / split
    for p in sorted(ds_dir.glob('*.npz')):
        data = np.load(p, allow_pickle=True)
        img = data['image'].astype(np.float32)
        label = data['costmap'].astype(np.float32)
        yield p.stem, img, label


def evaluate(processed_root: Path, dataset: str, split: str, near_thresh: float, dilation_radius: int, limit: int = 0, depth_mean: float = 0.5, depth_std: float = 0.25):
    records = []
    for i, (stem, rgbd, label) in enumerate(iter_npz(processed_root, dataset, split)):
        if limit and i >= limit:
            break
        depth_norm = depth_channel(rgbd)
        depth_m = depth_norm * depth_std + depth_mean
        pred = baseline_cost_from_depth(depth_m, near_thresh=near_thresh, dilation_radius=dilation_radius)
        # make sure shape match
        if pred.shape != label.shape:
            pred = ndimage.zoom(pred, (label.shape[0]/pred.shape[0], label.shape[1]/pred.shape[1]), order=1)
        pred_t = np.clip(pred, 0, 1)
        m_mae = mae(pred_t, label)
        m_iou = iou_binary(pred_t, label)
        p, r, f1 = precision_recall_f1(pred_t, label)
        records.append({
            'stem': stem,
            'mae': m_mae,
            'iou': m_iou,
            'precision': p,
            'recall': r,
            'f1': f1
        })
    if not records:
        return {}, []
    agg = {
        'dataset': dataset,
        'split': split,
        'near_thresh': near_thresh,
        'dilation_radius': dilation_radius,
        'num_samples': len(records),
        'mae': float(np.mean([r['mae'] for r in records])),
        'iou': float(np.mean([r['iou'] for r in records])),
        'precision': float(np.mean([r['precision'] for r in records])),
        'recall': float(np.mean([r['recall'] for r in records])),
        'f1': float(np.mean([r['f1'] for r in records])),
    }
    return agg, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--processed-root', type=Path, default=Path('data/processed'))
    ap.add_argument('--datasets', nargs='+', default=['nyu', 'kitti'])
    ap.add_argument('--split', default='val')
    ap.add_argument('--near-thresh', type=float, default=3.0, help='Obstacle threshold in METERS after un-normalization')
    ap.add_argument('--dilation-radius', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0, help='Optional cap on number of samples (0 = all)')
    ap.add_argument('--out', type=Path, default=Path('results'))
    ap.add_argument('--data-config', type=Path, default=Path('configs/data.yaml'))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    all_aggs = []
    # Load depth channel normalization parameters
    with open(args.data_config, 'r') as f:
        d = yaml.safe_load(f)
    mean = d['image']['normalize']['mean']
    std = d['image']['normalize']['std']
    depth_mean = float(mean[3]); depth_std = float(std[3])
    for ds in args.datasets:
        agg, recs = evaluate(args.processed_root, ds, args.split, args.near_thresh, args.dilation_radius, args.limit, depth_mean=depth_mean, depth_std=depth_std)
        if agg:
            all_aggs.append(agg)
            out_json = args.out / f"baseline_depth_to_cost_{ds}.json"
            # reproducibility meta
            try:
                git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            except Exception:
                git_hash = None
            meta = {"created_at": datetime.utcnow().isoformat() + "Z", "git_commit": git_hash, "source": "baseline_depth_heuristic"}
            with open(out_json, 'w') as f:
                json.dump({'meta': meta, 'aggregate': agg, 'per_sample': recs}, f, indent=2)
            print('Saved baseline:', out_json)
        else:
            print('No records for dataset', ds)

    combo_path = args.out / 'baseline_depth_to_cost_summary.json'
    with open(combo_path, 'w') as f:
        json.dump({'runs': all_aggs, 'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}, f, indent=2)
    print('Saved summary:', combo_path)


if __name__ == '__main__':
    main()
