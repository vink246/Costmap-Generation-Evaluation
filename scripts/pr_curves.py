import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
import subprocess
from datetime import datetime


def load_label(processed_root: Path, dataset: str, split: str, stem: str) -> np.ndarray:
    p = processed_root / dataset / split / f"{stem}.npz"
    d = np.load(p, allow_pickle=True)
    return d['costmap'].astype(np.float32)


def load_pred(pred_dir: Path, stem: str) -> np.ndarray:
    p_npy = pred_dir / f"{stem}.npy"
    p_npz = pred_dir / f"{stem}.npz"
    if p_npy.exists():
        arr = np.load(p_npy).astype(np.float32)
    elif p_npz.exists():
        d = np.load(p_npz)
        arr = d[list(d.keys())[0]].astype(np.float32)
    else:
        raise FileNotFoundError(f"Missing prediction for {stem}")
    # allow raw logits -> squash if range looks like logits
    if arr.min() < 0.0 or arr.max() > 1.0:
        arr = expit(arr)
    return arr


def binary_stats(pred: np.ndarray, target: np.ndarray, thresh: float, eps: float = 1e-6):
    # make sure same shape (resize pred if needed)
    if pred.shape != target.shape:
        # simple bilinear resize via zoom
        zoom_y = target.shape[0] / pred.shape[0]
        zoom_x = target.shape[1] / pred.shape[1]
        pred = np.clip(np.array(
            __import__('scipy').ndimage.zoom(pred, (zoom_y, zoom_x), order=1)
        ), 0, 1)
    pb = (pred > thresh).astype(np.float32)
    tb = (target > thresh).astype(np.float32)
    tp = float((pb * tb).sum())
    fp = float(((pb == 1) & (tb == 0)).sum())
    fn = float(((pb == 0) & (tb == 1)).sum())
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser(description='Compute PR curves for predicted costmaps')
    ap.add_argument('--processed-root', type=Path, default=Path('data/processed'))
    ap.add_argument('--dataset', choices=['nyu','kitti'], required=True)
    ap.add_argument('--split', choices=['val','train'], default='val')
    ap.add_argument('--pred-dir', type=Path, required=True, help='Directory containing <stem>.npy predictions')
    ap.add_argument('--out', type=Path, default=Path('results/pr_curves'))
    ap.add_argument('--num', type=int, default=0, help='Use first N samples (0 = all)')
    args = ap.parse_args()

    stems = [p.stem for p in sorted((args.processed_root / args.dataset / args.split).glob('*.npz'))]
    if args.num:
        stems = stems[:args.num]

    thresholds = [round(t,3) for t in np.linspace(0.0, 1.0, 51)]
    out = {'dataset': args.dataset, 'split': args.split, 'num_samples': len(stems), 'thresholds': thresholds, 'precision': [], 'recall': [], 'f1': []}

    precs = []; recs = []; f1s = []
    for thr in thresholds:
        ps = rs = fs = 0.0
        for stem in stems:
            lab = load_label(args.processed_root, args.dataset, args.split, stem)
            pred = load_pred(args.pred_dir, stem)
            p, r, f1 = binary_stats(pred, lab, thr)
            ps += p; rs += r; fs += f1
        n = max(1, len(stems))
        precs.append(ps/n); recs.append(rs/n); f1s.append(fs/n)
    out['precision'] = precs; out['recall'] = recs; out['f1'] = f1s

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_{args.split}_{args.pred_dir.parent.name}_{args.pred_dir.name}"
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = None
    out_full = {"meta": {"created_at": datetime.utcnow().isoformat() + "Z", "git_commit": git_hash}, "data": out}
    with open(args.out / f"pr_{tag}.json", 'w') as f:
        json.dump(out_full, f, indent=2)
    print('Wrote PR curve to', args.out / f"pr_{tag}.json")


if __name__ == '__main__':
    main()
