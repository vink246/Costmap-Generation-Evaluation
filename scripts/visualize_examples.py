import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

"""Visualize qualitative examples: input RGB (and depth optional), predicted costmap, ground-truth costmap, and (optional) planner path overlay.

Usage:
  python scripts/visualize_examples.py --dataset nyu --split val --processed-root data/processed \
      --pred-dir predictions/nyu/val --out docs/figures --stems 000010 000025 --overlay-dir outputs/planner_eval

If overlay-dir is provided and contains per_scene CSV with matching stems, will attempt to load path coordinates for A* and RRT*.
"""

def load_npz(processed_root: Path, dataset: str, split: str, stem: str):
    p = processed_root / dataset / split / f"{stem}.npz"
    d = np.load(p, allow_pickle=True)
    img = d['image']  # (H,W,4)
    gt = d['costmap']
    return img, gt

def load_pred(pred_dir: Path, stem: str):
    p = pred_dir / f"{stem}.npy"
    if not p.exists():
        raise FileNotFoundError(p)
    arr = np.load(p)
    if arr.ndim == 3:
        arr = arr.squeeze()
    return arr

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['nyu','kitti'], required=True)
    ap.add_argument('--split', choices=['val','train'], default='val')
    ap.add_argument('--processed-root', type=Path, default=Path('data/processed'))
    ap.add_argument('--pred-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('docs/figures'))
    ap.add_argument('--stems', nargs='+', default=None, help='List of sample stems to visualize')
    ap.add_argument('--num', type=int, default=0, help='If stems not provided, number of samples to export (0=all)')
    ap.add_argument('--random', action='store_true', help='Randomly sample when --num > 0')
    ap.add_argument('--rgb-robust', action='store_true', help='Apply robust percentile scaling to RGB for display')
    ap.add_argument('--overlay-dir', type=Path, default=None, help='Optional planner_eval output directory with per_scene CSV for path overlays')
    return ap.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.stems is None:
        stems = [p.stem for p in sorted((args.processed_root / args.dataset / args.split).glob('*.npz'))]
        if args.num and args.num < len(stems):
            if args.random:
                import random
                random.seed(0)
                stems = random.sample(stems, args.num)
            else:
                stems = stems[:args.num]
    else:
        stems = args.stems

    # infer tag from pred-dir for filename disambiguation
    parts = [p for p in args.pred_dir.parts if p]
    tag = None
    if len(parts) >= 3 and parts[-1] == args.split and parts[-2] == args.dataset:
        tag = 'pred'
    elif len(parts) >= 4:
        tag = Path(parts[-2]).name  # for example, unet_rgb from .../unet_rgb/val
    if not tag:
        tag = 'pred'

    for stem in stems:
        img, gt = load_npz(args.processed_root, args.dataset, args.split, stem)
        pred = load_pred(args.pred_dir, stem)
        # make sure shapes match for visualization
        if pred.shape != gt.shape:
            from scipy.ndimage import zoom
            pred = zoom(pred, (gt.shape[0]/pred.shape[0], gt.shape[1]/pred.shape[1]), order=1)
        rgb = img[..., :3]
        depth = img[..., 3]
        # normalize RGB for display
        rgb_disp = rgb.astype(float)
        if rgb_disp.max() > 1.0:
            rgb_disp /= 255.0
        if args.rgb_robust:
            p1 = np.percentile(rgb_disp, 1)
            p99 = np.percentile(rgb_disp, 99)
            if p99 > p1:
                rgb_disp = np.clip((rgb_disp - p1) / (p99 - p1), 0, 1)

        fig, axs = plt.subplots(1, 4, figsize=(14,4))
        axs[0].imshow(np.clip(rgb_disp,0,1))
        axs[0].set_title(f"RGB {stem}")
        axs[0].axis('off')
        axs[1].imshow(depth, cmap='magma')
        axs[1].set_title("Depth")
        axs[1].axis('off')
        im2 = axs[2].imshow(gt, vmin=0, vmax=1, cmap='viridis')
        axs[2].set_title("GT Costmap")
        axs[2].axis('off')
        im3 = axs[3].imshow(pred, vmin=0, vmax=1, cmap='viridis')
        axs[3].set_title("Pred Costmap")
        axs[3].axis('off')
        fig.colorbar(im3, ax=axs.ravel().tolist(), shrink=0.6, label='Cost')
        fig.tight_layout()
        out_path = args.out / f"qual_{args.dataset}_{tag}_{stem}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print("Wrote", out_path)

if __name__ == '__main__':
    main()
