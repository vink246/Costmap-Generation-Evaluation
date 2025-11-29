import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import yaml

"""Visualize qualitative examples: input RGB (and depth optional), predicted costmap, ground-truth costmap, and (optional) planner path overlay.

Usage:
  python scripts/visualize_examples.py --dataset nyu --split val --processed-root data/processed \
      --pred-dir predictions/nyu/val --out docs/figures --stems 000010 000025 --overlay-dir outputs/planner_eval

Usage with config:
  python scripts/visualize_examples.py --config configs/vit_kitti.yaml

If overlay-dir is provided and contains per_scene CSV with matching stems, will attempt to load path coordinates for A* and RRT*.
"""

def load_npz(processed_root: Path, dataset: str, split: str, stem: str):
    p = processed_root / dataset / split / f"{stem}.npz"
    d = np.load(p, allow_pickle=True)
    img = d['image']  # (H,W,4)
    gt = d['costmap']
    meta = d.get('meta', None)
    if meta is not None and isinstance(meta, np.ndarray):
        meta = meta.item()
    return img, gt, meta

def load_pred(pred_dir: Path, stem: str):
    p = pred_dir / f"{stem}.npy"
    if not p.exists():
        raise FileNotFoundError(p)
    arr = np.load(p)
    if arr.ndim == 3:
        arr = arr.squeeze()
    return arr

def visualize_normalized_depth(depth_norm: np.ndarray, colormap: str = 'magma', debug: bool = False):
    """Visualize z-score normalized depth with negative fill values."""
    
    # Find minimum value (likely the invalid fill)
    min_val = depth_norm.min()
    min_threshold = min_val + 0.01
    at_min_mask = depth_norm <= min_threshold
    at_min_count = at_min_mask.sum()
    
    if debug:
        print(f"\n  Depth stats: range=[{depth_norm.min():.4f}, {depth_norm.max():.4f}], "
              f"at_min={at_min_count:,} ({100*at_min_count/depth_norm.size:.1f}%), "
              f"positive={((depth_norm > 0).sum()):,} ({100*(depth_norm > 0).sum()/depth_norm.size:.1f}%)")
    
    # Valid pixels are those above the minimum fill value
    if at_min_count > depth_norm.size * 0.7:
        valid_mask = ~at_min_mask
    else:
        valid_mask = depth_norm > -1.5  # Fallback threshold
    
    coverage = 100 * valid_mask.sum() / depth_norm.size
    
    if debug:
        print(f"  => Valid: {valid_mask.sum():,} ({coverage:.1f}%)")
    
    if not valid_mask.any():
        return np.zeros((*depth_norm.shape, 3)), 0.0
    
    # Visualize valid depth
    valid_depth = depth_norm[valid_mask]
    vmin, vmax = np.percentile(valid_depth, [2, 98])
    
    depth_viz = np.zeros_like(depth_norm)
    if vmax > vmin:
        depth_viz[valid_mask] = np.clip((depth_norm[valid_mask] - vmin) / (vmax - vmin), 0, 1)
    
    cmap = plt.get_cmap(colormap)
    rgb_viz = cmap(depth_viz)[:, :, :3]
    
    if coverage < 98:
        rgb_viz[~valid_mask] = [0.1, 0.1, 0.1]
    
    return rgb_viz, coverage

def denormalize_rgb(rgb_norm: np.ndarray, mean=None, std=None):
    """Denormalize RGB (ImageNet stats by default)."""
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    
    rgb = rgb_norm * std + mean
    return np.clip(rgb, 0, 1)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=Path('configs/data.yaml'), help='Path to config YAML')
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
    ap.add_argument('--depth-colormap', default='magma', help='Colormap for depth visualization')
    ap.add_argument('--denormalize', action='store_true', help='Denormalize RGB using stats from config')
    return ap.parse_args()


def main():
    args = parse_args()
    
    # Load config
    if args.config.exists():
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}
    
    # Extract normalization parameters from config
    norm_cfg = cfg.get('image', {}).get('normalize', {})
    rgb_mean = norm_cfg.get('mean', [0.485, 0.456, 0.406, 0.5])[:3]  # Take only RGB channels
    rgb_std = norm_cfg.get('std', [0.229, 0.224, 0.225, 0.25])[:3]
    
    if args.dataset is None or args.pred_dir is None:
        raise ValueError("--dataset and --pred-dir are required")
    
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

    for idx, stem in enumerate(stems):
        img, gt, meta = load_npz(args.processed_root, args.dataset, args.split, stem)
        pred = load_pred(args.pred_dir, stem)
        # make sure shapes match for visualization
        if pred.shape != gt.shape:
            from scipy.ndimage import zoom
            pred = zoom(pred, (gt.shape[0]/pred.shape[0], gt.shape[1]/pred.shape[1]), order=1)
        rgb = img[..., :3]
        depth = img[..., 3]
        # normalize RGB for display
        if args.denormalize:
            rgb_disp = denormalize_rgb(rgb, mean=rgb_mean, std=rgb_std)
        else:
            rgb_disp = rgb.astype(float)
            if rgb_disp.max() > 1.0:
                rgb_disp /= 255.0
            # Handle normalized data
            if rgb_disp.min() < 0:
                rgb_disp = np.clip((rgb_disp + 3) / 6, 0, 1)
        if args.rgb_robust:
            p1 = np.percentile(rgb_disp, 1)
            p99 = np.percentile(rgb_disp, 99)
            if p99 > p1:
                rgb_disp = np.clip((rgb_disp - p1) / (p99 - p1), 0, 1)

        # Visualize depth (handle normalized depth)
        debug = (idx == 0)
        if depth.min() < -1.0:  # Normalized depth
            depth_viz, coverage = visualize_normalized_depth(depth, args.depth_colormap, debug=debug)
        else:  # Legacy unnormalized depth
            depth_viz = depth
            coverage = 100.0

        fig, axs = plt.subplots(1, 4, figsize=(14,4))
        axs[0].imshow(np.clip(rgb_disp,0,1))
        title = f"RGB {stem}"
        if meta and 'source' in meta:
            title += f"\n({meta['source']})"
        axs[0].set_title(title)
        axs[0].axis('off')
        if isinstance(depth_viz, np.ndarray) and depth_viz.ndim == 3:
            axs[1].imshow(depth_viz)
            depth_title = f"Depth ({coverage:.1f}%)" if coverage < 100 else "Depth"
        else:
            axs[1].imshow(depth_viz, cmap=args.depth_colormap)
            depth_title = "Depth"
        axs[1].set_title(depth_title)
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