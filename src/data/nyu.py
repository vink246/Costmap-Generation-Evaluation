import os
import glob
import numpy as np
import cv2
from .utils import Config, resize_image, normalize_rgbd, make_costmap_from_depth, save_pair_npz

"""
NYU Depth v2 preprocessing: expects dense depth PNG and aligned RGB under config.nyu.root.
Assumes per-sequence intrinsics provided or uses a common fallback.
"""


def load_intrinsics_default():
    # NYU intrinsics for 640x480, will be adjusted by resize
    fx, fy = 575.8, 575.8
    cx, cy = 319.5, 239.5
    return np.array([[fx, 0, cx], [0, fy, cy], [0,0,1]], dtype=np.float32)


def build_pairs(config_path: str, out_dir: str):
    cfg = Config.from_yaml(config_path)
    root = cfg['nyu']['root']
    splits = cfg['nyu']['splits']
    H_resize, W_resize = cfg['image']['resize']
    mean = cfg['image']['normalize']['mean']
    std = cfg['image']['normalize']['std']
    grid = tuple(cfg['costmap']['grid'])
    roi = cfg['roi']
    dilation = int(cfg['costmap']['dilation_radius_cells'])
    max_depth_m = float(cfg['costmap']['max_depth_m_nyu'])

    K = load_intrinsics_default()

    for split_name, folders in splits.items():
        for folder in folders:
            # Try flat layout first: <root>/<folder>/rgb/*.png and depth/*.png
            flat_rgb_glob = os.path.join(root, folder, 'rgb', '*.png')
            rgb_files = sorted(glob.glob(flat_rgb_glob))
            pairs = []

            if rgb_files:
                # Flat layout: pair by basename
                depth_dir = os.path.join(root, folder, 'depth')
                depth_map = {os.path.splitext(os.path.basename(p))[0]: p for p in sorted(glob.glob(os.path.join(depth_dir, '*.png')))}
                for rf in rgb_files:
                    base = os.path.splitext(os.path.basename(rf))[0]
                    df = depth_map.get(base)
                    if df and os.path.isfile(df):
                        pairs.append((rf, df))
            else:
                # Nested layout: <root>/<folder>/<id>/rgb/<id>.png with sibling depth/<id>.png
                nested_rgb_glob = os.path.join(root, folder, '**', 'rgb', '*.png')
                nested_rgbs = sorted(glob.glob(nested_rgb_glob, recursive=True))
                for rf in nested_rgbs:
                    # Replace .../rgb/<file>.png with .../depth/<file>.png
                    if f'{os.sep}rgb{os.sep}' in rf:
                        df = rf.replace(f'{os.sep}rgb{os.sep}', f'{os.sep}depth{os.sep}')
                    else:
                        # Fallback: construct sibling depth path
                        rgb_dir = os.path.dirname(rf)
                        sample_dir = os.path.dirname(rgb_dir)
                        base = os.path.splitext(os.path.basename(rf))[0]
                        df = os.path.join(sample_dir, 'depth', base + '.png')
                    if os.path.isfile(df):
                        pairs.append((rf, df))

            n = len(pairs)
            out_split_dir = os.path.join(out_dir, 'nyu', split_name)
            os.makedirs(out_split_dir, exist_ok=True)
            for i, (img_path, depth_path) in enumerate(pairs):
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
                img_res = resize_image(img, (H_resize, W_resize))
                depth_res = cv2.resize(depth, (W_resize, H_resize), interpolation=cv2.INTER_NEAREST)

                costmap = make_costmap_from_depth(depth_res, K, roi_cfg=roi, grid_hw=grid, dilation_radius=dilation, max_depth_m=max_depth_m)

                rgbd = np.concatenate([img_res, depth_res[...,None]], axis=-1)
                rgbd_norm = rgbd_norm = normalize_rgbd(rgbd, mean, std, max_depth=max_depth_m)

                meta = {
                    'frame': os.path.basename(img_path),
                    'folder': folder,
                    'split': split_name,
                    'K': K.tolist(),
                    'roi': roi,
                    'grid': grid,
                }
                out_file = os.path.join(out_split_dir, f'{i:06d}.npz')
                save_pair_npz(out_file, rgbd_norm, costmap, meta)
            print(f"NYU {folder} {split_name}: wrote {n} pairs to {out_split_dir}")
