import os
import argparse
import numpy as np
import cv2
import yaml

"""
Prepare NYU Depth v2 data (from Kaggle mirror):

Input (detected automatically under --root):
- data/raw/nyu_depth_v2/nyu_data/data/
  Common Kaggle mirrors include .npy arrays like:
  - images.npy or nyu_images.npy: shape (N, H, W, 3), uint8
  - depths.npy or nyu_depths.npy: shape (N, H, W), float32/uint16 in meters
  Or pre-extracted png folders: rgb/*.png, depth/*.png

Output structure:
- data/raw/nyu_depth_v2/subset_train/<sample_id>/rgb/<sample_id>.png
- data/raw/nyu_depth_v2/subset_train/<sample_id>/depth/<sample_id>.png
- data/raw/nyu_depth_v2/subset_val/<sample_id>/...

Split: 80/20 (train/val)
Depth PNGs are saved as 16-bit PNG with millimeter scale to preserve precision.
"""


def find_nyu_arrays(data_dir):
    candidates_img = [
        'images.npy', 'nyu_images.npy', 'NYU_images.npy', 'NYU_Depth_V2_images.npy'
    ]
    candidates_depth = [
        'depths.npy', 'nyu_depths.npy', 'NYU_depths.npy', 'NYU_Depth_V2_depths.npy', 'labels.npy'
    ]
    img_path = None
    depth_path = None
    for c in candidates_img:
        p = os.path.join(data_dir, c)
        if os.path.isfile(p):
            img_path = p
            break
    for c in candidates_depth:
        p = os.path.join(data_dir, c)
        if os.path.isfile(p):
            depth_path = p
            break
    return img_path, depth_path


def save_depth_png_mm(depth_m: np.ndarray, out_path: str):
    """Save depth in meters as 16-bit PNG in millimeters."""
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(out_path, depth_mm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/raw/nyu_depth_v2', help='NYU raw root folder')
    ap.add_argument('--source', default='nyu_data/data', help='Relative path under root where Kaggle content resides')
    ap.add_argument('--train_ratio', type=float, default=0.8)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--max_samples', type=int, default=0, help='Optional cap for quick prep (0 = all)')
    args = ap.parse_args()

    src_dir = os.path.join(args.root, args.source)
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Source dir not found: {src_dir}")

    # Case A: Kaggle layout with nyu2_train/nyu2_test and *_colors.png / *_depth.png
    pairs = []
    nyu2_train_dir = os.path.join(src_dir, 'nyu2_train')
    nyu2_test_dir = os.path.join(src_dir, 'nyu2_test')
    all_dirs = []
    if os.path.isdir(nyu2_train_dir):
        all_dirs.append(nyu2_train_dir)
    if os.path.isdir(nyu2_test_dir):
        all_dirs.append(nyu2_test_dir)
    if all_dirs:
        # Combine all samples from both folders
        files = []
        for d in all_dirs:
            files.extend([os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith('.png')])
        # Build pairs by matching *_colors.png w/ *_depth.png
        depth_map = {}
        color_map = {}
        for full in files:
            f = os.path.basename(full)
            name = os.path.splitext(f)[0]
            if name.endswith('_depth'):
                base = name[:-6]  # remove '_depth'
                depth_map[base] = full
            elif name.endswith('_colors'):
                base = name[:-7]  # remove '_colors'
                color_map[base] = full
        for base in sorted(color_map.keys()):
            if base in depth_map:
                pairs.append((color_map[base], depth_map[base], base))
    else:
        # Case B: pre-extracted PNGs already exist in generic rgb/depth dirs
        png_rgb_dir = None
        png_depth_dir = None
        for d in ['rgb', 'images']:
            p = os.path.join(src_dir, d)
            if os.path.isdir(p):
                png_rgb_dir = p
                break
        for d in ['depth', 'depths']:
            p = os.path.join(src_dir, d)
            if os.path.isdir(p):
                png_depth_dir = p
                break

        if png_rgb_dir and png_depth_dir:
            # Pair by basename
            rgb_files = sorted([f for f in os.listdir(png_rgb_dir) if f.lower().endswith('.png')])
            depth_files = sorted([f for f in os.listdir(png_depth_dir) if f.lower().endswith('.png')])
            depth_set = {os.path.splitext(f)[0]: f for f in depth_files}
            for rf in rgb_files:
                base = os.path.splitext(rf)[0]
                if base in depth_set:
                    pairs.append((os.path.join(png_rgb_dir, rf), os.path.join(png_depth_dir, depth_set[base]), base))
        else:
            # Case C: npy arrays
            img_npy, depth_npy = find_nyu_arrays(src_dir)
            if not (img_npy and depth_npy):
                raise FileNotFoundError('Could not find NYU images/depth arrays or PNG folders under ' + src_dir)
            images = np.load(img_npy)
            depths = np.load(depth_npy)
            if images.ndim != 4 or images.shape[-1] != 3:
                raise ValueError('Images array must be (N,H,W,3)')
            if depths.ndim != 3:
                raise ValueError('Depths array must be (N,H,W) in meters')
            N = min(images.shape[0], depths.shape[0])
            if args.max_samples > 0:
                N = min(N, args.max_samples)
            # Save temporary extracted PNGs to src_dir/png_extract to unify flow
            tmp_rgb_dir = os.path.join(src_dir, 'rgb')
            tmp_depth_dir = os.path.join(src_dir, 'depth')
            os.makedirs(tmp_rgb_dir, exist_ok=True)
            os.makedirs(tmp_depth_dir, exist_ok=True)
            for i in range(N):
                base = f'{i:06d}'
                rgb = images[i]
                depth_m = depths[i].astype(np.float32)
                cv2.imwrite(os.path.join(tmp_rgb_dir, base + '.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                save_depth_png_mm(depth_m, os.path.join(tmp_depth_dir, base + '.png'))
                pairs.append((os.path.join(tmp_rgb_dir, base + '.png'), os.path.join(tmp_depth_dir, base + '.png'), base))

    # Split 80/20
    n = len(pairs)
    n_train = max(1, int(n * args.train_ratio))
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    # Write output structure
    out_train_root = os.path.join(args.root, 'subset_train')
    out_val_root = os.path.join(args.root, 'subset_val')
    os.makedirs(out_train_root, exist_ok=True)
    os.makedirs(out_val_root, exist_ok=True)

    def write_sample(root_dir, rgb_src, depth_src, base):
        sample_dir = os.path.join(root_dir, base)
        rgb_dir = os.path.join(sample_dir, 'rgb')
        depth_dir = os.path.join(sample_dir, 'depth')
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        rgb_dst = os.path.join(rgb_dir, base + '.png')
        depth_dst = os.path.join(depth_dir, base + '.png')
        if rgb_src != rgb_dst:
            cv2.imwrite(rgb_dst, cv2.imread(rgb_src))
        if depth_src != depth_dst:
            cv2.imwrite(depth_dst, cv2.imread(depth_src, cv2.IMREAD_UNCHANGED))

    for rgb_src, depth_src, base in train_pairs:
        write_sample(out_train_root, rgb_src, depth_src, base)
    for rgb_src, depth_src, base in val_pairs:
        write_sample(out_val_root, rgb_src, depth_src, base)

    print('NYU prepared: train', len(train_pairs), 'val', len(val_pairs))
    print('Output root:', args.root)


if __name__ == '__main__':
    main()
