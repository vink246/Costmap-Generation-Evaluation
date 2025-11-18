import os
import glob
import json
import numpy as np
import cv2
from .utils import (
    Config,
    resize_image,
    normalize_rgbd,
    make_costmap_from_depth,
    save_pair_npz,
    points_to_bev,
    dilate_costmap,
    continuous_cost,
)

"""
KITTI preprocessing: 
- <drive>/image_02/data/*.png
- <drive>/velodyne_points/data/*.bin (optional if depth available)
- calibration files (intrinsics K, and T_cam_to_robot if needed)
This script focuses on depth + RGB pairing -> user should provide K.
"""


def load_intrinsics(calib_path: str):
    if os.path.exists(calib_path):
        with open(calib_path, 'r') as f:
            K = np.array(json.load(f)['K'], dtype=np.float32)
        return K
    # Fallback: approximate KITTI intrinsics for 1242x375
    fx, fy = 721.5377, 721.5377
    cx, cy = 609.5593, 172.8540
    K = np.array([[fx, 0, cx], [0, fy, cy], [0,0,1]], dtype=np.float32)
    return K

def read_velodyne_bin(bin_path: str):
    """Read KITTI velodyne .bin file. Returns (N,3) XYZ in meters."""
    data = np.fromfile(bin_path, dtype=np.float32)
    if data.size % 4 != 0:
        # Expect x,y,z,reflectance per point
        data = data[: (data.size // 4) * 4]
    pts = data.reshape(-1, 4)[:, :3]
    return pts
def velodyne_to_depth_image(pts: np.ndarray, K: np.ndarray, H: int, W: int, max_depth_m: float):
    """
    Project LiDAR points (N,3) to a depth image of shape (H, W) using camera intrinsics K.
    Points behind camera or beyond max_depth_m are ignored.
    """
    x, y, z = pts[:,0], pts[:,1], pts[:,2]

    # Only keep points in front of camera
    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]

    # Project to image plane
    u = (K[0,0]*x/z + K[0,2]).astype(np.int32)
    v = (K[1,1]*y/z + K[1,2]).astype(np.int32)

    # Clip to image bounds
    u = np.clip(u, 0, W-1)
    v = np.clip(v, 0, H-1)

    # Initialize depth image
    depth_img = np.zeros((H, W), dtype=np.float32)

    # Keep nearest point per pixel
    for ui, vi, zi in zip(u, v, z):
        if zi > max_depth_m:
            continue
        if depth_img[vi, ui] == 0 or zi < depth_img[vi, ui]:
            depth_img[vi, ui] = zi

    return depth_img


def build_pairs(config_path: str, out_dir: str):
    cfg = Config.from_yaml(config_path)
    root = cfg['kitti']['root']
    splits = cfg['kitti']['splits']
    H_resize, W_resize = cfg['image']['resize']
    mean = cfg['image']['normalize']['mean']
    std = cfg['image']['normalize']['std']
    grid = tuple(cfg['costmap']['grid'])
    roi = cfg['roi']
    dilation = int(cfg['costmap']['dilation_radius_cells'])
    max_depth_m = float(cfg['costmap']['max_depth_m_kitti'])

    for split_name, drives in splits.items():
        for drive in drives:
            img_glob = os.path.join(root, drive, 'image_02', 'data', '*.png')
            depth_glob = os.path.join(root, drive, 'depth', 'data', '*.png')  
            lidar_glob = os.path.join(root, drive, 'velodyne_points', 'data', '*.bin')
            calib_file = os.path.join(root, drive, 'calib', 'intrinsics.json')
            K = load_intrinsics(calib_file)

            img_files = sorted(glob.glob(img_glob))
            depth_files = sorted(glob.glob(depth_glob))
            lidar_files = sorted(glob.glob(lidar_glob))

            out_split_dir = os.path.join(out_dir, 'kitti', split_name)
            os.makedirs(out_split_dir, exist_ok=True)

            if len(depth_files) > 0:
                n = min(len(img_files), len(depth_files))
                for i in range(n):
                    img = cv2.cvtColor(cv2.imread(img_files[i]), cv2.COLOR_BGR2RGB)
                    depth = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # mm->m if applicable
                    img_res = resize_image(img, (H_resize, W_resize))
                    depth_res = cv2.resize(depth, (W_resize, H_resize), interpolation=cv2.INTER_NEAREST)

                    costmap = make_costmap_from_depth(depth_res, K, roi_cfg=roi, grid_hw=grid, dilation_radius=dilation, max_depth_m=max_depth_m)

                    rgbd = np.concatenate([img_res, depth_res[...,None]], axis=-1)
                    rgbd_norm = normalize_rgbd(rgbd, mean, std, max_depth=max_depth_m)

                    meta = {
                        'frame': os.path.basename(img_files[i]),
                        'drive': drive,
                        'split': split_name,
                        'K': K.tolist(),
                        'roi': roi,
                        'grid': grid,
                        'source': 'depth',
                    }
                    out_file = os.path.join(out_split_dir, f'{i:06d}.npz')
                    save_pair_npz(out_file, rgbd_norm, costmap, meta)
                print(f"KITTI {drive} {split_name}: wrote {n} depth-based pairs to {out_split_dir}")
            elif len(lidar_files) > 0:
                # Fallback: generate costmap from LiDAR directly, pair by index with images
                n = min(len(img_files), len(lidar_files))
                for i in range(n):
                    img = cv2.cvtColor(cv2.imread(img_files[i]), cv2.COLOR_BGR2RGB)
                    pts = read_velodyne_bin(lidar_files[i])
                    # Assume velodyne frame aligns w/ robot local (KITTI: x forward, y left, z up)
                    occ = points_to_bev(pts, roi, grid, max_depth_m=max_depth_m, obstacle_height_thresh=float(cfg['costmap']['obstacle_threshold_m']))
                    occ = dilate_costmap(occ, dilation)
                    costmap = continuous_cost(occ)

                    img_res = resize_image(img, (H_resize, W_resize))
                    # No depth channel -> pad w/ zeros to create RGB+D
                    # depth_res = np.zeros((H_resize, W_resize), dtype=np.float32)
                    depth_res = velodyne_to_depth_image(pts, K, H_resize, W_resize, max_depth_m)
                    rgbd = np.concatenate([img_res, depth_res[..., None]], axis=-1)
                    rgbd_norm = normalize_rgbd(rgbd, mean, std, max_depth=max_depth_m)

                    meta = {
                        'frame': os.path.basename(img_files[i]),
                        'drive': drive,
                        'split': split_name,
                        'K': K.tolist(),
                        'roi': roi,
                        'grid': grid,
                        'source': 'lidar',
                    }
                    out_file = os.path.join(out_split_dir, f'{i:06d}.npz')
                    save_pair_npz(out_file, rgbd_norm, costmap, meta)
                print(f"KITTI {drive} {split_name}: wrote {n} LiDAR-based pairs to {out_split_dir}!")
            else:
                print(f"KITTI {drive} {split_name}: no depth or LiDAR files found! Skipped ...")
