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

def velodyne_to_depth_image(pts: np.ndarray, K: np.ndarray, H: int, W: int, 
                            max_depth_m: float, original_size=(375, 1242)):
    """
    Project LiDAR points (N,3) to a depth image of shape (H, W) using camera intrinsics K.
    
    KITTI coordinate systems:
    - Velodyne frame: x=forward, y=left, z=up
    - Camera frame: x=right, y=down, z=forward
    """
    # Scale intrinsics to match target resolution
    orig_h, orig_w = original_size
    
    # FIXED: Scale only the relevant parameters
    K_scaled = K.copy()
    K_scaled[0, 0] *= (W / orig_w)  # fx
    K_scaled[0, 2] *= (W / orig_w)  # cx
    K_scaled[1, 1] *= (H / orig_h)  # fy
    K_scaled[1, 2] *= (H / orig_h)  # cy
    
    print(f"\n=== Intrinsics Check ===")
    print(f"Original K:\n{K}")
    print(f"Scaled K for {W}x{H}:\n{K_scaled}")
    print(f"Expected: fx≈{721.5*(W/orig_w):.1f}, cx≈{W/2:.1f}, fy≈{721.5*(H/orig_h):.1f}, cy≈{H/2:.1f}")
    print(f"Actual:   fx={K_scaled[0,0]:.1f}, cx={K_scaled[0,2]:.1f}, fy={K_scaled[1,1]:.1f}, cy={K_scaled[1,2]:.1f}")
    
    x_vel, y_vel, z_vel = pts[:,0], pts[:,1], pts[:,2]
    
    # Transform from velodyne to camera frame
    x_cam = -y_vel
    y_cam = -z_vel
    z_cam = x_vel

    # Only keep points in front of camera
    valid = z_cam > 0
    x_cam, y_cam, z_cam = x_cam[valid], y_cam[valid], z_cam[valid]

    # Project to image plane using SCALED intrinsics
    u = K_scaled[0,0]*x_cam/z_cam + K_scaled[0,2]
    v = K_scaled[1,1]*y_cam/z_cam + K_scaled[1,2]
    
    # Quick sanity check
    print(f"\nProjection sanity check (first 5 points):")
    for i in range(min(5, len(u))):
        print(f"  Point {i}: depth={z_cam[i]:.2f}m, x_cam={x_cam[i]:.2f}, y_cam={y_cam[i]:.2f} -> u={u[i]:.1f}, v={v[i]:.1f}")
    
    # Convert to integers for indexing
    u_int = u.astype(np.int32)
    v_int = v.astype(np.int32)

    # Keep points within image bounds and within max depth
    valid_img = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (z_cam <= max_depth_m)
    u_int = u_int[valid_img]
    v_int = v_int[valid_img]
    z_cam = z_cam[valid_img]

    # Initialize depth image
    depth_img = np.zeros((H, W), dtype=np.float32)

    # Vectorized: keep nearest point per pixel
    sort_idx = np.argsort(-z_cam)  # Sort by depth (farthest first)
    depth_img[v_int[sort_idx], u_int[sort_idx]] = z_cam[sort_idx]

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
                    depth = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
                    
                    # Handle different depth formats
                    if depth.max() > 1000:
                        depth = depth / 1000.0  # mm to meters
                    elif depth.max() > 100:
                        depth = depth / 256.0  # KITTI format (depth * 256)
                    
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
                    # Generate depth image from velodyne points
                    depth_res = velodyne_to_depth_image(pts, K, H_resize, W_resize, max_depth_m)
                    
                    # Generate costmap from velodyne points directly
                    # Assume velodyne frame aligns w/ robot local (KITTI: x forward, y left, z up)
                    occ = points_to_bev(pts, roi, grid, max_depth_m=max_depth_m, obstacle_height_thresh=float(cfg['costmap']['obstacle_threshold_m']))
                    occ = dilate_costmap(occ, dilation)
                    costmap = continuous_cost(occ)

                    img_res = resize_image(img, (H_resize, W_resize))
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