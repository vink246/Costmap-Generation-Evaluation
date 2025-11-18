import os
import math
import yaml
import numpy as np
import cv2
from scipy import ndimage

# Utilities for RGB+D -> BEV costmap generation

class Config:
    def __init__(self, d: dict):
        self.d = d
    @staticmethod
    def from_yaml(path: str):
        with open(path, 'r') as f:
            return Config(yaml.safe_load(f))
    def __getitem__(self, k):
        return self.d[k]


def normalize_rgbd(img_rgbd: np.ndarray, mean, std, max_depth=None):
    """Normalize RGB+D image with per-channel mean/std."""
    assert img_rgbd.shape[-1] in (1, 4), "Expected D or RGB+D"
    img = img_rgbd.astype(np.float32)
    # scale RGB channels
    if img.shape[-1] >= 3:
        img[...,:3] /= 255.0
    # scale depth channel
    if img.shape[-1] == 4 and max_depth is not None:
        img[...,3] = np.clip(img[...,3] / max_depth, 0.0, 1.0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (img - mean) / std


def resize_image(img: np.ndarray, size_hw):
    h, w = size_hw
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def points_to_bev(points_xyz: np.ndarray, roi, grid_hw, max_depth_m=30.0, obstacle_height_thresh=0.2):
    """
    Project 3D points to BEV occupancy.
    points_xyz: (N,3) in robot local frame (x forward, y left, z up)
    roi: {width_m, height_m}
    grid_hw: (H,W) costmap resolution
    Returns occupancy grid in [0,1].
    """
    H, W = grid_hw
    width_m = roi['width_m']
    height_m = roi['height_m']

    # Filter points within ROI and depth range
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    mask = (x >= 0) & (x <= height_m) & (np.abs(y) <= width_m/2) & (np.abs(z) <= max_depth_m)
    pts = points_xyz[mask]

    # Map x (forward) to row, y (left) to col
    meters_per_cell_y = width_m / W
    meters_per_cell_x = height_m / H
    rows = np.clip((pts[:,0] / meters_per_cell_x).astype(np.int32), 0, H-1)
    cols = np.clip(((pts[:,1] + width_m/2) / meters_per_cell_y).astype(np.int32), 0, W-1)

    occ = np.zeros((H, W), dtype=np.float32)
    # Occupancy based on height threshold
    occ[rows, cols] = np.maximum(occ[rows, cols], (pts[:,2] > obstacle_height_thresh).astype(np.float32))
    return occ


def dilate_costmap(costmap: np.ndarray, radius_cells: int):
    if radius_cells <= 0:
        return costmap
    struct = ndimage.generate_binary_structure(2, 1)
    struct = ndimage.iterate_structure(struct, radius_cells)
    dilated = ndimage.binary_dilation(costmap > 0.5, structure=struct)
    return np.where(dilated, 1.0, costmap)


def continuous_cost(costmap_occ: np.ndarray):
    """Simple continuous cost from occupancy via distance transform."""
    free = costmap_occ < 0.5
    dist = ndimage.distance_transform_edt(free)
    dist = dist / (dist.max() + 1e-6)
    return 1.0 - dist


def depth_to_points(depth: np.ndarray, K: np.ndarray):
    """Convert depth image (H,W) with intrinsics K to point cloud in camera frame."""
    H, W = depth.shape
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    z = depth
    x = (uu - K[0,2]) * z / K[0,0]
    y = (vv - K[1,2]) * z / K[1,1]
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return pts


def transform_points(pts: np.ndarray, R: np.ndarray, t: np.ndarray):
    return (pts @ R.T) + t[None, :]


def save_pair_npz(out_path: str, image_rgbd: np.ndarray, costmap: np.ndarray, meta: dict):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, image=image_rgbd.astype(np.float32), costmap=costmap.astype(np.float32), meta=np.array([meta], dtype=object))


def make_costmap_from_depth(depth: np.ndarray, K: np.ndarray, roi_cfg: dict, grid_hw=(64,64), dilation_radius=2, max_depth_m=30.0, obstacle_height_thresh=0.2, T_cam_to_robot=None):
    pts_cam = depth_to_points(depth, K)
    if T_cam_to_robot is not None:
        R = T_cam_to_robot[:3, :3]
        t = T_cam_to_robot[:3, 3]
        pts_robot = transform_points(pts_cam, R, t)
    else:
        pts_robot = pts_cam  # assume camera aligned w/ robot
    occ = points_to_bev(pts_robot, roi_cfg, grid_hw, max_depth_m=max_depth_m, obstacle_height_thresh=obstacle_height_thresh)
    occ = dilate_costmap(occ, dilation_radius)
    cont = continuous_cost(occ)
    return cont
