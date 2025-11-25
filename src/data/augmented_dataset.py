"""Dataset wrapper that applies perturbations to images for robustness testing.

Applies brightness adjustment, Gaussian blur, and perspective shift transformations
to input images while keeping costmap labels unchanged.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from typing import Optional, Tuple
import random


class AugmentedCostmapPairs(Dataset):
    """
    Wrapper around CostmapPairsNPZ that applies perturbations to images.
    
    Supports:
    - Brightness adjustment (multiplicative factor)
    - Gaussian blur (on RGB channels)
    - Perspective shift (small affine transformation)
    
    Labels (costmaps) are not modified.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        brightness_factor: Optional[float] = None,
        blur_sigma: Optional[float] = None,
        perspective_params: Optional[dict] = None,
        normalization_mean: Optional[np.ndarray] = None,
        normalization_std: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ):
        """
        Args:
            base_dataset: Base dataset to wrap (should return (image, costmap) tuples)
            brightness_factor: Brightness multiplier (e.g., 0.7 for darker, 1.3 for brighter)
                              If None, no brightness adjustment
            blur_sigma: Gaussian blur sigma for RGB channels. If None, no blur
            perspective_params: Dict with keys 'rotation_deg', 'translation_px' for affine transform
                               If None, no perspective shift
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        self.brightness_factor = brightness_factor
        self.blur_sigma = blur_sigma
        self.perspective_params = perspective_params
        # Default normalization (RGBD ImageNet-style)
        self.norm_mean = normalization_mean if normalization_mean is not None else np.array([0.485, 0.456, 0.406, 0.5])
        self.norm_std = normalization_std if normalization_std is not None else np.array([0.229, 0.224, 0.225, 0.25])
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def apply_brightness(self, img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Apply brightness adjustment to RGB channels.
        
        Denormalizes, applies brightness, clamps, then renormalizes.
        """
        if self.brightness_factor is None:
            return img
        
        # Denormalize RGB channels (assumes first 3 channels are RGB)
        img_denorm = img.copy()
        if img_denorm.shape[-1] >= 3:
            for c in range(3):
                img_denorm[..., c] = img_denorm[..., c] * std[c] + mean[c]
            
            # Apply brightness (multiplicative)
            img_denorm[..., :3] *= self.brightness_factor
            
            # Clamp to [0, 1] (assuming original was in [0, 1] after normalization)
            img_denorm[..., :3] = np.clip(img_denorm[..., :3], 0.0, 1.0)
            
            # Renormalize
            for c in range(3):
                img_denorm[..., c] = (img_denorm[..., c] - mean[c]) / std[c]
        
        return img_denorm
    
    def apply_blur(self, img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to RGB channels.
        
        Denormalizes, applies blur, then renormalizes.
        """
        if self.blur_sigma is None or self.blur_sigma <= 0:
            return img
        
        img_blurred = img.copy()
        if img_blurred.shape[-1] >= 3:
            # Denormalize RGB
            for c in range(3):
                img_blurred[..., c] = img_blurred[..., c] * std[c] + mean[c]
            
            # Apply Gaussian blur to RGB channels
            # Convert to uint8 for cv2.blur, then back
            for c in range(3):
                channel = img_blurred[..., c]
                channel_uint8 = (channel * 255).astype(np.uint8)
                # Gaussian blur kernel size should be odd
                ksize = int(6 * self.blur_sigma) | 1  # Make odd
                if ksize < 3:
                    ksize = 3
                blurred = cv2.GaussianBlur(channel_uint8, (ksize, ksize), self.blur_sigma)
                img_blurred[..., c] = blurred.astype(np.float32) / 255.0
            
            # Renormalize
            for c in range(3):
                img_blurred[..., c] = (img_blurred[..., c] - mean[c]) / std[c]
        
        return img_blurred
    
    def apply_perspective_shift(self, img: np.ndarray) -> np.ndarray:
        """Apply small affine transformation (rotation + translation).
        
        Note: This is a simplified version. For true perspective shift,
        would need to also transform the costmap correspondingly.
        For robustness testing, we apply small shifts that shouldn't
        drastically change the semantic content.
        """
        if self.perspective_params is None:
            return img
        
        rotation_deg = self.perspective_params.get('rotation_deg', 0)
        translation_px = self.perspective_params.get('translation_px', 0)
        
        if rotation_deg == 0 and translation_px == 0:
            return img
        
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        
        # Create rotation matrix
        M_rot = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
        
        # Add translation
        M_rot[0, 2] += translation_px
        M_rot[1, 2] += translation_px
        
        # Apply transformation to each channel
        img_shifted = np.zeros_like(img)
        for c in range(img.shape[-1]):
            # For normalized images, we need to handle carefully
            # Denormalize, apply transform, renormalize
            channel = img[..., c]
            # Convert to uint8 for transform
            channel_uint8 = ((channel - channel.min()) / (channel.max() - channel.min() + 1e-6) * 255).astype(np.uint8)
            transformed = cv2.warpAffine(channel_uint8, M_rot, (w, h), borderMode=cv2.BORDER_REPLICATE)
            img_shifted[..., c] = transformed.astype(np.float32) / 255.0
        
        return img_shifted
    
    def __getitem__(self, idx):
        img, costmap = self.base_dataset[idx]
        
        # Convert to numpy for augmentation
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
        else:
            img_np = img
        
        # Apply augmentations in order: brightness -> blur -> perspective
        img_aug = img_np.copy()
        img_aug = self.apply_brightness(img_aug, self.norm_mean, self.norm_std)
        img_aug = self.apply_blur(img_aug, self.norm_mean, self.norm_std)
        img_aug = self.apply_perspective_shift(img_aug)
        
        # Convert back to tensor if needed
        if isinstance(img, torch.Tensor):
            img_aug = torch.from_numpy(img_aug).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            costmap = costmap if isinstance(costmap, torch.Tensor) else torch.from_numpy(costmap)
        
        return img_aug, costmap

