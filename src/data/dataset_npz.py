import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CostmapPairsNPZ(Dataset):
    """
    Torch dataset for .npz pairs produced by scripts/build_pairs.py.
    Each sample returns:
      - image: Tensor (C,H,W) float32 (already normalized)
      - costmap: Tensor (1,64,64) float32 in [0,1]
    """

    def __init__(self, root: str, split: str = 'train', dataset: str = 'nyu', channels: str = 'rgbd'):
        # root is typically 'data/processed'
        self.root = root
        self.split = split
        self.dataset = dataset
        self.channels = channels  # 'rgbd' (default) or 'rgb'
        # Expect root to be a path like 'data/processed'; do not prefix with src
        pattern = os.path.join(self.root, self.dataset, self.split, '*.npz')
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f'No NPZ found under {self.root}/{self.dataset}/{self.split}. Did you run build_pairs?')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)
        img = data['image'].astype(np.float32)       # (H,W,4) expected
        cm = data['costmap'].astype(np.float32)      # (64,64)
        # select channels per config
        if self.channels == 'rgb':
            if img.shape[-1] >= 3:
                img = img[..., :3]
            else:
                raise ValueError(f"RGB mode requested but image has shape {img.shape}")
        # else keep all 4 channels (RGBD)
        # convert to torch tensors
        img_t = torch.from_numpy(img).permute(2, 0, 1)  # (C,H,W)
        cm_t = torch.from_numpy(cm).unsqueeze(0)        # (1,64,64)
        return img_t, cm_t
