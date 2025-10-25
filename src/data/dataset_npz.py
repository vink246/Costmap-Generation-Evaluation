import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CostmapPairsNPZ(Dataset):
    """
    Torch dataset for .npz pairs produced by scripts/build_pairs.py.
    Each sample returns:
      - image: Tensor (4,H,W) float32 (already normalized)
      - costmap: Tensor (1,64,64) float32 in [0,1]
    """

    def __init__(self, root: str, split: str = 'train', dataset: str = 'nyu'):
        # root is typically 'data/processed'
        self.root = root
        self.split = split
        self.dataset = dataset
        pattern = os.path.join('src', root, dataset, split, '*.npz')
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f'No NPZ found under {root}/{dataset}/{split}. Did you run build_pairs?')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)
        img = data['image'].astype(np.float32)       # (H,W,4)
        cm = data['costmap'].astype(np.float32)      # (64,64)
        # Convert to torch tensors
        img_t = torch.from_numpy(img).permute(2, 0, 1)  # (4,H,W)
        cm_t = torch.from_numpy(cm).unsqueeze(0)        # (1,64,64)
        return img_t, cm_t
