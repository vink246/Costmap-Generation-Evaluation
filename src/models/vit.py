import torch
import torch.nn as nn


class ViT(nn.Module):
    """
    Placeholder Vision Transformer-based encoder with conv upsampling decoder.
    - Transformer encoder (e.g., ViT) -> lightweight decoder to 64x64 map.
    TODO: Implement actual transformer blocks (use timm or custom) and decoder.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 1, embed_dim: int = 256, depth: int = 8, num_heads: int = 8, **kwargs):
        super().__init__()
        # TODO: implement ViT backbone and decoder
        self._dummy = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: replace with real pipeline; ensure output spatial size can be resized to 64x64
        return x[:, :1]
