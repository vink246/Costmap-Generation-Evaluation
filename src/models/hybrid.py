import torch
import torch.nn as nn


class HybridCNNTransformer(nn.Module):
    """
    Placeholder hybrid model: CNN encoder with mid-level transformer refinement + conv decoder.
    - Refine mid-level features with a small transformer block.
    TODO: Implement CNN stem, transformer block(s), and upsampling decoder.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_channels: int = 32, transformer_dim: int = 256, transformer_depth: int = 2, **kwargs):
        super().__init__()
        # TODO: implement hybrid architecture
        self._dummy = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: replace with real pipeline
        return x[:, :1]
