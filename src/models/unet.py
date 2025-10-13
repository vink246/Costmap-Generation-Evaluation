import torch
import torch.nn as nn


class UNet(nn.Module):
    """
    Placeholder U-Net model (CNN encoder-decoder).
    TODO: Implement the actual layers/forward.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_channels: int = 32, **kwargs):
        super().__init__()
        # TODO: implement U-Net modules
        self._dummy = nn.Identity()
        self._out = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: replace with actual forward
        return x[:, :1]  # dummy shape passthrough for wiring
