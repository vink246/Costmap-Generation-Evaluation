import torch
import torch.nn as nn
from src.models.unet_parts import *

class UNet(nn.Module):
    """
    U-Net for costmap generation from RGB+D images.
    Input: (batch, 4, 256, 256) - RGB+D
    Output: (batch, 1, 64, 64) - costmap
    
    Modified from the U-Net implementation found here:
    https://github.com/milesial/Pytorch-UNet
    to output at 64x64 resolution instead of full input size.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_channels: int = 32, bilinear: bool = True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear

        # Encoder: 256 -> 128 -> 64 -> 32 -> 16
        bc = base_channels
        self.inc = DoubleConv(in_channels, bc)
        self.down1 = Down(bc, bc * 2)
        self.down2 = Down(bc * 2, bc * 4)
        self.down3 = Down(bc * 4, bc * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(bc * 8, bc * 16 // factor)
        
        # Decoder: 16 -> 32 -> 64 (stop here for 64x64 output)
        self.up1 = Up(bc * 16, bc * 8 // factor, bilinear)
        self.up2 = Up(bc * 8, bc * 4 // factor, bilinear)
        
        # Output convolution
        self.outc = OutConv(bc * 4 // factor, out_channels)

    def forward(self, x):
        """
        Args:
            x: (batch, 4, 256, 256) RGB+D input
        Returns:
            (batch, 1, 64, 64) costmap predictions (logits)
        """
        # Encoder with skip connections
        x1 = self.inc(x)      # (B, bc, 256, 256)
        x2 = self.down1(x1)   # (B, bc*2, 128, 128)
        x3 = self.down2(x2)   # (B, bc*4, 64, 64)
        x4 = self.down3(x3)   # (B, bc*8, 32, 32)
        x5 = self.down4(x4)   # (B, bc*16//factor, 16, 16)
        
        # Decoder with skip connections - stop at 64x64
        x = self.up1(x5, x4)  # (B, bc*8//factor, 32, 32)
        x = self.up2(x, x3)   # (B, bc*4//factor, 64, 64)
        
        # Final output
        logits = self.outc(x) # (B, 1, 64, 64)
        return logits

    def use_checkpointing(self):
        """Enable gradient checkpointing to save memory during training."""
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.outc = torch.utils.checkpoint(self.outc)
