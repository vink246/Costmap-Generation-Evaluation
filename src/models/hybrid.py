# src/models/hybrid.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Utility blocks ----------
class ConvBlock(nn.Module):
    """
    Two 3x3 convs with BN+ReLU. Optional downsample by stride=2 on the first conv.
    """
    def __init__(self, cin, cout, down=False):
        super().__init__()
        stride = 2 if down else 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpConvBlock(nn.Module):
    """
    Simple upsampling (bilinear) followed by a 3x3 conv block to refine features.
    """
    def __init__(self, cin, cout):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, size=None):
        if size is not None:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.refine(x)


# ---------- Transformer bottleneck ----------
class TransformerBottleneck(nn.Module):
    """
    Small ViT-style encoder over spatial tokens:
    - Flatten BxCxHxW -> Bx(H*W)xC
    - Positional encoding
    - 2-4 layers of MHSA + MLP (torch.nn.TransformerEncoder)
    - Reshape back to BxCxHxW
    """
    def __init__(self, channels, num_layers=3, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.channels = channels

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=int(channels * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable 2D positional embeddings (added to tokens)
        self.pos_embed = None  # created at runtime for the current H*W if we want fixed size
        self.register_parameter("pos_table", None)  # kept for clarity; we build per-shape pe on the fly

    @staticmethod
    def build_2d_sincos_pos_embed(h, w, c, device):
        """
        Sinusoidal 2D positional embedding (H*W, C), split half for x and half for y.
        Works without learnable params, stable across image sizes.
        """
        import math
        assert c % 2 == 0, "channels for sinusoidal PE should be even"
        y = torch.arange(h, device=device).unsqueeze(1).repeat(1, w)
        x = torch.arange(w, device=device).unsqueeze(0).repeat(h, 1)
        omega = torch.arange(c // 2, device=device) / (c // 2)
        omega = 1.0 / (10000 ** omega)  # (c/2,)
        out = torch.zeros(h, w, c, device=device)
        out[..., 0:c//2] = torch.sin(x.unsqueeze(-1) * omega)  # sin over x
        out[..., c//2:] = torch.cos(y.unsqueeze(-1) * omega)   # cos over y
        return out.view(h * w, c)  # (H*W, C)

    def forward(self, x):
        """
        x: B x C x H x W
        """
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # B x (H*W) x C

        # positional encodings
        pe = self.build_2d_sincos_pos_embed(H, W, C, x.device)  # (H*W, C)
        pe = pe.unsqueeze(0).expand(B, -1, -1)                  # B x (H*W) x C
        tokens = tokens + pe

        tokens = self.encoder(tokens)  # B x (H*W) x C
        y = tokens.transpose(1, 2).reshape(B, C, H, W)  # back to BxCxHxW
        return y


# ---------- Hybrid model ----------
class HybridCNNTransformer(nn.Module):
    """
    Encoder: 3-4 CNN stages (local features) -> Transformer bottleneck (global context)
    Decoder: lightweight upsampling with skip fusion, output 64x64 costmap in [0,1]
    """
    def __init__(
        self,
        in_channels=4,        # RGBD by default; set to 3 for RGB-only in config
        base=64,              # base channel width
        tf_layers=3,
        tf_heads=4,
        tf_mlp=2.0,
        tf_dropout=0.0,
        out_size=64,          # final costmap size (HxW)
        out_channels=1,       # 1 = continuous costmap; add more heads if needed
        add_uncertainty=False # optional extra output head
    ):
        super().__init__()
        self.out_size = out_size
        self.add_uncertainty = add_uncertainty

        # Encoder
        self.enc1 = ConvBlock(in_channels, base, down=False)      # 256x256
        self.enc2 = ConvBlock(base, base * 2, down=True)          # 128x128
        self.enc3 = ConvBlock(base * 2, base * 4, down=True)      # 64x64
        self.enc4 = ConvBlock(base * 4, base * 4, down=True)      # 32x32 (bottleneck input)

        # Transformer bottleneck at 32x32
        self.tf = TransformerBottleneck(
            channels=base * 4,
            num_layers=tf_layers,
            num_heads=tf_heads,
            mlp_ratio=tf_mlp,
            dropout=tf_dropout
        )

        # Decoder: fuse transformer output with the 64x64 skip (from enc3)
        self.up1 = UpConvBlock(cin=base * 8, cout=base * 2)       # concat -> up to 64x64
        self.up2 = UpConvBlock(cin=base * 2, cout=base)           # up to 128x128 (for refinement)
        self.head = nn.Conv2d(base, out_channels, kernel_size=1)  # final cost head

        # Optional uncertainty head (parallel)
        if add_uncertainty:
            self.unc_head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)                    # 256
        s2 = self.enc2(s1)                   # 128
        s3 = self.enc3(s2)                   # 64
        b  = self.enc4(s3)                   # 32

        # Transformer bottleneck
        b = self.tf(b)                       # 32 with global context

        # Decoder: up to 64, fuse with s3 (skip), refine
        u  = F.interpolate(b, size=s3.shape[-2:], mode="bilinear", align_corners=False)  # 32->64
        u  = torch.cat([u, s3], dim=1)  # channel concat
        u  = self.up1(u, size=(64, 64))       # ensure 64x64
        u  = self.up2(u)                      # 64->128 (refinement)

        # Force final output to the target costmap size (e.g., 64x64)
        u  = F.interpolate(u, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        cost = torch.sigmoid(self.head(u))

        if self.add_uncertainty:
            # Predict log-variance or sigma; here simple sigma in [0, +inf) with softplus
            sigma = F.softplus(self.unc_head(u))
            return cost, sigma

        return cost