import torch.nn as nn
import math
import torch 
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
    def forward(self, x):
        return F.relu(x + self.block(x))

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=4, embed_dim=256, patch_size=16):
        super().__init__()
        self.patch_size=patch_size
        self.proj= nn.Conv2d(in_channels, 
                    embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    def forward(self, x):
        #x: [B,C,H,W]
        x= self.proj(x) # [B, D, H/P, W/P]
        B, D, H_p, W_p = x.shape 
        x = x.flatten(2).transpose(1, 2) # [B, N, D] where N = H_p * W_p
        return x, (H_p, W_p) 
    
def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    """Generate 2D sine-cosine positional embedding."""
    grid_y = torch.arange(grid_h, dtype=torch.float32)
    grid_x = torch.arange(grid_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_y, grid_x, indexing='ij')
    grid = torch.stack(grid, dim=0)  # [2, H, W]

    # Flatten and compute frequencies
    grid = grid.reshape(2, -1).T  # [H*W, 2]
    assert embed_dim % 4 == 0
    dim_part = embed_dim // 4
    omega = 1. / (10000 ** (torch.arange(dim_part) / dim_part))

    pos_y = grid[:, 0:1] * omega
    pos_x = grid[:, 1:2] * omega
    pos = torch.cat([
        torch.sin(pos_y), torch.cos(pos_y),
        torch.sin(pos_x), torch.cos(pos_x)
    ], dim=1)  # [H*W, D]
    return pos

class PositionalEmbedding(nn.Module):
    """Learnable positional embedding initialized with sin-cos pattern."""
    def __init__(self, embed_dim, grid_h, grid_w):
        super().__init__()
        pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w)
        self.pos_embed = nn.Parameter(pos_embed.unsqueeze(0))  # [1, N, D]

    def forward(self, x):
        return x + self.pos_embed

class TransformerBlock(nn.Module):
    """Single Transformer Encoder Block (Pre-LayerNorm style)."""
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Multi-head self-attention with residual
        x_res = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)  # self-attention
        x = x_res + attn_out

        # Feedforward (MLP) with residual
        x_res = x
        x = self.norm2(x)
        x = x_res + self.mlp(x)
        return x

class TransformerEncoder(nn.Module):
    """Stack of Transformer blocks."""
    def __init__(self, embed_dim=256, num_heads=8, depth=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)  # final normalization

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        x = self.norm(x)
        return x
    
class ConvDecoder(nn.Module):
    def __init__(self, embed_dim=256, out_channels=1, target_size=64, in_channels=4):
        super().__init__()
        self.target_size = target_size

        # Low-level features from input for skip connection
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Decoder: progressively upsample transformer features to target
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1),
            ResBlock(256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16 -> 32

            nn.Conv2d(256, 128, 3, padding=1),
            ResBlock(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32 -> 64

            nn.Conv2d(128, 64, 3, padding=1),
            ResBlock(64)
        )

        # Refine and output
        self.refine_head = nn.Sequential(
            nn.Conv2d(64 + 64, 32, 3, padding=1),  # concatenate skip features
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1)  # output logits
        )

    def forward(self, x_input, transformer_features):
        """
        x_input: raw input image [B, C, H, W] for skip
        transformer_features: encoded features from ViT [B, D, H_p, W_p]
        """
        # Low-level features for skip connection
        low_feat = self.low_level_conv(x_input)  # [B, 64, H, W]
        low_feat = F.interpolate(low_feat, size=(self.target_size, self.target_size),
                                mode='bilinear', align_corners=False)

        # Upsample transformer features to initial decoder resolution
        x = F.interpolate(transformer_features, size=(self.target_size // 4, self.target_size // 4),
                        mode='bilinear', align_corners=False)
        x = self.decoder(x)  # upsample to target_size inside decoder

        # Concatenate skip connection
        x = torch.cat([x, low_feat], dim=1)
        x = self.refine_head(x)  # final output logits
        return x


class ViT(nn.Module):
    """
    Placeholder Vision Transformer-based encoder with conv upsampling decoder.
    - Transformer encoder (e.g., ViT) -> lightweight decoder to 64x64 map.
    TODO: Implement actual transformer blocks (use timm or custom) and decoder.
    """
    def __init__(self, 
                in_channels: int = 4, 
                out_channels: int = 1, 
                embed_dim: int = 256, 
                depth: int = 10, 
                num_heads: int = 16, 
                patch_size=16,
                target_size=64,
                **kwargs):
        super().__init__()
        # Patch embedding: convert image to patch tokens
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)

        # Compute patch grid size for positional embedding (256/16 = 16)
        grid_h = grid_w = 256 // patch_size  

        # Positional embedding (learnable, initialized with sin-cos)
        self.pos_embed = PositionalEmbedding(embed_dim, grid_h, grid_w)

        # Transformer encoder stack
        self.encoder = TransformerEncoder(embed_dim, num_heads, depth)

        # Convolutional decoder to reconstruct output map
        self.decoder = ConvDecoder(embed_dim, out_channels, target_size, in_channels=in_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_low = x  # [B, C, H, W]

        # Patch embedding
        x_tok, (H_p, W_p) = self.patch_embed(x)
        x_tok = self.pos_embed(x_tok)

        # Transformer encoding
        x_tok = self.encoder(x_tok)
        B, N, D = x_tok.shape

        # Reshape to 2D feature map
        x_feat = x_tok.transpose(1, 2).reshape(B, D, H_p, W_p)

        # Pass both original input and transformer features to decoder
        out = self.decoder(x_low, x_feat)
        return out