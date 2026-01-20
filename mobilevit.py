import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """Standard convolution block with batch normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        activation: nn.Module = nn.SiLU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InvertedResidualBlock(nn.Module):
    """MobileNet-style inverted residual block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 4,
    ):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # Point-wise expansion
            layers.append(ConvBlock(in_channels, hidden_dim, kernel_size=1, padding=0))
        
        # Depth-wise convolution - adjust padding for stride=2
        if stride == 2:
            # For stride=2, we need padding=1 to get output_size = input_size/2
            padding = 1
        else:
            # For stride=1, padding=1 maintains spatial dimensions
            padding = 1
        
        layers.append(
            ConvBlock(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=padding,
                groups=hidden_dim,
            )
        )
        
        # Point-wise linear projection
        layers.append(
            nn.Sequential(
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_residual:
            return x + out
        return out


class TransformerBlock(nn.Module):
    """Standard Transformer block with multi-head self-attention and feed-forward network."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, N, P, d)
        # where:
        #   N = number of patches
        #   P = number of pixels within each patch
        #   d = embedding dimension
        B, N, P, d = x.shape
        
        # Reshape for attention: (B*N, P, d)
        x_reshaped = x.view(B * N, P, d)
        
        # Self-attention
        x_norm = self.norm1(x_reshaped)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_reshaped = x_reshaped + attn_out
        
        # Feed-forward (MLP)
        x_norm = self.norm2(x_reshaped)
        mlp_out = self.mlp(x_norm)
        x_reshaped = x_reshaped + mlp_out
        
        # Reshape back: (B, N, P, d)
        x = x_reshaped.view(B, N, P, d)
        return x


class MobileViTBlock(nn.Module):
    """
    MobileViT Block implementing the "Unfold - Transform - Fold" mechanism.
    
    This block combines:
    - Local Representation (L): 3x3 convolutions for local features
    - Global Representation (G): Sparse Global Attention via Transformer
    - Fusion (F): Concatenation of local and global features
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        num_transformer_blocks: int = 2,
        patch_size: Tuple[int, int] = (2, 2),
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.transformer_dim = transformer_dim
        
        # Local representation: 3x3 convolution
        # Both should preserve spatial dimensions (stride=1, proper padding)
        self.local_rep = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, transformer_dim, kernel_size=1, stride=1, padding=0),
        )
        
        # Global representation: Transformer blocks
        self.global_rep = nn.ModuleList([
            TransformerBlock(
                embed_dim=transformer_dim,
                num_heads=num_heads,
                mlp_ratio=ffn_dim / transformer_dim,
                dropout=dropout,
            )
            for _ in range(num_transformer_blocks)
        ])
        
        self.norm = nn.LayerNorm(transformer_dim)
        
        # Fusion: Project and concatenate
        self.fusion = ConvBlock(
            transformer_dim * 2,  # local + global
            out_channels,
            kernel_size=1,
            padding=0,
        )
        
        # Convolution for local features to match dimensions
        self.conv_proj = ConvBlock(in_channels, transformer_dim, kernel_size=1, padding=0)
    
    def unfolding(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Unfold operation: Convert (B, C, H, W) to (B, N, P, d)
        
        Notation (following the MobileViT paper):
        - N = number of patches
        - P = number of pixels in each patch
        - d = channels / embedding dimension
        
        This allows the transformer to process "pixels at the same position
        across all patches" rather than just neighboring pixels.
        """
        B, C, H_orig, W_orig = x.shape
        ph, pw = self.patch_size
        
        # Calculate number of patches (ceiling division)
        num_patches_h = (H_orig + ph - 1) // ph
        num_patches_w = (W_orig + pw - 1) // pw
        
        # Calculate padding needed to make dimensions divisible by patch size
        pad_h = (ph - (H_orig % ph)) % ph
        pad_w = (pw - (W_orig % pw)) % pw
        
        # Pad if necessary
        H_padded, W_padded = H_orig, W_orig
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H_padded, W_padded = H_orig + pad_h, W_orig + pad_w
        
        # Unfold: (B, C, H, W) -> (B, C, num_patches, ph*pw)
        # Each patch is ph*pw pixels (P), and we have num_patches total patches (N)
        num_patches = num_patches_h * num_patches_w
        
        # Reshape to extract patches
        # (B, C, H, W) -> (B, C, num_patches_h, ph, num_patches_w, pw)
        x = x.view(B, C, num_patches_h, ph, num_patches_w, pw)
        # -> (B, C, num_patches_h, num_patches_w, ph, pw)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        # -> (B, C, num_patches, ph*pw)
        x = x.view(B, C, num_patches, ph * pw)
        # -> (B, num_patches, ph*pw, C)  (shape: B, N, P, d)
        x = x.permute(0, 2, 3, 1).contiguous()
        # -> (B, N, P, d)
        
        info_dict = {
            "orig_size": (H_orig, W_orig),  # Store original unpadded size
            "padded_size": (H_padded, W_padded),  # Store padded size for reconstruction
            "num_patches": (num_patches_h, num_patches_w),
            "patch_size": (ph, pw),
        }
        
        return x, info_dict
    
    def folding(self, x: torch.Tensor, info_dict: dict) -> torch.Tensor:
        """
        Fold operation: Convert (B, N, P, d) back to (B, C, H, W)
        
        Notation matches `unfolding`:
        - N = number of patches
        - P = number of pixels per patch
        - d = channels / embedding dimension
        """
        B, N, P, d = x.shape
        num_patches_h, num_patches_w = info_dict["num_patches"]
        ph, pw = info_dict["patch_size"]
        H_orig, W_orig = info_dict["orig_size"]
        H_padded, W_padded = info_dict["padded_size"]
        
        # (B, N, P, d) -> (B, num_patches, ph*pw, d)
        # -> (B, d, num_patches, ph*pw)
        x = x.permute(0, 3, 1, 2).contiguous()
        # -> (B, d, num_patches_h, num_patches_w, ph, pw)
        x = x.view(B, d, num_patches_h, num_patches_w, ph, pw)
        # -> (B, d, num_patches_h, ph, num_patches_w, pw)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        # -> (B, d, H_padded, W_padded)
        x = x.view(B, d, num_patches_h * ph, num_patches_w * pw)
        
        # Remove padding to restore original size
        if H_padded != H_orig or W_padded != W_orig:
            x = x[:, :, :H_orig, :W_orig]
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MobileViT block:
        1. Local representation (3x3 conv)
        2. Unfold -> Transform -> Fold (Global representation)
        3. Fusion (concatenate local + global)
        """
        # Local representation
        local_features = self.local_rep(x)
        
        # Global representation: Unfold - Transform - Fold
        # Unfold: (B, C, H, W) -> (B, N, P, d)
        global_features, info_dict = self.unfolding(local_features)
        
        # Transform: Apply transformer blocks
        for transformer in self.global_rep:
            global_features = transformer(global_features)
        
        global_features = self.norm(global_features)
        
        # Fold: (B, N, P, d) -> (B, C, H, W)
        global_features = self.folding(global_features, info_dict)
        
        # Fusion: Concatenate local and global features
        local_proj = self.conv_proj(x)
        fused = torch.cat([local_proj, global_features], dim=1)
        
        # Final projection
        out = self.fusion(fused)
        
        return out


class MobileViT(nn.Module):
    """
    MobileViT model for image classification.
    Optimized for CIFAR-10 (32x32 images).
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        width_multiplier: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Calculate channel dimensions based on width multiplier
        def make_divisible(v, divisor=8):
            return max(divisor, int(v + divisor / 2) // divisor * divisor)
        
        base_channels = 16
        channels = [make_divisible(base_channels * width_multiplier * (2 ** i)) 
                   for i in range(5)]
        
        # Initial stem: Convert RGB to feature maps
        self.stem = nn.Sequential(
            ConvBlock(3, channels[0], kernel_size=3, stride=1, padding=1),
            InvertedResidualBlock(channels[0], channels[1], stride=1),
        )
        
        # Stage 1: Standard convolutions
        self.stage1 = nn.Sequential(
            InvertedResidualBlock(channels[1], channels[2], stride=2),
            InvertedResidualBlock(channels[2], channels[2], stride=1),
        )
        
        # Stage 2: First MobileViT block
        self.stage2 = nn.Sequential(
            InvertedResidualBlock(channels[2], channels[3], stride=2),
            MobileViTBlock(
                in_channels=channels[3],
                out_channels=channels[3],
                transformer_dim=channels[3],
                ffn_dim=channels[3] * 2,
                num_transformer_blocks=2,
                patch_size=(2, 2),
                num_heads=4,
                dropout=dropout,
            ),
        )
        
        # Stage 3: Second MobileViT block
        self.stage3 = nn.Sequential(
            InvertedResidualBlock(channels[3], channels[4], stride=2),
            MobileViTBlock(
                in_channels=channels[4],
                out_channels=channels[4],
                transformer_dim=channels[4],
                ffn_dim=channels[4] * 2,
                num_transformer_blocks=4,
                patch_size=(2, 2),
                num_heads=4,
                dropout=dropout,
            ),
        )
        
        # Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[4], num_classes),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x


def mobilevit_xxs(num_classes: int = 10) -> MobileViT:
    """Extra extra small variant."""
    return MobileViT(num_classes=num_classes, width_multiplier=0.5)


def mobilevit_xs(num_classes: int = 10) -> MobileViT:
    """Extra small variant."""
    return MobileViT(num_classes=num_classes, width_multiplier=0.75)


def mobilevit_s(num_classes: int = 10) -> MobileViT:
    """Small variant."""
    return MobileViT(num_classes=num_classes, width_multiplier=1.0)
