"""
LV-ViT (Token Labeling Vision Transformer)
==========================================

Implementation of LV-ViT-S/M from:
    "All Tokens Matter: Token Labeling for Training Better Vision Transformers"
    Jiang et al., NeurIPS 2021 (arXiv:2104.10858)

Key architectural differences from DeiT:
    1. 4-layer convolutional patch embedding stem (not single linear projection)
    2. Residual scaling: x = x + F(x)/skip_lam where skip_lam=2.0
    3. MLP ratio = 3.0 (vs DeiT's 4.0)
    4. More blocks: 16 (S), 20 (M), 24 (L) vs DeiT's 12

The attention mechanism is standard MHSA — identical to DeiT.
This allows direct PPT++ injection without modification.

Architecture Table:
    | Variant  | Depth | Dim | Heads | MLP | Params | Top-1 |
    |----------|-------|-----|-------|-----|--------|-------|
    | LV-ViT-T |  12   | 240 |   4   | 3.0 |  8.5M  | 79.1% |
    | LV-ViT-S |  16   | 384 |   6   | 3.0 | 26.2M  | 83.3% |
    | LV-ViT-M |  20   | 512 |   8   | 3.0 | 55.8M  | 84.1% |
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional


class ConvPatchEmbed(nn.Module):
    """
    4-layer convolutional patch embedding stem.
    
    Replaces the single Conv2d(3, embed_dim, 16, 16) in DeiT with:
        Conv(7, stride=2) → BN → ReLU →
        Conv(3, stride=1) → BN → ReLU →
        Conv(3, stride=1) → BN → ReLU →
        Conv(8, stride=8) → flatten
    
    Effective stride = 2×1×1×8 = 16 → same 14×14 grid as DeiT.
    But provides better low-level feature extraction.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        # 4-layer conv stem matching official TokenLabeling p_emb='4_2'
        mid_dim = 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, mid_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, embed_dim, kernel_size=patch_size // 2, stride=patch_size // 2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            tokens: [B, num_patches, embed_dim]
        """
        x = self.stem(x)  # [B, embed_dim, grid_h, grid_w]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class LVViTAttention(nn.Module):
    """
    Standard multi-head self-attention (identical to DeiT/ViT).
    
    LV-ViT does NOT modify the attention mechanism itself.
    The improvements come from the patch embedding and training recipe.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LVViTMLP(nn.Module):
    """
    FFN with MLP ratio = 3.0 (vs DeiT's 4.0).
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 3.0)
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LVViTBlock(nn.Module):
    """
    Transformer block with residual scaling.
    
    Key difference from DeiT: residual is scaled by 1/skip_lam:
        x = x + attn(norm1(x)) / skip_lam
        x = x + ffn(norm2(x)) / skip_lam
    
    With skip_lam=2.0, this is effectively:
        x = x + 0.5 * attn(norm1(x))
    
    This is NOT LayerScale (CaiT). It's a simpler constant scaling.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        skip_lam: float = 2.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LVViTAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = LVViTMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.skip_lam = skip_lam
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor_(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


class LVViT(nn.Module):
    """
    LV-ViT: Leveled Vision Transformer with Token Labeling.
    
    Architecture matches the official TokenLabeling implementation.
    The blocks use standard MHSA + FFN with residual scaling.
    
    PPT++ compression can be injected at blocks [4, 8, 12] for LV-ViT-S.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 384,
        depth: int = 16,
        num_heads: int = 6,
        mlp_ratio: float = 3.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = None,
        skip_lam: float = 2.0,
    ):
        """
        Args:
            img_size: input image resolution
            patch_size: patch size (16 for 14x14 grid at 224px)
            num_classes: number of classification classes
            embed_dim: transformer hidden dimension
            depth: number of transformer blocks
            num_heads: number of attention heads
            mlp_ratio: FFN expansion ratio (3.0 for LV-ViT)
            skip_lam: residual scaling factor (2.0 = multiply by 0.5)
            drop_path_rate: stochastic depth rate
        """
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Patch embedding: 4-conv stem
        self.patch_embed = ConvPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        
        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            LVViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                skip_lam=skip_lam,
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)
    
    @staticmethod
    def _init_module_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features (before classification head)."""
        B = x.shape[0]
        
        x = self.patch_embed(x)  # [B, N, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]
        x = self.pos_drop(x + self.pos_embed)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x[:, 0]  # CLS token output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: features → classification."""
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def lvvit_small(num_classes: int = 1000, pretrained: bool = False, **kwargs) -> LVViT:
    """
    LV-ViT-S: 16 blocks, dim=384, heads=6, mlp_ratio=3.0.
    26.2M params, 6.6G FLOPs, 83.3% Top-1 on ImageNet.
    """
    model = LVViT(
        embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.0,
        num_classes=num_classes, skip_lam=2.0, **kwargs,
    )
    if pretrained:
        print("Note: LV-ViT-S pretrained weights must be loaded from "
              "https://github.com/zihangJiang/TokenLabeling")
    return model


def lvvit_medium(num_classes: int = 1000, pretrained: bool = False, **kwargs) -> LVViT:
    """
    LV-ViT-M: 20 blocks, dim=512, heads=8, mlp_ratio=3.0.
    55.8M params, 84.1% Top-1 on ImageNet.
    """
    model = LVViT(
        embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.0,
        num_classes=num_classes, skip_lam=2.0, **kwargs,
    )
    return model
