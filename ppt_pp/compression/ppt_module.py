"""
PPT Compression Module & PPT++ Framework
=========================================

The main compression module that integrates:
- Token scoring
- Prune/Pool decision (variance-based)
- Token pruning with squeezing
- Spatial-aware token pooling

And the PPT++ framework that injects these into ViT models via hooks.

Key Algorithm (PPT++ Algorithm 1):
    1. Score tokens: Score_i = A[cls→i]·||V_i|| / Σ_j A[cls→j]·||V_j||
    2. Compute variance: S_op = Var(Score_1, ..., Score_N)
    3. Get threshold: τ_l = EMA_mean_l + α·EMA_std_l
    4. If S_op > τ_l: Prune (Top-K + Squeeze)
       Else: Pool (Spatial-Aware BSM)
    5. Return compressed tokens
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from ppt_pp.compression.scorer import TokenScorer
from ppt_pp.compression.threshold import FixedThreshold, AdaptiveThreshold
from ppt_pp.compression.pruner import TokenPruner
from ppt_pp.compression.pooler import BipartiteSoftMatching, SpatialAwareBSM


class PPTCompressor(nn.Module):
    """
    Single PPT++ compression module for one layer.
    
    Decides between pruning and pooling per sample based on
    score variance, then applies the chosen strategy.
    """
    
    def __init__(
        self,
        layer_idx: int,
        r: int,
        threshold_fn,
        pruner: TokenPruner,
        pooler: nn.Module,
        layer_depth_ratio: float = 0.5,
    ):
        """
        Args:
            layer_idx: index in the compression layer list
            r: number of tokens to remove at this layer
            threshold_fn: callable(layer_idx, variance) -> threshold
            pruner: TokenPruner instance
            pooler: BSM or SpatialAwareBSM instance
            layer_depth_ratio: depth position for spatial alpha
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.r = r
        self.threshold_fn = threshold_fn
        self.pruner = pruner
        self.pooler = pooler
        self.layer_depth_ratio = layer_depth_ratio
        self.scorer = TokenScorer()
        
        # Statistics tracking
        self.stats = {
            'prune_count': 0,
            'pool_count': 0,
            'total_count': 0,
            'variances': [],
        }
    
    def forward(
        self,
        tokens: torch.Tensor,
        attn_weights: torch.Tensor,
        values: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply PPT++ compression.
        
        Args:
            tokens: [B, N+1, D] all tokens including CLS
            attn_weights: [B, H, N+1, N+1] attention weights
            values: [B, H, N+1, D_h] value embeddings
            keys: [B, H, N+1, D_h] key embeddings
            
        Returns:
            compressed: [B, N+1-r, D] compressed token sequence
        """
        B, N_plus_1, D = tokens.shape
        N = N_plus_1 - 1
        
        cls_token = tokens[:, :1]     # [B, 1, D]
        img_tokens = tokens[:, 1:]    # [B, N, D]
        
        # 1. Score tokens
        scores = self.scorer(attn_weights, values)  # [B, N]
        
        # 2. Compute score variance and make decision
        score_var = TokenScorer.compute_variance(scores)  # [B]
        threshold = self.threshold_fn(self.layer_idx, score_var)
        decisions = score_var > threshold  # [B], True = prune
        
        # Track stats
        self.stats['prune_count'] += decisions.sum().item()
        self.stats['pool_count'] += (~decisions).sum().item()
        self.stats['total_count'] += decisions.numel()
        self.stats['variances'].append(score_var.mean().item())
        
        # 3. Apply compression per sample
        output_tokens = torch.zeros(B, N - self.r, D, device=tokens.device)
        
        prune_mask = decisions
        pool_mask = ~decisions
        
        # Prune path
        if prune_mask.any():
            idx = prune_mask.nonzero(as_tuple=True)[0]
            pruned_result = self.pruner(
                img_tokens[idx], scores[idx], self.r
            )
            output_tokens[idx] = pruned_result
        
        # Pool path
        if pool_mask.any():
            idx = pool_mask.nonzero(as_tuple=True)[0]
            # Average keys across heads for BSM
            pool_keys = keys[idx, :, 1:, :].mean(dim=1)  # [B_pool, N, D_h]
            pool_tokens = img_tokens[idx]
            
            if isinstance(self.pooler, SpatialAwareBSM):
                merged, _ = self.pooler(
                    pool_keys, pool_tokens, self.r,
                    layer_depth_ratio=self.layer_depth_ratio,
                )
            else:
                merged, _ = self.pooler(pool_keys, pool_tokens, self.r)
            
            output_tokens[idx] = merged
        
        # Reassemble with CLS token
        compressed = torch.cat([cls_token, output_tokens], dim=1)
        return compressed
    
    def get_stats(self) -> Dict:
        """Return compression statistics."""
        total = max(self.stats['total_count'], 1)
        return {
            'prune_ratio': self.stats['prune_count'] / total,
            'pool_ratio': self.stats['pool_count'] / total,
            'avg_variance': (sum(self.stats['variances']) / max(len(self.stats['variances']), 1)),
            'total_decisions': self.stats['total_count'],
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'prune_count': 0, 'pool_count': 0,
            'total_count': 0, 'variances': [],
        }


class PPTPlusPlus(nn.Module):
    """
    PPT++ Framework: Hook-based injection into any ViT model.
    
    Wraps attention modules at specified layers to capture attn weights,
    keys, and values, then applies PPT++ compression after each block.
    
    Supports:
        - DeiT-Ti/S/B (12 blocks, compression at [3, 6, 9])
        - LV-ViT-S (16 blocks, compression at [4, 8, 12])
        - Custom configurations
    
    Usage:
        model = timm.create_model('deit_small_patch16_224', pretrained=True)
        ppt = PPTPlusPlus(model, config)
        ppt.inject()  # Modifies model in-place
        output = model(images)  # Now runs with compression
    """
    
    # Preset configurations
    PRESETS = {
        'deit_tiny': {
            'compression_layers': [3, 6, 9],
            'tokens_per_layer': [16, 16, 16],
            'tau': 7e-5,
            'num_blocks': 12,
            'embed_dim': 192,
            'grid_size': 14,
        },
        'deit_small': {
            'compression_layers': [3, 6, 9],
            'tokens_per_layer': [16, 16, 16],
            'tau': 7e-5,
            'num_blocks': 12,
            'embed_dim': 384,
            'grid_size': 14,
        },
        'deit_base': {
            'compression_layers': [3, 6, 9],
            'tokens_per_layer': [16, 16, 16],
            'tau': 7e-5,
            'num_blocks': 12,
            'embed_dim': 768,
            'grid_size': 14,
        },
        'lvvit_small': {
            'compression_layers': [4, 8, 12],
            'tokens_per_layer': [50, 50, 50],  # 50 tokens per stage as in PPT paper
            'tau': 5e-4,
            'num_blocks': 16,
            'embed_dim': 384,
            'grid_size': 14,
        },
        'lvvit_medium': {
            'compression_layers': [5, 10, 15],
            'tokens_per_layer': [50, 50, 50],
            'tau': 5e-4,
            'num_blocks': 20,
            'embed_dim': 512,
            'grid_size': 14,
        },
    }
    
    COMPRESSION_LEVELS = {
        'light':    {'scale': 0.5},   # half the default tokens removed
        'medium':   {'scale': 1.0},   # default
        'heavy':    {'scale': 1.5},   # 50% more removed
        'extreme':  {'scale': 2.0},   # 2x removed
    }
    
    def __init__(
        self,
        model: nn.Module,
        preset: Optional[str] = None,
        compression_layers: Optional[List[int]] = None,
        tokens_per_layer: Optional[List[int]] = None,
        tau: Optional[float] = None,
        use_adaptive_threshold: bool = True,
        use_squeezing: bool = True,
        use_spatial_bsm: bool = True,
        squeeze_weight: float = 0.3,
        spatial_sigma: float = 2.0,
        grid_size: int = 14,
        adaptive_alpha: float = 0.0,
        adaptive_momentum: float = 0.1,
    ):
        """
        Args:
            model: ViT model (timm or LV-ViT)
            preset: one of PRESETS keys (auto-configures everything)
            compression_layers: block indices for compression
            tokens_per_layer: tokens to remove per layer
            tau: fixed threshold (used if use_adaptive_threshold=False)
            use_adaptive_threshold: PPT++ Contribution 2
            use_squeezing: PPT++ Contribution 1
            use_spatial_bsm: PPT++ Contribution 3
            squeeze_weight: γ for squeezing
            spatial_sigma: σ for spatial kernel
            grid_size: spatial grid dimension
        """
        super().__init__()
        self.model = model
        
        # Load preset if specified
        if preset and preset in self.PRESETS:
            cfg = self.PRESETS[preset]
            compression_layers = compression_layers or cfg['compression_layers']
            tokens_per_layer = tokens_per_layer or cfg['tokens_per_layer']
            tau = tau or cfg['tau']
            grid_size = cfg.get('grid_size', 14)
        
        assert compression_layers is not None, "Must specify compression_layers or preset"
        assert tokens_per_layer is not None, "Must specify tokens_per_layer or preset"
        assert len(compression_layers) == len(tokens_per_layer)
        
        self.compression_layers = compression_layers
        self.tokens_per_layer = tokens_per_layer
        self.tau = tau
        self.grid_size = grid_size
        self.num_compression_layers = len(compression_layers)
        
        # Build threshold strategy
        if use_adaptive_threshold:
            self.threshold = AdaptiveThreshold(
                num_layers=len(compression_layers),
                momentum=adaptive_momentum,
                alpha=adaptive_alpha,
            )
        else:
            self.threshold = FixedThreshold(tau=tau or 7e-5)
        
        # Build pruner
        self.pruner = TokenPruner(
            use_squeezing=use_squeezing,
            squeeze_weight=squeeze_weight,
        )
        
        # Build pooler
        if use_spatial_bsm:
            self.pooler = SpatialAwareBSM(
                grid_size=grid_size,
                sigma=spatial_sigma,
            )
        else:
            self.pooler = BipartiteSoftMatching()
        
        # Build per-layer compressors
        max_layer = max(compression_layers) if compression_layers else 1
        self.compressors = nn.ModuleList()
        for i, (layer_idx, r) in enumerate(zip(compression_layers, tokens_per_layer)):
            depth_ratio = i / max(len(compression_layers) - 1, 1)
            compressor = PPTCompressor(
                layer_idx=i,
                r=r,
                threshold_fn=self.threshold,
                pruner=self.pruner,
                pooler=self.pooler,
                layer_depth_ratio=depth_ratio,
            )
            self.compressors.append(compressor)
        
        # Storage for captured attention info
        self._captured = {}
        self._hooks = []
        self._injected = False
    
    def inject(self):
        """
        Inject PPT++ into the model by wrapping attention modules at
        compression layers. Call this ONCE before inference/training.
        """
        if self._injected:
            return
        
        for i, layer_idx in enumerate(self.compression_layers):
            block = self.model.blocks[layer_idx]
            
            # Register forward hook on the attention module
            def make_hook(comp_idx, blk_idx):
                def hook_fn(module, input, output):
                    # Extract QKV from the attention input
                    x = input[0]
                    B, N, C = x.shape
                    
                    attn = module
                    qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, attn.head_dim)
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    
                    scale = attn.head_dim ** -0.5
                    attn_weights = (q @ k.transpose(-2, -1)) * scale
                    attn_weights = attn_weights.softmax(dim=-1)
                    
                    self._captured[blk_idx] = {
                        'attn_weights': attn_weights.detach(),
                        'values': v.detach(),
                        'keys': k.detach(),
                    }
                return hook_fn
            
            handle = block.attn.register_forward_hook(make_hook(i, layer_idx))
            self._hooks.append(handle)
        
        self._injected = True
    
    def remove_hooks(self):
        """Remove all injected hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._injected = False
    
    def compress_at_layer(self, tokens: torch.Tensor, comp_idx: int, block_idx: int) -> torch.Tensor:
        """
        Apply PPT++ compression at a specific layer.
        
        Args:
            tokens: [B, N+1, D] current token sequence
            comp_idx: index in self.compressors
            block_idx: block index in the model
            
        Returns:
            compressed: [B, N+1-r, D]
        """
        if block_idx not in self._captured:
            return tokens  # No captured attention info
        
        captured = self._captured[block_idx]
        return self.compressors[comp_idx](
            tokens,
            captured['attn_weights'],
            captured['values'],
            captured['keys'],
        )
    
    def get_all_stats(self) -> List[Dict]:
        """Get compression stats from all layers."""
        return [c.get_stats() for c in self.compressors]
    
    def reset_all_stats(self):
        """Reset all layer statistics."""
        for c in self.compressors:
            c.reset_stats()
    
    def get_config(self) -> Dict:
        """Return current configuration as dict."""
        return {
            'compression_layers': self.compression_layers,
            'tokens_per_layer': self.tokens_per_layer,
            'tau': self.tau,
            'threshold_type': type(self.threshold).__name__,
            'use_squeezing': self.pruner.use_squeezing,
            'use_spatial_bsm': isinstance(self.pooler, SpatialAwareBSM),
            'squeeze_weight': self.pruner.squeeze_weight,
            'grid_size': self.grid_size,
        }
