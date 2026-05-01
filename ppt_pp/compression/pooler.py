"""
Token Pooling Module
====================

Implements Bipartite Soft Matching (BSM) for token merging, both standard
and spatial-aware variants.

Standard BSM (ToMe, Bolya et al. 2023):
    1. Partition tokens into two alternating sets A, B
    2. Compute cosine similarity between A and B tokens
    3. Match each A token to most similar B token
    4. Merge top-r pairs via weighted average (proportional to token size)

Spatial-Aware BSM (PPT++ Contribution 3, inspired by ToSA 2025):
    Augments feature similarity with spatial proximity:
        S_fused = α · S_visual + (1-α) · S_spatial
    where S_spatial = exp(-||p_i - p_j||² / 2σ²)
    and α increases linearly with depth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable


class BipartiteSoftMatching(nn.Module):
    """
    Standard Bipartite Soft Matching (BSM) from ToMe.
    
    Partitions tokens into alternating sets and merges the most similar pairs.
    Uses proportional attention: size vector s tracks original token count.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        keys: torch.Tensor,
        tokens: torch.Tensor,
        r: int,
        token_sizes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge r tokens via bipartite soft matching.
        
        Args:
            keys: [B, N, D] key embeddings for similarity
            tokens: [B, N, D_tok] token embeddings to merge
            r: number of tokens to reduce
            token_sizes: [B, N, 1] current token sizes (None = all 1s)
            
        Returns:
            merged_tokens: [B, N-r, D_tok] merged token sequence
            new_sizes: [B, N-r, 1] updated token sizes
        """
        B, N, D = keys.shape
        D_tok = tokens.shape[-1]
        
        if r <= 0:
            sizes = token_sizes if token_sizes is not None else torch.ones(B, N, 1, device=tokens.device)
            return tokens, sizes
        
        # Initialize sizes
        if token_sizes is None:
            token_sizes = torch.ones(B, N, 1, device=tokens.device)
        
        # Alternating partition
        a_idx = torch.arange(0, N, 2, device=keys.device)  # even indices
        b_idx = torch.arange(1, N, 2, device=keys.device)  # odd indices
        
        # Normalize for cosine similarity
        keys_norm = F.normalize(keys, dim=-1)
        a_keys = keys_norm[:, a_idx]  # [B, N_a, D]
        b_keys = keys_norm[:, b_idx]  # [B, N_b, D]
        
        # Similarity scores: [B, N_a, N_b]
        scores = torch.bmm(a_keys, b_keys.transpose(1, 2))
        
        return self._merge_with_scores(
            scores, tokens, token_sizes, a_idx, b_idx, r
        )
    
    def _merge_with_scores(
        self,
        scores: torch.Tensor,
        tokens: torch.Tensor,
        sizes: torch.Tensor,
        a_idx: torch.Tensor,
        b_idx: torch.Tensor,
        r: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute the merge given similarity scores."""
        B = tokens.shape[0]
        D_tok = tokens.shape[-1]
        
        # Best match for each token in set A
        node_max, node_idx = scores.max(dim=-1)  # [B, N_a]
        
        # Select top-r edges (most similar pairs)
        edge_idx = node_max.argsort(dim=-1, descending=True)  # [B, N_a]
        
        unm_idx = edge_idx[:, r:]     # unmerged A indices (keep)
        src_idx = edge_idx[:, :r]     # source A indices (to merge)
        dst_idx = node_idx.gather(dim=-1, index=src_idx)  # dest B indices
        
        # Sort unmerged for order preservation
        unm_idx = unm_idx.sort(dim=-1)[0]
        
        # Get tokens from each partition
        src_tok = tokens[:, a_idx]   # [B, N_a, D_tok]
        dst_tok = tokens[:, b_idx].clone()  # [B, N_b, D_tok]
        
        # Get sizes
        s_src = sizes[:, a_idx].clone()
        s_dst = sizes[:, b_idx].clone()
        
        # Gather source tokens to merge
        src_sel = src_tok.gather(1, src_idx.unsqueeze(-1).expand(-1, -1, D_tok))
        s_src_sel = s_src.gather(1, src_idx.unsqueeze(-1).expand(-1, -1, 1))
        
        # Proportional merge: dst = (dst·s_dst + src·s_src) / (s_dst + s_src)
        dst_expanded = dst_idx.unsqueeze(-1).expand(-1, -1, D_tok)
        s_dst_expanded = dst_idx.unsqueeze(-1).expand(-1, -1, 1)
        
        # Accumulate source tokens into destinations
        weighted_src = src_sel * s_src_sel
        dst_tok.scatter_add_(1, dst_expanded, weighted_src)
        s_dst.scatter_add_(1, s_dst_expanded, s_src_sel)
        
        # Normalize by total size
        dst_tok = dst_tok / (s_dst + 1e-8) * s_dst
        
        # Gather unmerged A tokens
        unm_tok = src_tok.gather(1, unm_idx.unsqueeze(-1).expand(-1, -1, D_tok))
        s_unm = s_src.gather(1, unm_idx.unsqueeze(-1).expand(-1, -1, 1))
        
        # Concatenate: unmerged_A + merged_B
        merged = torch.cat([unm_tok, dst_tok], dim=1)
        new_sizes = torch.cat([s_unm, s_dst], dim=1)
        
        return merged, new_sizes


class SpatialAwareBSM(BipartiteSoftMatching):
    """
    PPT++ Contribution 3: Spatial-Aware Bipartite Soft Matching.
    
    Augments visual similarity with spatial proximity to prevent merging
    spatially distant tokens in shallow layers:
    
        S_fused(i,j) = α · S_visual(i,j) + (1-α) · S_spatial(i,j)
        S_spatial(i,j) = exp(-||p_i - p_j||² / 2σ²)
    
    The mixing coefficient α increases linearly with depth:
        α_l = α_min + (α_max - α_min) · l / (L-1)
    
    α_min = 0.5 (shallow: 50% spatial weight)
    α_max = 0.9 (deep: 10% spatial weight)
    """
    
    def __init__(
        self,
        grid_size: int = 14,
        sigma: float = 2.0,
        alpha_min: float = 0.5,
        alpha_max: float = 0.9,
    ):
        """
        Args:
            grid_size: spatial grid dimension (14 for 196 tokens)
            sigma: Gaussian kernel width
            alpha_min: visual weight at shallowest compression layer
            alpha_max: visual weight at deepest compression layer
        """
        super().__init__()
        self.grid_size = grid_size
        self.sigma = sigma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # Precompute spatial similarity matrix
        self._spatial_sim = None
        self._spatial_device = None
    
    def _get_spatial_sim(self, device: torch.device) -> torch.Tensor:
        """Compute or retrieve cached spatial similarity matrix."""
        if self._spatial_sim is None or self._spatial_device != device:
            g = self.grid_size
            coords = torch.stack(torch.meshgrid(
                torch.arange(g, device=device, dtype=torch.float32),
                torch.arange(g, device=device, dtype=torch.float32),
                indexing='ij'
            ), dim=-1).reshape(-1, 2)  # [N, 2]
            
            dist = torch.cdist(coords, coords, p=2)  # [N, N]
            self._spatial_sim = torch.exp(-dist ** 2 / (2 * self.sigma ** 2))
            self._spatial_device = device
        
        return self._spatial_sim
    
    def get_alpha(self, layer_depth_ratio: float) -> float:
        """
        Compute spatial mixing coefficient for current layer.
        
        Args:
            layer_depth_ratio: position in [0, 1] (0=shallowest, 1=deepest)
            
        Returns:
            alpha: visual similarity weight
        """
        return self.alpha_min + (self.alpha_max - self.alpha_min) * layer_depth_ratio
    
    def forward(
        self,
        keys: torch.Tensor,
        tokens: torch.Tensor,
        r: int,
        token_sizes: Optional[torch.Tensor] = None,
        layer_depth_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Spatial-aware token merging.
        
        Args:
            keys: [B, N, D] key embeddings
            tokens: [B, N, D_tok] token embeddings  
            r: number of tokens to reduce
            token_sizes: [B, N, 1] current sizes
            layer_depth_ratio: layer position for alpha scheduling
            
        Returns:
            merged_tokens: [B, N-r, D_tok]
            new_sizes: [B, N-r, 1]
        """
        B, N, D = keys.shape
        
        if r <= 0:
            sizes = token_sizes if token_sizes is not None else torch.ones(B, N, 1, device=tokens.device)
            return tokens, sizes
        
        if token_sizes is None:
            token_sizes = torch.ones(B, N, 1, device=tokens.device)
        
        # Alternating partition
        a_idx = torch.arange(0, N, 2, device=keys.device)
        b_idx = torch.arange(1, N, 2, device=keys.device)
        
        # Visual similarity
        keys_norm = F.normalize(keys, dim=-1)
        a_keys = keys_norm[:, a_idx]
        b_keys = keys_norm[:, b_idx]
        visual_scores = torch.bmm(a_keys, b_keys.transpose(1, 2))  # [B, N_a, N_b]
        
        # Spatial similarity
        alpha = self.get_alpha(layer_depth_ratio)
        
        if N == self.grid_size * self.grid_size:
            spatial_sim = self._get_spatial_sim(keys.device)
            spatial_ab = spatial_sim[a_idx][:, b_idx]  # [N_a, N_b]
            spatial_ab = spatial_ab / (spatial_ab.max() + 1e-8)
            
            # Fused similarity
            scores = alpha * visual_scores + (1 - alpha) * spatial_ab.unsqueeze(0)
        else:
            # Non-standard token count: fall back to pure visual
            scores = visual_scores
        
        return self._merge_with_scores(
            scores, tokens, token_sizes, a_idx, b_idx, r
        )
