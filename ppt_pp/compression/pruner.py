"""
Token Pruning Module
====================

Implements Top-K token pruning with optional Token Squeezing.

Token Squeezing (PPT++ Contribution 1, from TPS CVPR 2023):
    After pruning, instead of discarding tokens completely, squeeze their
    information into retained tokens via cosine-similarity-weighted fusion:
    
    Step 1 - Match: For each pruned token x_i, find nearest reserved token:
        host(x_i) = argmax_{x_j ∈ S^r} cos(x_i, x_j)
    
    Step 2 - Fuse: Each reserved token absorbs matched pruned tokens:
        y_j = x_j + γ · Σ_{x_i ∈ M(j)} w_i · x_i
        w_i = sim(x_i, x_j) / Σ_{x_k ∈ M(j)} sim(x_k, x_j)
    
    where γ = 0.3 (squeeze weight), M(j) = set of pruned tokens matched to j.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TokenPruner(nn.Module):
    """
    Top-K token pruning with optional token squeezing.
    
    Selects the K = N - r highest-scoring tokens and optionally
    squeezes information from the pruned tokens back.
    """
    
    def __init__(self, use_squeezing: bool = True, squeeze_weight: float = 0.3):
        """
        Args:
            use_squeezing: Enable token squeezing (Contribution 1)
            squeeze_weight: γ in the squeezing formula (default: 0.3)
        """
        super().__init__()
        self.use_squeezing = use_squeezing
        self.squeeze_weight = squeeze_weight
        self.squeezer = TokenSqueezer(squeeze_weight=squeeze_weight)
    
    def forward(
        self,
        img_tokens: torch.Tensor,
        scores: torch.Tensor,
        r: int,
    ) -> torch.Tensor:
        """
        Prune r tokens from img_tokens based on scores.
        
        Args:
            img_tokens: [B, N, D] image tokens (no CLS)
            scores: [B, N] significance scores
            r: number of tokens to remove
            
        Returns:
            pruned_output: [B, N-r, D] remaining tokens after pruning
        """
        B, N, D = img_tokens.shape
        K = N - r  # keep K tokens
        
        # Select top-K indices
        _, top_indices = scores.topk(K, dim=-1)  # [B, K]
        top_indices_sorted = top_indices.sort(dim=-1)[0]  # maintain spatial order
        
        # Gather reserved tokens
        reserved = img_tokens.gather(
            1, top_indices_sorted.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, K, D]
        
        if self.use_squeezing and r > 0:
            # Get pruned tokens
            mask = torch.ones(B, N, dtype=torch.bool, device=img_tokens.device)
            mask.scatter_(1, top_indices_sorted, False)
            pruned = img_tokens[mask].reshape(B, r, D)
            
            # Apply squeezing
            reserved = self.squeezer(reserved, pruned)
        
        return reserved


class TokenSqueezer(nn.Module):
    """
    Token Squeezing (PPT++ Contribution 1)
    
    Recovers information from pruned tokens by fusing them into their
    most similar retained tokens via cosine-similarity weighting.
    
    Reference: TPS (Wei et al., CVPR 2023), Section 3.3
    
    Mathematical formulation:
        y_j = x_j + γ · Σ_{x_i ∈ M(j)} w_i · x_i
        w_i = sim(x_i, x_j) / Σ_k sim(x_k, x_j)
    """
    
    def __init__(self, squeeze_weight: float = 0.3):
        """
        Args:
            squeeze_weight: γ, controls injection strength (default: 0.3)
        """
        super().__init__()
        self.squeeze_weight = squeeze_weight
    
    def forward(
        self,
        reserved: torch.Tensor,
        pruned: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Squeeze information from pruned tokens into reserved tokens.
        
        Args:
            reserved: [B, K, D] retained tokens
            pruned: [B, P, D] pruned tokens
            temperature: softness of similarity weighting
            
        Returns:
            squeezed: [B, K, D] reserved tokens with injected information
        """
        if pruned.shape[1] == 0:
            return reserved
        
        B, K, D = reserved.shape
        P = pruned.shape[1]
        
        # Cosine similarity: [B, P, K]
        res_norm = F.normalize(reserved, dim=-1)
        pru_norm = F.normalize(pruned, dim=-1)
        sim_matrix = torch.bmm(pru_norm, res_norm.transpose(1, 2))  # [B, P, K]
        
        # Match each pruned token to its nearest reserved token
        host_idx = sim_matrix.argmax(dim=-1)  # [B, P]
        
        # Build sparse assignment: for each reserved token j, aggregate matched pruned tokens
        # Create one-hot assignment matrix
        assignment = torch.zeros(B, P, K, device=reserved.device)
        assignment.scatter_(2, host_idx.unsqueeze(-1), 1.0)  # [B, P, K]
        
        # Masked similarity weights (only for matched pairs)
        weights = (sim_matrix * assignment) / temperature  # [B, P, K]
        
        # Transpose to [B, K, P] for aggregation per reserved token
        weights_t = weights.transpose(1, 2)
        
        # Normalize: softmax over pruned tokens matched to each reserved token
        has_match = (weights_t.sum(dim=-1, keepdim=True) > 0).float()  # [B, K, 1]
        match_sum = weights_t.sum(dim=-1, keepdim=True) + 1e-8
        weights_normalized = weights_t / match_sum  # [B, K, P]
        
        # Weighted aggregation of pruned tokens
        pruned_contribution = torch.bmm(weights_normalized, pruned)  # [B, K, D]
        
        # Additive fusion with squeeze weight
        squeezed = reserved + self.squeeze_weight * pruned_contribution * has_match
        
        return squeezed
