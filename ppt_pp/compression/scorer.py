"""
Token Scoring Module
====================
Computes significance scores for tokens using ATS-style scoring (PPT Eq. 1).

Score_i = A[cls→i] × ||V_i|| / Σ_j A[cls→j] × ||V_j||

This combines CLS-token attention (which tokens the model attends to)
with the Value norm (which tokens carry meaningful features).
"""

import torch
import torch.nn as nn
from typing import Optional


class TokenScorer(nn.Module):
    """
    Significance scorer for vision transformer tokens.
    
    Reference: PPT Section 3.2, Eq. 1; ATS (Fayyaz et al., 2022)
    
    Given attention weights A ∈ R^{B×H×(N+1)×(N+1)} and values V ∈ R^{B×H×(N+1)×D_h},
    computes per-token scores that combine attention importance with value magnitude:
    
        raw_i = A_{cls→i} · ||V_i||₂
        Score_i = raw_i / Σ_j raw_j
    
    Averaged across attention heads for robust scoring.
    """
    
    def __init__(self, use_value_norm: bool = True):
        super().__init__()
        self.use_value_norm = use_value_norm
    
    def forward(self, attn_weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        cls_attn = attn_weights[:, :, 0, 1:]
        if self.use_value_norm:
            v_norms = values[:, :, 1:, :].norm(dim=-1)
            raw_scores = cls_attn * v_norms
        else:
            raw_scores = cls_attn
        scores = raw_scores / (raw_scores.sum(dim=-1, keepdim=True) + 1e-8)
        scores = scores.mean(dim=1)
        return scores
    
    @staticmethod
    def compute_variance(scores: torch.Tensor) -> torch.Tensor:
        return scores.var(dim=-1)
