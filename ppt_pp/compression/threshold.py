"""
Threshold Strategies for Prune/Pool Decision
=============================================

Implements two threshold strategies:
1. FixedThreshold: Single global τ (original PPT)
2. AdaptiveThreshold: Per-layer EMA-based τ (PPT++ Contribution 2)

The decision rule (PPT Eq. 2):
    S_op = Var(Score_1, ..., Score_N)
    if S_op > τ: PRUNE
    else: POOL
"""

import math
import torch
from typing import Optional, List


class FixedThreshold:
    """Original PPT threshold: single global τ for all layers."""
    def __init__(self, tau: float = 7e-5):
        self.tau = tau
    def __call__(self, layer_idx: int, score_variance: torch.Tensor) -> float:
        return self.tau
    def __repr__(self):
        return f"FixedThreshold(τ={self.tau:.1e})"


class AdaptiveThreshold:
    """
    PPT++ Contribution 2: Adaptive Per-Layer Threshold.
    τ_l = μ_l + α·σ_l where μ_l, σ_l are EMA-tracked statistics.
    """
    def __init__(self, num_layers: int, momentum: float = 0.1, alpha: float = 0.0):
        self.num_layers = num_layers
        self.momentum = momentum
        self.alpha = alpha
        self.running_mean: List[Optional[float]] = [None] * num_layers
        self.running_var: List[Optional[float]] = [None] * num_layers
        self.initialized: List[bool] = [False] * num_layers
    
    def __call__(self, layer_idx: int, score_variance: torch.Tensor) -> float:
        batch_mean = score_variance.mean().item()
        batch_var = score_variance.var().item() if score_variance.numel() > 1 else 0.0
        if not self.initialized[layer_idx]:
            self.running_mean[layer_idx] = batch_mean
            self.running_var[layer_idx] = batch_var
            self.initialized[layer_idx] = True
        else:
            m = self.momentum
            self.running_mean[layer_idx] = (1 - m) * self.running_mean[layer_idx] + m * batch_mean
            self.running_var[layer_idx] = (1 - m) * self.running_var[layer_idx] + m * batch_var
        mu = self.running_mean[layer_idx]
        sigma = math.sqrt(max(self.running_var[layer_idx], 1e-12))
        return mu + self.alpha * sigma
    
    def reset(self):
        self.running_mean = [None] * self.num_layers
        self.running_var = [None] * self.num_layers
        self.initialized = [False] * self.num_layers
    
    def get_thresholds(self) -> List[Optional[float]]:
        thresholds = []
        for i in range(self.num_layers):
            if self.initialized[i]:
                mu = self.running_mean[i]
                sigma = math.sqrt(max(self.running_var[i], 1e-12))
                thresholds.append(mu + self.alpha * sigma)
            else:
                thresholds.append(None)
        return thresholds
    
    def __repr__(self):
        return f"AdaptiveThreshold(n_layers={self.num_layers}, momentum={self.momentum}, α={self.alpha})"
