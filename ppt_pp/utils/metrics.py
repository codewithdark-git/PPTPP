"""
FLOPs Computation and Throughput Benchmarking
=============================================

Provides accurate theoretical FLOPs computation and real throughput
measurement for both baseline and PPT++-compressed models.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_flops(
    num_blocks: int,
    embed_dim: int,
    initial_tokens: int,
    compression_layers: List[int],
    tokens_to_reduce: List[int],
    mlp_ratio: float = 4.0,
) -> Dict:
    """
    Compute theoretical FLOPs for baseline and compressed ViT.
    
    FLOPs per block:
        Self-Attention: 4·N·D² (QKV proj + output proj) + 2·N²·D (attn + attn×V)
        FFN: 2·N·D·(mlp_ratio·D) = 2·mlp_ratio·N·D²
    
    Args:
        num_blocks: total number of transformer blocks
        embed_dim: transformer hidden dimension D
        initial_tokens: number of image tokens (196 for 14×14)
        compression_layers: block indices with compression
        tokens_to_reduce: tokens removed at each compression layer
        mlp_ratio: FFN expansion ratio
        
    Returns:
        dict with baseline_gflops, compressed_gflops, reduction_pct, etc.
    """
    N0 = initial_tokens + 1  # +CLS
    D = embed_dim
    comp_map = dict(zip(compression_layers, tokens_to_reduce))
    
    baseline_flops = 0
    compressed_flops = 0
    current_N = N0
    
    token_schedule = [N0]
    
    for block_idx in range(num_blocks):
        # Self-Attention FLOPs
        attn_base = 4 * N0 * D**2 + 2 * N0**2 * D
        attn_comp = 4 * current_N * D**2 + 2 * current_N**2 * D
        
        # FFN FLOPs
        ffn_base = 2 * mlp_ratio * N0 * D**2
        ffn_comp = 2 * mlp_ratio * current_N * D**2
        
        baseline_flops += attn_base + ffn_base
        compressed_flops += attn_comp + ffn_comp
        
        if block_idx in comp_map:
            current_N -= comp_map[block_idx]
            token_schedule.append(current_N)
    
    return {
        'baseline_gflops': baseline_flops / 1e9,
        'compressed_gflops': compressed_flops / 1e9,
        'reduction_pct': (1 - compressed_flops / baseline_flops) * 100,
        'flops_ratio': compressed_flops / baseline_flops,
        'final_tokens': current_N,
        'token_schedule': token_schedule,
    }


def compute_params(model: nn.Module) -> Dict:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total,
        'total_params_m': total / 1e6,
        'trainable_params': trainable,
        'trainable_params_m': trainable / 1e6,
    }


def benchmark_throughput(
    model: nn.Module,
    batch_size: int = 64,
    img_size: int = 224,
    num_warmup: int = 10,
    num_runs: int = 50,
    device: str = 'cpu',
) -> Dict:
    """
    Benchmark model throughput (images/sec).
    
    Args:
        model: ViT model
        batch_size: inference batch size
        img_size: input image size
        num_warmup: warmup iterations
        num_runs: benchmark iterations
        device: 'cpu' or 'cuda'
        
    Returns:
        dict with throughput, latency, and statistics
    """
    model = model.to(device).eval()
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    
    times = np.array(times)
    avg = times.mean()
    std = times.std()
    throughput = batch_size / avg
    
    return {
        'throughput_imgs_per_sec': throughput,
        'avg_latency_ms': avg * 1000,
        'std_latency_ms': std * 1000,
        'p50_latency_ms': np.percentile(times, 50) * 1000,
        'p95_latency_ms': np.percentile(times, 95) * 1000,
        'batch_size': batch_size,
        'device': device,
        'num_runs': num_runs,
    }


def measure_compression_quality(
    model: nn.Module,
    images: torch.Tensor,
    compression_layers: List[int],
    tokens_to_reduce: List[int],
    scorer_fn=None,
) -> Dict:
    """
    Measure feature preservation quality at each compression layer.
    
    Computes cosine similarity and L2 distance between original and
    compressed feature representations.
    """
    import torch.nn.functional as F
    from ppt_pp.compression.scorer import TokenScorer
    
    model.eval()
    scorer = scorer_fn or TokenScorer()
    
    results = {}
    
    for i, (layer_idx, r) in enumerate(zip(compression_layers, tokens_to_reduce)):
        # Extract attention info at this layer
        captured = {}
        
        def hook_fn(module, input, output):
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            scale = module.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            captured['attn'] = attn.detach()
            captured['values'] = v.detach()
            captured['keys'] = k.detach()
        
        block = model.blocks[layer_idx]
        handle = block.attn.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            _ = model(images)
        
        handle.remove()
        
        if captured:
            scores = scorer(captured['attn'], captured['values'])
            K = scores.shape[1] - r
            _, top_idx = scores.topk(K, dim=-1)
            
            # Get intermediate tokens
            B = images.shape[0]
            x = model.patch_embed(images)
            cls = model.cls_token.expand(B, -1, -1)
            x = torch.cat((cls, x), dim=1)
            if hasattr(model, 'pos_drop'):
                x = model.pos_drop(x + model.pos_embed)
            else:
                x = x + model.pos_embed
            
            with torch.no_grad():
                for j in range(layer_idx + 1):
                    x = model.blocks[j](x)
            
            img_tokens = x[:, 1:]
            top_idx_sorted = top_idx.sort(dim=-1)[0]
            reserved = img_tokens.gather(
                1, top_idx_sorted.unsqueeze(-1).expand(-1, -1, img_tokens.shape[-1])
            )
            
            orig_mean = img_tokens.mean(dim=1)
            res_mean = reserved.mean(dim=1)
            
            cos_sim = F.cosine_similarity(orig_mean, res_mean, dim=-1).mean().item()
            l2_dist = (orig_mean - res_mean).norm(dim=-1).mean().item()
            
            results[f'layer_{layer_idx}'] = {
                'cos_sim': cos_sim,
                'l2_dist': l2_dist,
                'tokens_before': img_tokens.shape[1],
                'tokens_after': K,
                'reduction_pct': r / img_tokens.shape[1] * 100,
            }
    
    return results
