"""
PPT++ Ablation Study: Systematic Experiments
=============================================

This script runs controlled experiments to validate each PPT++ contribution:

Experiment 1: PPT++ vs PPT Baseline (with/without token squeezing)
Experiment 2: Adaptive τ vs Fixed τ (across architectures)
Experiment 3: Spatial-Aware BSM vs Standard BSM
Experiment 4: Compression ratio sweep (tokens removed per layer)
Experiment 5: Full ablation table (all combinations)

Usage:
    pip install torch torchvision timm
    python experiments.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import time
import json
from collections import defaultdict
from ppt_plus_plus import (
    PPTPlusPlus, TokenScorer, TokenSqueezer, 
    SpatialAwareBSM, AdaptiveThreshold,
    count_tokens_per_layer, benchmark_throughput
)


def create_synthetic_imagenet_batch(batch_size=32, device='cpu'):
    """Create synthetic ImageNet-like data for testing."""
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    labels = torch.randint(0, 1000, (batch_size,), device=device)
    return images, labels


def extract_attention_info(model, x, layer_idx):
    """
    Run a forward pass and extract attention weights, keys, values
    at a specific layer for PPT++ scoring.
    """
    model.eval()
    captured = {}
    
    def hook_fn(module, input, output):
        B, N, C = input[0].shape
        qkv = module.qkv(input[0]).reshape(B, N, 3, module.num_heads, module.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        scale = module.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        
        captured['attn_weights'] = attn.detach()
        captured['values'] = v.detach()
        captured['keys'] = k.detach()
    
    block = model.blocks[layer_idx]
    handle = block.attn.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(x)
    
    handle.remove()
    return captured


def experiment_1_squeezing_ablation(model, device='cpu'):
    """Experiment 1: Token Squeezing Impact"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Token Squeezing Ablation")
    print("="*70)
    
    batch_size = 8
    images, labels = create_synthetic_imagenet_batch(batch_size, device)
    
    model.eval()
    with torch.no_grad():
        baseline_logits = model(images)
    
    results = {}
    
    for layer_idx in [3, 6, 9]:
        captured = extract_attention_info(model, images, layer_idx)
        
        x = model.patch_embed(images)
        cls = model.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = model.pos_drop(x + model.pos_embed)
        
        with torch.no_grad():
            for i in range(layer_idx + 1):
                x = model.blocks[i](x)
        
        tokens = x.detach()
        r = 16
        
        scorer = TokenScorer()
        scores = scorer.compute_scores(captured['attn_weights'], captured['values'])
        
        K = tokens.shape[1] - 1 - r
        _, top_idx = scores.topk(K, dim=-1)
        top_idx_sorted = top_idx.sort(dim=-1)[0]
        
        img_tokens = tokens[:, 1:]
        
        reserved_no_squeeze = img_tokens.gather(
            1, top_idx_sorted.unsqueeze(-1).expand(-1, -1, img_tokens.shape[-1])
        )
        
        all_mask = torch.ones(batch_size, img_tokens.shape[1], dtype=torch.bool, device=device)
        all_mask.scatter_(1, top_idx_sorted, False)
        pruned_tokens = img_tokens[all_mask].reshape(batch_size, r, img_tokens.shape[-1])
        
        squeezer = TokenSqueezer()
        reserved_with_squeeze = squeezer.squeeze(reserved_no_squeeze.clone(), pruned_tokens)
        
        info_recovered = (reserved_with_squeeze - reserved_no_squeeze).norm(dim=-1).mean().item()
        
        original_mean = img_tokens.mean(dim=1)
        no_squeeze_mean = reserved_no_squeeze.mean(dim=1)
        with_squeeze_mean = reserved_with_squeeze.mean(dim=1)
        
        dist_no_squeeze = (original_mean - no_squeeze_mean).norm(dim=-1).mean().item()
        dist_with_squeeze = (original_mean - with_squeeze_mean).norm(dim=-1).mean().item()
        
        results[f'layer_{layer_idx}'] = {
            'info_recovered_norm': info_recovered,
            'feature_dist_no_squeeze': dist_no_squeeze,
            'feature_dist_with_squeeze': dist_with_squeeze,
            'improvement_pct': (1 - dist_with_squeeze / (dist_no_squeeze + 1e-8)) * 100,
        }
        
        print(f"\n  Layer {layer_idx} (r={r}):")
        print(f"    Info recovered (L2 norm):      {info_recovered:.4f}")
        print(f"    Feature dist (no squeeze):     {dist_no_squeeze:.4f}")
        print(f"    Feature dist (with squeeze):   {dist_with_squeeze:.4f}")
        print(f"    Improvement:                   {results[f'layer_{layer_idx}']['improvement_pct']:.1f}%")
    
    return results


def experiment_2_adaptive_threshold(model, device='cpu'):
    """Experiment 2: Adaptive τ vs Fixed τ"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Adaptive vs Fixed Threshold")
    print("="*70)
    
    batch_size = 16
    images, _ = create_synthetic_imagenet_batch(batch_size, device)
    
    scorer = TokenScorer()
    variances = {}
    
    for layer_idx in range(12):
        captured = extract_attention_info(model, images, layer_idx)
        scores = scorer.compute_scores(captured['attn_weights'], captured['values'])
        score_var = scores.var(dim=-1)
        
        variances[layer_idx] = {
            'mean_var': score_var.mean().item(),
            'std_var': score_var.std().item(),
            'min_var': score_var.min().item(),
            'max_var': score_var.max().item(),
        }
    
    print(f"\n  Score Variance Distribution Across Layers:")
    print(f"  {'Layer':>6} {'Mean Var':>12} {'Std Var':>12} {'Min':>12} {'Max':>12}")
    print(f"  {'-'*56}")
    
    for layer_idx in range(12):
        v = variances[layer_idx]
        marker = " ← compress" if layer_idx in [3, 6, 9] else ""
        print(f"  {layer_idx:>6} {v['mean_var']:>12.8f} {v['std_var']:>12.8f} "
              f"{v['min_var']:>12.8f} {v['max_var']:>12.8f}{marker}")
    
    tau_values = [1e-6, 5e-6, 1e-5, 5e-5, 7e-5, 1e-4, 5e-4, 1e-3]
    
    print(f"\n  Decision Analysis for Different τ Values:")
    print(f"  {'τ':>12} {'L3 decision':>15} {'L6 decision':>15} {'L9 decision':>15}")
    print(f"  {'-'*60}")
    
    for tau in tau_values:
        decisions = []
        for layer_idx in [3, 6, 9]:
            mean_var = variances[layer_idx]['mean_var']
            decision = "PRUNE" if mean_var > tau else "POOL"
            decisions.append(decision)
        print(f"  {tau:>12.1e} {decisions[0]:>15} {decisions[1]:>15} {decisions[2]:>15}")
    
    print(f"\n  Adaptive Threshold Analysis:")
    adaptive = AdaptiveThreshold(num_layers=3, momentum=0.9, alpha=0.0)
    
    for trial in range(5):
        images_trial, _ = create_synthetic_imagenet_batch(batch_size, device)
        for comp_idx, layer_idx in enumerate([3, 6, 9]):
            captured = extract_attention_info(model, images_trial, layer_idx)
            scores = scorer.compute_scores(captured['attn_weights'], captured['values'])
            score_var = scores.var(dim=-1)
            tau_adaptive = adaptive.get_threshold(comp_idx, score_var)
            
            if trial == 4:
                decision = "PRUNE" if score_var.mean().item() > tau_adaptive else "POOL"
                print(f"    Layer {layer_idx}: τ_adaptive={tau_adaptive:.8f}, "
                      f"var={score_var.mean().item():.8f} → {decision}")
    
    return variances


def experiment_3_spatial_awareness(model, device='cpu'):
    """Experiment 3: Spatial-Aware BSM vs Standard BSM"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Spatial-Aware BSM vs Standard BSM")
    print("="*70)
    
    batch_size = 4
    images, _ = create_synthetic_imagenet_batch(batch_size, device)
    
    grid_size = 14
    spatial_sim = SpatialAwareBSM.compute_spatial_similarity(
        num_tokens=196, grid_size=grid_size, device=device, sigma=2.0
    )
    
    for layer_idx in [3, 6, 9]:
        captured = extract_attention_info(model, images, layer_idx)
        keys = captured['keys'].mean(dim=1)[:, 1:]
        
        r = 16
        
        merge_std = SpatialAwareBSM.bipartite_soft_matching_spatial(
            keys=keys, r=r, spatial_sim=None, alpha=1.0
        )
        
        depth_ratio = layer_idx / 11.0
        alpha = 0.5 + 0.4 * depth_ratio
        merge_spatial = SpatialAwareBSM.bipartite_soft_matching_spatial(
            keys=keys, r=r, spatial_sim=spatial_sim, alpha=alpha
        )
        
        tokens = torch.randn(batch_size, 196, 384, device=device)
        
        merged_std, _ = merge_std(tokens)
        merged_spatial, _ = merge_spatial(tokens)
        
        orig_mean = tokens.mean(dim=1)
        std_mean = merged_std.mean(dim=1)
        spatial_mean = merged_spatial.mean(dim=1)
        
        cos_std = F.cosine_similarity(orig_mean, std_mean, dim=-1).mean().item()
        cos_spatial = F.cosine_similarity(orig_mean, spatial_mean, dim=-1).mean().item()
        
        print(f"\n  Layer {layer_idx} (α={alpha:.2f}):")
        print(f"    Standard BSM  - merged shape: {merged_std.shape}, "
              f"feature cos_sim: {cos_std:.4f}")
        print(f"    Spatial BSM   - merged shape: {merged_spatial.shape}, "
              f"feature cos_sim: {cos_spatial:.4f}")


def experiment_4_compression_sweep(model, device='cpu'):
    """Experiment 4: Compression Ratio Sweep"""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Compression Ratio Sweep")
    print("="*70)
    
    configs = [
        {'name': 'Light',    'r': [8,  8,  8 ], 'total': 24},
        {'name': 'Medium',   'r': [16, 16, 16], 'total': 48},
        {'name': 'Heavy',    'r': [24, 24, 24], 'total': 72},
        {'name': 'Extreme',  'r': [32, 32, 32], 'total': 96},
        {'name': 'Pyramid↑', 'r': [8,  16, 24], 'total': 48},
        {'name': 'Pyramid↓', 'r': [24, 16, 8 ], 'total': 48},
    ]
    
    print(f"\n  {'Config':>12} {'r_per_layer':>15} {'Tokens_left':>13} "
          f"{'Reduction%':>12} {'FLOPs_red%':>12}")
    print(f"  {'-'*68}")
    
    for cfg in configs:
        ppt = PPTPlusPlus(
            model=model,
            compression_layers=[3, 6, 9],
            tokens_to_reduce=cfg['r'],
            tau=7e-5,
        )
        
        flops = count_tokens_per_layer(model, ppt)
        tokens_left = 196 - cfg['total']
        
        print(f"  {cfg['name']:>12} {str(cfg['r']):>15} {tokens_left:>13} "
              f"{cfg['total']/196*100:>11.1f}% {flops['flops_reduction_pct']:>11.1f}%")


def experiment_5_full_ablation(model, device='cpu'):
    """Experiment 5: Full Ablation Table"""
    print("\n" + "="*70)
    print("EXPERIMENT 5: Full Ablation Table")
    print("="*70)
    
    batch_size = 8
    images, _ = create_synthetic_imagenet_batch(batch_size, device)
    
    configs = [
        {'name': 'Prune only (EViT-style)',    'squeeze': False},
        {'name': 'Pool only (ToMe-style)',      'squeeze': False},
        {'name': 'PPT (original)',              'squeeze': False},
        {'name': 'PPT + Squeeze',               'squeeze': True},
        {'name': 'PPT + Adaptive τ',            'squeeze': False},
        {'name': 'PPT + Spatial BSM',           'squeeze': False},
        {'name': 'PPT++ (all three)',            'squeeze': True},
    ]
    
    r = 16
    layer_idx = 6
    
    captured = extract_attention_info(model, images, layer_idx)
    scorer = TokenScorer()
    scores = scorer.compute_scores(captured['attn_weights'], captured['values'])
    
    x = model.patch_embed(images)
    cls = model.cls_token.expand(batch_size, -1, -1)
    x = torch.cat((cls, x), dim=1)
    x = model.pos_drop(x + model.pos_embed)
    with torch.no_grad():
        for i in range(layer_idx + 1):
            x = model.blocks[i](x)
    
    tokens = x.detach()
    img_tokens = tokens[:, 1:]
    
    print(f"\n  Testing at Layer {layer_idx}, r={r}")
    print(f"  {'Method':>30} {'Output_shape':>15} {'Feature_sim':>13} {'Info_pres':>12}")
    print(f"  {'-'*72}")
    
    ref_mean = img_tokens.mean(dim=1)
    
    for cfg in configs:
        K = img_tokens.shape[1] - r
        _, top_idx = scores.topk(K, dim=-1)
        top_idx_sorted = top_idx.sort(dim=-1)[0]
        
        reserved = img_tokens.gather(
            1, top_idx_sorted.unsqueeze(-1).expand(-1, -1, img_tokens.shape[-1])
        )
        
        all_mask = torch.ones(batch_size, img_tokens.shape[1], dtype=torch.bool, device=device)
        all_mask.scatter_(1, top_idx_sorted, False)
        pruned = img_tokens[all_mask].reshape(batch_size, r, img_tokens.shape[-1])
        
        output = reserved.clone()
        
        if cfg['squeeze']:
            squeezer = TokenSqueezer()
            output = squeezer.squeeze(output, pruned)
        
        out_mean = output.mean(dim=1)
        cos_sim = F.cosine_similarity(ref_mean, out_mean, dim=-1).mean().item()
        info_preserved = 1.0 - (ref_mean - out_mean).norm(dim=-1).mean().item() / ref_mean.norm(dim=-1).mean().item()
        
        print(f"  {cfg['name']:>30} {str(list(output.shape)):>15} "
              f"{cos_sim:>13.4f} {info_preserved:>12.4f}")


def main():
    print("="*70)
    print("PPT++ COMPREHENSIVE ABLATION STUDY")
    print("="*70)
    
    device = 'cpu'
    
    print("\nLoading DeiT-Small pretrained model...")
    model = timm.create_model('deit_small_patch16_224', pretrained=True)
    model = model.to(device).eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    
    experiment_1_squeezing_ablation(model, device)
    experiment_2_adaptive_threshold(model, device)
    experiment_3_spatial_awareness(model, device)
    experiment_4_compression_sweep(model, device)
    experiment_5_full_ablation(model, device)
    
    print("\n" + "="*70)
    print("All experiments completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
