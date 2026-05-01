#!/usr/bin/env python3
"""
PPT++ Main Entry Point
======================

Run experiments:
    python main.py --experiment all          # Run all experiments
    python main.py --experiment ablation     # Only ablation table
    python main.py --experiment lvvit        # Only LV-ViT experiments
    python main.py --experiment comparison   # Cross-architecture comparison
    python main.py --demo                    # Quick framework demo
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_demo():
    """Quick PPT++ framework demonstration."""
    import torch
    import timm
    from ppt_pp.compression.scorer import TokenScorer
    from ppt_pp.compression.pruner import TokenPruner, TokenSqueezer
    from ppt_pp.compression.pooler import SpatialAwareBSM
    from ppt_pp.compression.threshold import AdaptiveThreshold
    from ppt_pp.compression.ppt_module import PPTPlusPlus
    from ppt_pp.models.lvvit import lvvit_small
    from ppt_pp.utils.metrics import compute_flops, compute_params
    
    print("=" * 70)
    print("PPT++: Enhanced Token Pruning & Pooling for Efficient ViTs")
    print("=" * 70)
    
    # --- DeiT-S Demo ---
    print("\n[1] Loading DeiT-Small pretrained...")
    deit = timm.create_model('deit_small_patch16_224', pretrained=True).eval()
    p = compute_params(deit)
    print(f"    DeiT-S: {p['total_params_m']:.1f}M params, "
          f"{len(deit.blocks)} blocks, embed={deit.embed_dim}")
    
    # Compute FLOPs
    flops_base = compute_flops(12, 384, 196, [], [], mlp_ratio=4.0)
    flops_comp = compute_flops(12, 384, 196, [3,6,9], [16,16,16], mlp_ratio=4.0)
    print(f"    Baseline FLOPs: {flops_base['baseline_gflops']:.2f}G")
    print(f"    PPT++ FLOPs:    {flops_comp['compressed_gflops']:.2f}G "
          f"(↓{flops_comp['reduction_pct']:.1f}%)")
    
    # Test token scoring
    print("\n[2] Testing PPT++ components...")
    B, H, N1, Dh = 2, 6, 197, 64
    fake_attn = torch.randn(B, H, N1, N1).softmax(dim=-1)
    fake_vals = torch.randn(B, H, N1, Dh)
    
    scorer = TokenScorer()
    scores = scorer(fake_attn, fake_vals)
    print(f"    Token scores: {scores.shape} (sum per sample: {scores.sum(dim=-1).tolist()})")
    
    # Test pruning + squeezing
    pruner = TokenPruner(use_squeezing=True, squeeze_weight=0.3)
    fake_tokens = torch.randn(2, 196, 384)
    pruned = pruner(fake_tokens, scores, r=16)
    print(f"    Pruning: {fake_tokens.shape} → {pruned.shape}")
    
    # Test spatial BSM
    bsm = SpatialAwareBSM(grid_size=14, sigma=2.0)
    fake_keys = torch.randn(2, 196, 64)
    merged, sizes = bsm(fake_keys, fake_tokens, r=16, layer_depth_ratio=0.5)
    print(f"    Pooling: {fake_tokens.shape} → {merged.shape}")
    
    # --- LV-ViT-S Demo ---
    print("\n[3] Loading LV-ViT-S (custom implementation)...")
    lvvit = lvvit_small(pretrained=False).eval()
    p = compute_params(lvvit)
    print(f"    LV-ViT-S: {p['total_params_m']:.1f}M params, "
          f"{len(lvvit.blocks)} blocks, embed={lvvit.embed_dim}")
    
    flops_lv_base = compute_flops(16, 384, 196, [], [], mlp_ratio=3.0)
    flops_lv_comp = compute_flops(16, 384, 196, [4,8,12], [50,50,50], mlp_ratio=3.0)
    print(f"    Baseline FLOPs: {flops_lv_base['baseline_gflops']:.2f}G")
    print(f"    PPT++ FLOPs:    {flops_lv_comp['compressed_gflops']:.2f}G "
          f"(↓{flops_lv_comp['reduction_pct']:.1f}%)")
    
    # Test forward pass
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = lvvit(dummy)
    print(f"    Forward pass: input={dummy.shape} → logits={out.shape}")
    
    print("\n[4] Framework validated successfully ✓")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="PPT++ Experiments")
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'ablation', 'lvvit', 'comparison', 'threshold',
                                'squeezing', 'spatial', 'schedules', 'sota'],
                       help='Which experiment to run')
    parser.add_argument('--demo', action='store_true', help='Quick demo')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
        return
    
    from ppt_pp.experiments.run_all import (
        run_all_experiments, table_1_sota_comparison, table_2_ablation,
        table_3_threshold_analysis, table_4_squeezing, table_5_spatial_bsm,
        table_6_schedules, table_7_lvvit, table_8_cross_architecture,
    )
    
    device = args.device
    
    if args.experiment == 'all':
        run_all_experiments(device)
    elif args.experiment == 'sota':
        table_1_sota_comparison()
    elif args.experiment == 'ablation':
        import timm
        model = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
        table_2_ablation(model, device)
    elif args.experiment == 'lvvit':
        table_7_lvvit(device)
    elif args.experiment == 'comparison':
        import timm
        model = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
        table_8_cross_architecture(model, device)
    elif args.experiment == 'threshold':
        import timm
        model = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
        table_3_threshold_analysis(model, device)
    elif args.experiment == 'squeezing':
        import timm
        model = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
        table_4_squeezing(model, device)
    elif args.experiment == 'spatial':
        import timm
        model = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
        table_5_spatial_bsm(model, device)
    elif args.experiment == 'schedules':
        import timm
        model = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
        table_6_schedules(model, device)


if __name__ == '__main__':
    main()
