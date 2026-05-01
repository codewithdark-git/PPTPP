"""
Model Builder
=============

Unified factory for creating ViT models with PPT++ compression.

Supports:
    - DeiT-Ti/S/B via timm
    - LV-ViT-S/M via custom implementation
    
Usage:
    from ppt_pp.models.builder import build_model
    model, ppt = build_model('deit_small', compression='medium')
    model, ppt = build_model('lvvit_small', compression='medium')
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

import timm
from ppt_pp.models.lvvit import LVViT, lvvit_small, lvvit_medium
from ppt_pp.compression.ppt_module import PPTPlusPlus


# Model registry
MODEL_REGISTRY = {
    'deit_tiny': {
        'timm_name': 'deit_tiny_patch16_224',
        'ppt_preset': 'deit_tiny',
        'num_blocks': 12,
        'embed_dim': 192,
    },
    'deit_small': {
        'timm_name': 'deit_small_patch16_224',
        'ppt_preset': 'deit_small',
        'num_blocks': 12,
        'embed_dim': 384,
    },
    'deit_base': {
        'timm_name': 'deit_base_patch16_224',
        'ppt_preset': 'deit_base',
        'num_blocks': 12,
        'embed_dim': 768,
    },
    'lvvit_small': {
        'build_fn': lvvit_small,
        'ppt_preset': 'lvvit_small',
        'num_blocks': 16,
        'embed_dim': 384,
    },
    'lvvit_medium': {
        'build_fn': lvvit_medium,
        'ppt_preset': 'lvvit_medium',
        'num_blocks': 20,
        'embed_dim': 512,
    },
}


def build_model(
    model_name: str,
    pretrained: bool = True,
    compression: str = 'medium',
    inject: bool = True,
    **ppt_kwargs,
) -> Tuple[nn.Module, PPTPlusPlus]:
    """
    Build a ViT model with PPT++ compression.
    
    Args:
        model_name: one of MODEL_REGISTRY keys
        pretrained: load pretrained weights (DeiT only for now)
        compression: 'light', 'medium', 'heavy', 'extreme'
        inject: whether to inject hooks immediately
        **ppt_kwargs: override PPT++ config (e.g., use_squeezing=False)
        
    Returns:
        model: the ViT model (modified in place if inject=True)
        ppt: PPTPlusPlus controller
    """
    assert model_name in MODEL_REGISTRY, (
        f"Unknown model: {model_name}. "
        f"Available: {list(MODEL_REGISTRY.keys())}"
    )
    
    config = MODEL_REGISTRY[model_name]
    
    # Build the base model
    if 'timm_name' in config:
        model = timm.create_model(config['timm_name'], pretrained=pretrained)
    else:
        model = config['build_fn'](pretrained=pretrained)
    
    model.eval()
    
    # Apply compression level scaling
    preset = config['ppt_preset']
    base_config = PPTPlusPlus.PRESETS[preset]
    
    if compression in PPTPlusPlus.COMPRESSION_LEVELS:
        scale = PPTPlusPlus.COMPRESSION_LEVELS[compression]['scale']
        tokens = [max(1, int(t * scale)) for t in base_config['tokens_per_layer']]
        ppt_kwargs.setdefault('tokens_per_layer', tokens)
    
    # Build PPT++
    ppt = PPTPlusPlus(
        model=model,
        preset=preset,
        **ppt_kwargs,
    )
    
    if inject:
        ppt.inject()
    
    return model, ppt


def inject_ppt(
    model: nn.Module,
    preset: str,
    **kwargs,
) -> PPTPlusPlus:
    """
    Inject PPT++ into an existing model.
    
    Args:
        model: pre-built ViT model
        preset: PPT++ preset name
        **kwargs: PPT++ config overrides
        
    Returns:
        ppt: PPTPlusPlus controller
    """
    ppt = PPTPlusPlus(model=model, preset=preset, **kwargs)
    ppt.inject()
    return ppt
