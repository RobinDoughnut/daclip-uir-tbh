#!/usr/bin/env python3
"""
Integration script for custom residual attention blocks in OpenCLIP.
This script shows how to integrate the experimental blocks into the existing codebase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Any
from collections import OrderedDict
import math

# Import the existing OpenCLIP components
import sys
sys.path.append('universal-image-restoration/open_clip')

try:
    from transformer import LayerNorm, LayerScale, Transformer
except ImportError:
    print("Warning: Could not import OpenCLIP transformer components")
    # Define minimal components for standalone usage
    class LayerNorm(nn.LayerNorm):
        def forward(self, x: torch.Tensor):
            orig_type = x.dtype
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            return x.to(orig_type)
    
    class LayerScale(nn.Module):
        def __init__(self, dim, init_values=1e-5, inplace=False):
            super().__init__()
            self.inplace = inplace
            self.gamma = nn.Parameter(init_values * torch.ones(dim))
        
        def forward(self, x):
            return x.mul_(self.gamma) if self.inplace else x * self.gamma


# ============================================================================
# CUSTOM ATTENTION BLOCKS FOR OPENCLIP INTEGRATION
# ============================================================================

class ParallelResidualAttentionBlock(nn.Module):
    """
    Parallel residual connections: Apply attention and MLP in parallel.
    Best choice for speed-critical applications.
    """
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        # Handle cross-attention keys/values
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        # Parallel computation: attention and MLP simultaneously
        attn_out = self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        mlp_out = self.ls_2(self.mlp(self.ln_2(q_x)))

        return q_x + attn_out + mlp_out


class LinearAttentionResidualBlock(nn.Module):
    """
    Linear Attention: O(n) complexity instead of O(n²).
    Best choice for high-resolution images and long sequences.
    """
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.head_dim = d_model // n_head
        self.n_head = n_head

        # Linear attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ln_2 = norm_layer(d_model)
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def linear_attention(self, q, k, v):
        """Linear attention with O(n) complexity"""
        # Apply ELU + 1 to ensure positivity
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Compute k*v and q*(k*v) efficiently
        kv = torch.einsum('bnhd,bnhe->bhde', k, v)
        qkv = torch.einsum('bnhd,bhde->bnhe', q, kv)

        # Normalization
        k_sum = k.sum(dim=1, keepdim=True)
        qk = torch.einsum('bnhd,bnhd->bnh', q, k_sum)

        return qkv / (qk.unsqueeze(-1) + 1e-6)

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        B, N, C = q_x.shape
        
        # Use q_x for k,v if not provided (self-attention)
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        
        # Apply layer norms
        q_norm = self.ln_1(q_x)
        k_norm = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") else self.ln_1(k_x)
        v_norm = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") else self.ln_1(v_x)

        # Project to q, k, v
        q = self.q_proj(q_norm).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        k_seq_len = k_x.shape[1] if k_x is not None else N
        v_seq_len = v_x.shape[1] if v_x is not None else N
        k = self.k_proj(k_norm).view(B, k_seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(v_norm).view(B, v_seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Apply linear attention
        attn_out = self.linear_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, C)
        attn_out = self.out_proj(attn_out)

        x = q_x + self.ls_1(attn_out)
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class DeepNormResidualAttentionBlock(nn.Module):
    """
    DeepNorm: Use different scaling factors for different layers.
    Best choice for very large models (>24 layers).
    """
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            layer_depth: int = 0,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        # DeepNorm scaling: different for different layers
        self.alpha = (2 * layer_depth) ** 0.25 if layer_depth > 0 else 1.0
        self.beta = (8 * layer_depth) ** -0.25 if layer_depth > 0 else 1.0

        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ln_2 = norm_layer(d_model)

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        # Handle cross-attention keys/values
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        # DeepNorm: Apply different scaling
        x = self.alpha * q_x + self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        x = self.beta * x + self.mlp(self.ln_2(x))
        return x


# ============================================================================
# ENHANCED TRANSFORMER WITH CUSTOM BLOCK SELECTION
# ============================================================================

class EnhancedTransformer(nn.Module):
    """
    Enhanced Transformer that supports different attention block types.
    Drop-in replacement for OpenCLIP's Transformer class.
    """
    
    BLOCK_TYPES = {
        'standard': 'ResidualAttentionBlock',
        'parallel': ParallelResidualAttentionBlock,
        'linear': LinearAttentionResidualBlock,
        'deepnorm': DeepNormResidualAttentionBlock,
    }
    
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            block_type: str = 'standard',
            **kwargs
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.block_type = block_type

        # Select block class
        if block_type == 'standard':
            # Use original OpenCLIP ResidualAttentionBlock
            try:
                from transformer import ResidualAttentionBlock
                block_class = ResidualAttentionBlock
            except ImportError:
                raise ImportError("Cannot import ResidualAttentionBlock. Use other block types.")
        else:
            block_class = self.BLOCK_TYPES[block_type]

        # Create residual blocks
        self.resblocks = nn.ModuleList()
        for layer_idx in range(layers):
            if block_type == 'deepnorm':
                # Pass layer depth for DeepNorm scaling
                block = block_class(
                    width, heads, mlp_ratio, 
                    ls_init_value=ls_init_value, 
                    act_layer=act_layer, 
                    norm_layer=norm_layer,
                    layer_depth=layer_idx + 1,
                    **kwargs
                )
            else:
                block = block_class(
                    width, heads, mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            self.resblocks.append(block)

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0], 'mlp') and hasattr(self.resblocks[0].mlp, 'c_fc'):
            if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
                return self.resblocks[0].mlp.c_fc.int8_original_dtype
            return self.resblocks[0].mlp.c_fc.weight.dtype
        return torch.float32

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, 
                output_hiddens: Optional[bool] = False, control: Optional[torch.Tensor] = None):
        if output_hiddens:
            hiddens = []
        
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
            
            if output_hiddens:
                hiddens.append(x)
            if control is not None:
                x += control.pop()
        
        return (x, hiddens) if output_hiddens else x


# ============================================================================
# INTEGRATION UTILITIES
# ============================================================================

def replace_transformer_in_model(model, block_type: str = 'parallel', **kwargs):
    """
    Replace the transformer in an OpenCLIP model with enhanced version.
    
    Args:
        model: OpenCLIP model instance
        block_type: Type of attention block to use
        **kwargs: Additional arguments for the attention blocks
    
    Returns:
        Modified model with enhanced transformer
    """
    if hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
        # Vision transformer
        old_transformer = model.visual.transformer
        model.visual.transformer = EnhancedTransformer(
            width=old_transformer.width,
            layers=old_transformer.layers,
            heads=old_transformer.resblocks[0].attn.num_heads,
            block_type=block_type,
            **kwargs
        )
        # Copy weights if same architecture
        if block_type in ['parallel', 'deepnorm']:
            copy_compatible_weights(old_transformer, model.visual.transformer)
    
    if hasattr(model, 'transformer'):
        # Text transformer
        old_transformer = model.transformer
        model.transformer = EnhancedTransformer(
            width=old_transformer.width,
            layers=old_transformer.layers,
            heads=old_transformer.resblocks[0].attn.num_heads,
            block_type=block_type,
            **kwargs
        )
        # Copy weights if same architecture
        if block_type in ['parallel', 'deepnorm']:
            copy_compatible_weights(old_transformer, model.transformer)
    
    return model


def copy_compatible_weights(source_transformer, target_transformer):
    """Copy weights from source to target transformer when architectures are compatible."""
    try:
        for src_block, tgt_block in zip(source_transformer.resblocks, target_transformer.resblocks):
            # Copy attention weights
            if hasattr(tgt_block, 'attn') and hasattr(src_block, 'attn'):
                tgt_block.attn.load_state_dict(src_block.attn.state_dict())
            
            # Copy MLP weights
            if hasattr(tgt_block, 'mlp') and hasattr(src_block, 'mlp'):
                tgt_block.mlp.load_state_dict(src_block.mlp.state_dict())
            
            # Copy layer norms
            if hasattr(tgt_block, 'ln_1') and hasattr(src_block, 'ln_1'):
                tgt_block.ln_1.load_state_dict(src_block.ln_1.state_dict())
            if hasattr(tgt_block, 'ln_2') and hasattr(src_block, 'ln_2'):
                tgt_block.ln_2.load_state_dict(src_block.ln_2.state_dict())
        
        print("Successfully copied compatible weights")
    except Exception as e:
        print(f"Warning: Could not copy weights - {e}")


def benchmark_attention_blocks(
    d_model: int = 768,
    n_head: int = 12,
    seq_len: int = 197,
    batch_size: int = 8,
    device: str = 'cuda',
    num_warmup: int = 10,
    num_iterations: int = 100
):
    """
    Benchmark different attention block types.
    
    Returns:
        Dictionary with performance metrics for each block type
    """
    import time
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # Define blocks to test
    blocks = {
        'parallel': ParallelResidualAttentionBlock(d_model, n_head).to(device),
        'linear': LinearAttentionResidualBlock(d_model, n_head).to(device),
        'deepnorm': DeepNormResidualAttentionBlock(d_model, n_head, layer_depth=6).to(device),
    }
    
    results = {}
    
    for name, block in blocks.items():
        block.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = block(x)
        
        # Benchmark
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                output = block(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations * 1000  # Convert to ms
        
        # Memory usage
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = block(x)
            memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            memory_used = 0  # CPU memory tracking is complex
        
        # Parameter count
        param_count = sum(p.numel() for p in block.parameters())
        
        results[name] = {
            'avg_time_ms': avg_time,
            'memory_mb': memory_used,
            'parameters': param_count,
            'output_shape': output.shape
        }
        
        print(f"{name:12s}: {avg_time:6.2f}ms, {memory_used:6.1f}MB, {param_count:8d} params")
    
    return results


if __name__ == "__main__":
    print("Custom Attention Blocks for OpenCLIP Integration")
    print("=" * 50)
    
    # Example usage
    print("\n1. Creating enhanced transformer...")
    transformer = EnhancedTransformer(
        width=768,
        layers=12,
        heads=12,
        block_type='parallel'
    )
    print(f"Created {transformer.block_type} transformer with {transformer.layers} layers")
    
    print("\n2. Running benchmark...")
    if torch.cuda.is_available():
        results = benchmark_attention_blocks()
        
        print("\nBenchmark Results:")
        print("-" * 50)
        for name, metrics in results.items():
            print(f"{name:15s}: {metrics['avg_time_ms']:6.2f}ms")
    else:
        print("CUDA not available, skipping benchmark")
    
    print("\n3. Integration example...")
    print("To integrate into OpenCLIP:")
    print("  model = replace_transformer_in_model(clip_model, 'parallel')")
    print("  # Your model now uses ParallelResidualAttentionBlock!")