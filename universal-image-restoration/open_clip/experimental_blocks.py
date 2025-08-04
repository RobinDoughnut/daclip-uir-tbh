import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from collections import OrderedDict
import math

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
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
# EXPERIMENTAL RESIDUAL ATTENTION BLOCKS
# ============================================================================

class PostNormResidualAttentionBlock(nn.Module):
    """
    Post-norm variant: Apply LayerNorm after attention/MLP
    """
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = norm_layer(d_model)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ln_2 = norm_layer(d_model)
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # Post-norm: Apply norm after residual connection
        x = x + self.ls_1(self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0])
        x = self.ln_1(x)
        
        x = x + self.ls_2(self.mlp(x))
        x = self.ln_2(x)
        return x

class ParallelResidualAttentionBlock(nn.Module):
    """
    Parallel residual connections: Apply attention and MLP in parallel
    """
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        
        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # Parallel: Apply attention and MLP simultaneously
        attn_out = self.ls_1(self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), 
                                      need_weights=False, attn_mask=attn_mask)[0])
        mlp_out = self.ls_2(self.mlp(self.ln_2(x)))
        
        return x + attn_out + mlp_out

class DeepNormResidualAttentionBlock(nn.Module):
    """
    DeepNorm: Use different scaling factors for different layers
    """
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            layer_depth: int = 0,  # Current layer depth
    ):
        super().__init__()
        
        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        
        # DeepNorm scaling: different for different layers
        self.alpha = (2 * layer_depth) ** 0.25
        self.beta = (8 * layer_depth) ** -0.25
        
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ln_2 = norm_layer(d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # DeepNorm: Apply different scaling
        x = self.alpha * x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), 
                                      need_weights=False, attn_mask=attn_mask)[0]
        x = self.beta * x + self.mlp(self.ln_2(x))
        return x

class LinearAttentionResidualBlock(nn.Module):
    """
    Linear Attention: O(n) complexity instead of O(n²)
    """
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
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

        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ln_2 = norm_layer(d_model)
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def linear_attention(self, q, k, v):
        # Linear attention: O(n) complexity
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        kv = torch.einsum('bnhd,bnhe->bhde', k, v)
        qkv = torch.einsum('bnhd,bhde->bnhe', q, kv)
        
        k_sum = k.sum(dim=1, keepdim=True)
        qk = torch.einsum('bnhd,bnhd->bnh', q, k_sum)
        
        return qkv / qk.unsqueeze(-1)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        x_norm = self.ln_1(x)
        
        q = self.q_proj(x_norm).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        
        attn_out = self.linear_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, C)
        attn_out = self.out_proj(attn_out)
        
        x = x + self.ls_1(attn_out)
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x

class SparseAttentionResidualBlock(nn.Module):
    """
    Sparse Attention: Only attend to a subset of tokens
    """
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            sparsity_ratio: float = 0.5,  # Only attend to 50% of tokens
    ):
        super().__init__()
        
        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.sparsity_ratio = sparsity_ratio

        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ln_2 = norm_layer(d_model)
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def sparse_attention(self, x, attn_mask=None):
        B, N, C = x.shape
        x = x.transpose(0, 1)  # (N, B, C)
        
        # Create sparse attention mask
        if self.training:
            # During training, randomly select tokens to attend to
            num_attend = int(N * self.sparsity_ratio)
            attend_indices = torch.randperm(N)[:num_attend].sort()[0]
            sparse_mask = torch.zeros(N, N, device=x.device)
            sparse_mask[attend_indices[:, None], attend_indices] = 1
        else:
            # During inference, use fixed pattern (e.g., local attention)
            sparse_mask = torch.eye(N, device=x.device)
            # Add local connections
            for i in range(N):
                start = max(0, i - 2)
                end = min(N, i + 3)
                sparse_mask[i, start:end] = 1
        
        # Apply sparse attention
        attn_out, _ = self.attn(x, x, x, attn_mask=sparse_mask)
        return attn_out.transpose(0, 1)  # (B, N, C)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.sparse_attention(self.ln_1(x), attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x