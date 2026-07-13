"""
Image Spatio-Temporal Refiner + Frame Compression.

Step 1: 8+8 alternating spatio-temporal attention over patch tokens of all frames
Step 2: 4 layers of cross-attention pooling compress 192 tokens per frame -> 1 token

Input: HaMeR backbone output (B, T, 192, 1280)
Output: (B, T, d_model) — one condensed image condition token per frame
"""

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from model import attention


# ──────────────────────────────────────────────────────────────────────────────
# 2D RoPE (consistent with the legacy TemporalRefiner)
# ──────────────────────────────────────────────────────────────────────────────


class RotaryEmbedding1D(nn.Module):
    """1D RoPE: used for temporal positional encoding."""

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, L, D), positions: (L,)
        """
        freqs = torch.einsum("i,j->ij", positions.float(), self.inv_freq.to(positions))
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class RotaryEmbedding2D(nn.Module):
    """2D RoPE: split head_dim in half and apply 1D RoPE with y/x coordinates respectively."""

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.dim_per_axis = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim_per_axis, 2).float() / self.dim_per_axis))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, L, D), positions: (L, 2) (y, x)
        """
        x_y, x_x = x.chunk(2, dim=-1)
        x_y = self._apply_1d(x_y, positions[:, 0])
        x_x = self._apply_1d(x_x, positions[:, 1])
        return torch.cat([x_y, x_x], dim=-1)

    def _apply_1d(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        freqs = torch.einsum("i,j->ij", pos.float(), self.inv_freq.to(pos))
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class RefinerBlock(nn.Module):
    """Pre-LN Transformer block with optional 2D RoPE."""

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rope: Optional[RotaryEmbedding2D] = None, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope is not None and positions is not None:
            q = rope(q, positions)
            k = rope(k, positions)
        x = attention(q, k, v)
        x = self.proj(x)
        x = residual + self.drop(x)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class CrossAttentionPoolBlock(nn.Module):
    """Cross-attention pooling: 1 learnable query token cross-attends K patches."""

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, d_model * 2)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.ff = nn.Sequential(nn.Linear(d_model, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """
        query: (B, 1, D), patches: (B, N, D)
        Returns: (B, 1, D)
        """
        residual = query
        q = self.q_proj(self.norm_q(query))
        kv = self.kv_proj(self.norm_kv(patches))
        k, v = rearrange(kv, "B N (K H D) -> K B H N D", K=2, H=self.num_heads)
        q = rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
        q = self.q_norm(q)
        k = self.k_norm(k)
        out = attention(q, k, v)  # (B, 1, H*D) = (B, 1, D)
        out = self.out_proj(out)
        query = residual + self.drop(out)
        query = query + self.drop(self.ff(self.norm_ff(query)))
        return query


# ──────────────────────────────────────────────────────────────────────────────
# Spatio-Temporal Refiner
# ──────────────────────────────────────────────────────────────────────────────


class SpatioTemporalRefiner(nn.Module):
    """
    8+8 alternating spatio-temporal attention.

    Even layers: intra-frame spatial attention (B*T, 192, dim)
    Odd layers:  cross-frame same-position attention (B, T, dim) per spatial position
    """

    def __init__(
        self,
        d_model: int,
        hamer_dim: int = 1280,
        spatial_layers: int = 8,
        temporal_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        n_patches_h: int = 16,
        n_patches_w: int = 12,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_patches = n_patches_h * n_patches_w
        self.depth = spatial_layers + temporal_layers
        self.spatial_layers = spatial_layers
        self.temporal_layers = temporal_layers

        self.input_proj = nn.Linear(hamer_dim, d_model)

        # 2D spatial positions
        y = torch.arange(n_patches_h)
        x = torch.arange(n_patches_w)
        self.register_buffer("spatial_positions", torch.cartesian_prod(y, x))

        self.rope = RotaryEmbedding2D(dim=d_model // num_heads)
        self.temporal_rope = RotaryEmbedding1D(dim=d_model // num_heads)
        self.blocks = nn.ModuleList([
            RefinerBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(self.depth)
        ])
        self.use_checkpoint = False

    def gradient_checkpointing_enable(self):
        self.use_checkpoint = True

    def forward(self, hamer_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hamer_features: (B, T, 192, 1280)
        Returns:
            (B, T, 192, d_model)
        """
        B, T, N, _ = hamer_features.shape
        x = self.input_proj(hamer_features)

        spatial_pos = self.spatial_positions                           # (N, 2)
        temporal_pos = torch.arange(T, device=x.device)               # (T,)
        temporal_pos_expanded = temporal_pos.unsqueeze(1).expand(T, N).reshape(T * N)  # (T*N,)

        for i, blk in enumerate(self.blocks):
            if i % 2 == 0:
                # Intra-frame spatial attention (2D RoPE: y, x)
                x = x.reshape(B * T, N, self.d_model)
                x = blk(x, rope=self.rope, positions=spatial_pos)
                x = x.reshape(B, T, N, self.d_model)
            else:
                # Cross-frame global attention (1D temporal RoPE: frame index t)
                x = x.reshape(B, T * N, self.d_model)
                x = blk(x, rope=self.temporal_rope, positions=temporal_pos_expanded)
                x = x.reshape(B, T, N, self.d_model)

        return x


# ──────────────────────────────────────────────────────────────────────────────
# Frame Compressor (Cross-Attention Pooling)
# ──────────────────────────────────────────────────────────────────────────────


class FrameCompressor(nn.Module):
    """
    Compress N patch tokens of each frame into 1 condensed token.
    Uses a learnable query token with multi-layer cross-attention.
    Includes input_proj: supports arbitrary input dims (e.g. HaMeR 1280D -> d_model).
    """

    def __init__(
        self,
        d_model: int,
        input_dim: int | None = None,
        num_heads: int = 8,
        num_layers: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim else nn.Identity()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.layers = nn.ModuleList([
            CrossAttentionPoolBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (B, T, N, input_dim or d_model)
        Returns: (B, T, d_model)
        """
        B, T, N, _ = patches.shape
        patches = self.input_proj(patches)                    # (B, T, N, d_model)
        patches = patches.reshape(B * T, N, -1)              # (B*T, N, D)
        q = self.query.expand(B * T, -1, -1)                 # (B*T, 1, D)
        for layer in self.layers:
            q = layer(q, patches)
        return q.reshape(B, T, -1)
