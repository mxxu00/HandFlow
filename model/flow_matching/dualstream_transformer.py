"""
Dual-stream transformer for flow matching on hand pose sequences.

Derived from ShapeR/model/flow_matching/dualstream_transformer.py with changes:
  - Import path adjusted from `from model import attention` (same, works as-is)
  - Renamed internal variables: img→z (latent stream), txt→cond (condition stream)
  - Added RoPE support: position IDs + rope_module passed through block calls

Architecture (Flux-style):
  - DoubleStreamBlock: joint attention over (latent z_t, condition context)
  - SingleStreamBlock: self-attention on concatenated tokens
  - LastLayer: final projection to output dimension
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.
#
# This file is inspired by the Flux dual stream transformer architecture.
# Original source: https://github.com/black-forest-labs/flux
# Original license: Apache License 2.0

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from einops import rearrange
from model import attention
from model.pose_embedding import RotaryEmbedding
from torch import nn, Tensor


class FlowMatchingTransformer(nn.Module):
    """Transformer model for flow matching on sequences."""

    def __init__(
        self,
        in_channels,
        out_channels,
        use_context_in,
        context_in_dim,
        use_txt_in,
        vec_in_dim,
        use_pre_text_attn,
        config,
    ):
        super().__init__()

        self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_txt_in = use_txt_in
        self.use_context_in = use_context_in
        self.use_pre_text_attn = use_pre_text_attn
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} must be divisible by num_heads {config.num_heads}"
            )
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.pc_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        if self.use_txt_in:
            self.vector_in = MLPEmbedder(vec_in_dim, self.hidden_size)

        self.dropout = getattr(config, 'dropout', 0.1)

        if self.use_pre_text_attn:
            self.text_attn_in = nn.Linear(context_in_dim, self.hidden_size)
            self.text_blocks = nn.ModuleList(
                [
                    DoubleStreamBlock(
                        self.hidden_size,
                        self.num_heads,
                        mlp_ratio=config.mlp_ratio,
                        qkv_bias=config.qkv_bias,
                        dropout=self.dropout,
                    )
                    for _ in range(4)
                ]
            )

        if self.use_context_in:
            self.dino_in = nn.Linear(context_in_dim, self.hidden_size)
            self.double_blocks = nn.ModuleList(
                [
                    DoubleStreamBlock(
                        self.hidden_size,
                        self.num_heads,
                        mlp_ratio=config.mlp_ratio,
                        qkv_bias=config.qkv_bias,
                        dropout=self.dropout,
                    )
                    for _ in range(config.depth)
                ]
            )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size, self.num_heads, mlp_ratio=config.mlp_ratio,
                    dropout=self.dropout,
                )
                for _ in range(
                    config.depth_single_blocks
                    + (config.depth if not self.use_context_in else 0)
                )
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        pc: Tensor,
        dino: Tensor,
        timesteps: Tensor,
        y: Tensor,
        txt_tokens: Tensor,
        z_positions: Tensor = None,
        cond_positions: Tensor = None,
        rope_module=None,
    ) -> Tensor:
        if pc.ndim != 3 or (dino is not None and dino.ndim != 3):
            raise ValueError("Input pc and dino tensors must have 3 dimensions.")

        pc = self.pc_in(pc)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        intermediates = None
        if self.use_txt_in and y is not None:
            vec = vec + self.vector_in(y)

        if self.use_pre_text_attn:
            txt_tokens = self.text_attn_in(txt_tokens)
            for block in self.text_blocks:
                pc, txt_tokens = block(z=pc, cond=txt_tokens, vec=vec)

        cond_len = 0
        if self.use_context_in:
            dino = self.dino_in(dino)
            cond_len = dino.shape[1]
            for block in self.double_blocks:
                pc, dino = block(
                    z=pc, cond=dino, vec=vec,
                    z_positions=z_positions,
                    cond_positions=cond_positions,
                    rope_module=rope_module,
                )
            # Build combined positions (matching concatenation order: [cond, z])
            if z_positions is not None and cond_positions is not None:
                combined_positions = torch.cat([cond_positions, z_positions], dim=0)
            else:
                combined_positions = None
            pc = torch.cat((dino, pc), 1)

        for block_idx, block in enumerate(self.single_blocks):
            pc = block(pc, vec=vec, positions=combined_positions, rope_module=rope_module)
            if block_idx == 3:
                intermediates = pc

        if self.use_context_in:
            pc = pc[:, cond_len:, ...]

        pc = self.final_layer(pc, vec)
        return pc, intermediates


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """Create sinusoidal timestep embeddings."""
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    """Two-layer MLP for embedding."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    """Query-Key normalization for stable attention."""

    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    """Standard self-attention with QK normalization."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    """Adaptive layer norm modulation from timestep embedding."""

    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> Tuple[ModulationOut, Optional[ModulationOut]]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(
            self.multiplier, dim=-1
        )
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    """Double-stream attention block with separate z (latent) and cond (condition) streams."""

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float,
        qkv_bias: bool = False, dropout: float = 0.1,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        # z stream (latent) modules
        self.z_mod = Modulation(hidden_size, double=True)
        self.z_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.z_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )
        self.z_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.z_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.z_attn_drop = nn.Dropout(dropout)
        self.z_mlp_drop = nn.Dropout(dropout)

        # cond stream (condition) modules
        self.cond_mod = Modulation(hidden_size, double=True)
        self.cond_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cond_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )
        self.cond_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.cond_attn_drop = nn.Dropout(dropout)
        self.cond_mlp_drop = nn.Dropout(dropout)

    def forward(
        self,
        z: Tensor,
        cond: Tensor,
        vec: Tensor,
        z_positions: Tensor = None,
        cond_positions: Tensor = None,
        rope_module=None,
    ) -> Tuple[Tensor, Tensor]:
        z_mod1, z_mod2 = self.z_mod(vec)
        cond_mod1, cond_mod2 = self.cond_mod(vec)

        # prepare z stream for attention
        z_modulated = self.z_norm1(z)
        z_modulated = (1 + z_mod1.scale) * z_modulated + z_mod1.shift
        z_qkv = self.z_attn.qkv(z_modulated)
        z_q, z_k, z_v = rearrange(
            z_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        z_q, z_k = self.z_attn.norm(z_q, z_k, z_v)

        # prepare cond stream for attention
        cond_modulated = self.cond_norm1(cond)
        cond_modulated = (1 + cond_mod1.scale) * cond_modulated + cond_mod1.shift
        cond_qkv = self.cond_attn.qkv(cond_modulated)
        cond_q, cond_k, cond_v = rearrange(
            cond_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        cond_q, cond_k = self.cond_attn.norm(cond_q, cond_k, cond_v)

        # Apply RoPE (after QK-norm, before concatenation)
        if rope_module is not None and z_positions is not None:
            sin, cos = RotaryEmbedding.get_sin_cos_from_positions(
                z_positions, rope_module.inv_freq
            )
            z_q = RotaryEmbedding.apply_rope(z_q, sin, cos)
            z_k = RotaryEmbedding.apply_rope(z_k, sin, cos)
        if rope_module is not None and cond_positions is not None:
            sin, cos = RotaryEmbedding.get_sin_cos_from_positions(
                cond_positions, rope_module.inv_freq
            )
            cond_q = RotaryEmbedding.apply_rope(cond_q, sin, cos)
            cond_k = RotaryEmbedding.apply_rope(cond_k, sin, cos)

        # run actual attention (concat [cond, z] streams)
        q = torch.cat((cond_q, z_q), dim=2)
        k = torch.cat((cond_k, z_k), dim=2)
        v = torch.cat((cond_v, z_v), dim=2)

        attn = attention(q, k, v)
        cond_attn, z_attn = attn[:, : cond.shape[1]], attn[:, cond.shape[1] :]

        # calculate the z stream blocks
        z = z + z_mod1.gate * self.z_attn_drop(self.z_attn.proj(z_attn))
        z = z + z_mod2.gate * self.z_mlp_drop(self.z_mlp(
            (1 + z_mod2.scale) * self.z_norm2(z) + z_mod2.shift
        ))

        # calculate the cond stream blocks
        cond = cond + cond_mod1.gate * self.cond_attn_drop(self.cond_attn.proj(cond_attn))
        cond = cond + cond_mod2.gate * self.cond_mlp_drop(self.cond_mlp(
            (1 + cond_mod2.scale) * self.cond_norm2(cond) + cond_mod2.shift
        ))
        return z, cond


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: Optional[float] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)
        self.attn_drop = nn.Dropout(dropout)
        self.mlp_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        positions: Tensor = None,
        rope_module=None,
    ) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # Apply RoPE (after QK-norm, before attention)
        if rope_module is not None and positions is not None:
            sin, cos = RotaryEmbedding.get_sin_cos_from_positions(
                positions, rope_module.inv_freq
            )
            q = RotaryEmbedding.apply_rope(q, sin, cos)
            k = RotaryEmbedding.apply_rope(k, sin, cos)

        attn = self.attn_drop(attention(q, k, v))
        output = self.mlp_drop(self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2)))
        return x + mod.gate * output


class LastLayer(nn.Module):
    """Final projection layer with adaptive layer norm."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
