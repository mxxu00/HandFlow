"""
RoPE rotary positional encoding, reused by the Denoiser and Transformer.
Extracted from model/vae/autoencoder.py.
"""

from typing import Tuple

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """RoPE rotary positional encoding (applied to Q/K)."""

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dim must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_sin_cos(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device=device, dtype=dtype))
        sin = freqs.sin()
        cos = freqs.cos()
        return sin, cos

    @staticmethod
    def get_sin_cos_from_positions(
        positions: torch.Tensor,
        inv_freq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sin/cos from arbitrary position indices (supports repeated indices).

        Args:
            positions: (L,) int — arbitrary position indices, duplicates allowed
            inv_freq:  (D/2,)   — from RotaryEmbedding.inv_freq

        Returns:
            sin, cos: each (L, D/2)
        """
        freqs = torch.einsum("i,j->ij", positions.float(), inv_freq.to(positions))
        return freqs.sin(), freqs.cos()

    @staticmethod
    def apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, L, D)
        sin/cos: (L, D/2)
        """
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        sin = sin.unsqueeze(0).unsqueeze(0)
        cos = cos.unsqueeze(0).unsqueeze(0)
        x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot.flatten(-2)
