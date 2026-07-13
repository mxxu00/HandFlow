"""
Condition Builder (with cmask mechanism).

Concatenates image token + skeleton token into the condition stream.
cmask: learnable mask token; both training and inference use HaMeR confidence as a continuous mask.

Inputs:
  - image_tokens: (B, T, d_model) — from Image Refiner + Compression
  - skeleton_tokens: (B, T, d_model) — from Skeleton Encoder
  - hamer_confidence: (B, T) — HaMeR detection confidence

Output: condition (B, 2*T, d_model)
"""

import torch
import torch.nn as nn


class ConditionBuilder(nn.Module):
    def __init__(self, d_model: int, mask_ratio: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.mask_ratio = mask_ratio

        # Learnable mask token — initialized with truncated normal
        self.c_mask = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.c_mask, std=0.02)

    def forward(
        self,
        image_tokens: torch.Tensor,
        skeleton_tokens: torch.Tensor,
        hamer_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_tokens: (B, T, d_model)
            skeleton_tokens: (B, T, d_model)
            hamer_confidence: (B, T) — used as m_t during inference

        Returns:
            condition: (B, 2*T, d_model)
            m: (B, T) — actual mask values used (for logging)
        """
        B, T, D = image_tokens.shape

        # concat: (B, T, 2, d_model) -> (B, 2*T, d_model)
        tokens = torch.stack([image_tokens, skeleton_tokens], dim=2)  # (B, T, 2, D)
        tokens = tokens.reshape(B, 2 * T, D)

        # Use HaMeR confidence as a continuous mask (consistent between training and inference)
        m = hamer_confidence.float()  # (B, T)

        # Optional: randomly dropout some frames during training as regularization
        if self.training and self.mask_ratio > 0:
            keep = (torch.rand(B, T, device=tokens.device) >= self.mask_ratio).float()
            m = m * keep

        # Expand m to match 2*T tokens (the two tokens per frame share the same m)
        # m: (B, T) -> (B, 2*T)
        m_expanded = m.unsqueeze(-1).expand(-1, -1, 2).reshape(B, 2 * T)  # (B, 2*T)
        m_expanded = m_expanded.unsqueeze(-1)  # (B, 2*T, 1)

        # cmask: c'_t = m_t * c_t + (1 - m_t) * c_mask
        condition = m_expanded * tokens + (1 - m_expanded) * self.c_mask

        return condition, m
