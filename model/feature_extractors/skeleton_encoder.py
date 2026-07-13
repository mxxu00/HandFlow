"""
Ray Direction Skeleton Encoder.

2D skeleton -> inverse intrinsics -> ray direction -> high-frequency sinusoidal PE -> flatten -> MLP -> embed_dim.
Outputs one skeleton token per frame.

Input: landmarks (B, T, 21, 2) + crop_intrinsics (B, T, 4)
Output: (B, T, embed_dim)
"""

import torch
import torch.nn as nn


class RayDirectionSkeletonEncoder(nn.Module):
    """
    Encodes a 2D skeleton into one token per frame via ray direction.

    Pipeline:
      1. un-normalize landmarks -> pixel coordinates (in crop coordinate system)
      2. ray = [(u-cx)/fx, (v-cy)/fy, 1] -> normalize -> (21, 3)
      3. High-frequency sinusoidal PE: [sin(2^L * ray), cos(2^L * ray)] L=0..num_freqs-1
      4. flatten: 21 joints x (3 x 2 x num_freqs) per joint
      5. MLP -> embed_dim
    """

    def __init__(
        self,
        n_joints: int = 21,
        num_freqs: int = 6,
        embed_dim: int = 512,
        crop_h: int = 256,
        crop_w: int = 256,
        mlp_hidden: int = 512,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.num_freqs = num_freqs
        self.crop_h = crop_h
        self.crop_w = crop_w

        # Per-joint dim after PE: 3 (ray) x 2 (sin+cos) x num_freqs
        pe_dim = 3 * 2 * num_freqs  # = 36
        flat_dim = n_joints * pe_dim  # = 756

        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def forward(
        self,
        landmarks: torch.Tensor,
        crop_intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            landmarks: (B, T, 21, 2) normalized [0, 1]
            crop_intrinsics: (B, T, 4) = [fx, fy, cx, cy] in crop coords
        Returns:
            (B, T, embed_dim)
        """
        B, T, J, _ = landmarks.shape
        fx = crop_intrinsics[..., 0:1]  # (B, T, 1)
        fy = crop_intrinsics[..., 1:2]
        cx = crop_intrinsics[..., 2:3]
        cy = crop_intrinsics[..., 3:4]

        # un-normalize -> crop pixel coordinates
        u = landmarks[..., 0:1] * self.crop_w  # (B, T, 21, 1)
        v = landmarks[..., 1:2] * self.crop_h

        # ray direction: [(u-cx)/fx, (v-cy)/fy, 1]
        ray_x = (u - cx.unsqueeze(-2)) / (fx.unsqueeze(-2) + 1e-8)
        ray_y = (v - cy.unsqueeze(-2)) / (fy.unsqueeze(-2) + 1e-8)
        ray_z = torch.ones_like(ray_x)
        ray = torch.cat([ray_x, ray_y, ray_z], dim=-1)  # (B, T, 21, 3)

        # normalize ray direction
        ray = ray / (ray.norm(dim=-1, keepdim=True) + 1e-8)

        # High-frequency sinusoidal PE
        pe_parts = []
        for L in range(self.num_freqs):
            freq = 2.0 ** L
            pe_parts.append(torch.sin(freq * ray))
            pe_parts.append(torch.cos(freq * ray))
        # pe: (B, T, 21, 3 * 2 * num_freqs) = (B, T, 21, 36)
        pe = torch.cat(pe_parts, dim=-1)

        # flatten all joints: (B, T, 21 * 36) = (B, T, 756)
        flat = pe.reshape(B, T, -1)

        # MLP -> embed_dim
        return self.mlp(flat)
