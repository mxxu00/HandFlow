"""
MANO FK and joint computation utilities.

pose format: 48D axis-angle = [global_aa(3) | joints_aa(45)]
"""

from typing import List

import numpy as np
import torch
from manopth.manolayer import ManoLayer


class MANOForwardKinematics:
    """Batched FK, supports mixed left/right hands."""

    def __init__(self, mano_root: str, device: torch.device):
        self.device = device
        self._layers = {}
        for side in ("right", "left"):
            layer = ManoLayer(
                mano_root=mano_root,
                side=side,
                use_pca=False,
                flat_hand_mean=True,
            ).to(device).eval()
            for p in layer.parameters():
                p.requires_grad_(False)
            self._layers[side] = layer

    def joints_torch(
        self,
        pose: torch.Tensor,
        betas: torch.Tensor,
        trans: torch.Tensor,
        sides: List[str],
    ) -> torch.Tensor:
        """
        Args:
            pose:  (N, 48)  — global_aa(3) + joints_aa(45)
            betas: (N, 10)
            trans: (N, 3)   — translation in meters
            sides: list of "right"/"left", length N
        Returns:
            joints: (N, 21, 3) in mm
        """
        n_samples = pose.shape[0]
        joints_out = torch.zeros(
            n_samples, 21, 3, device=pose.device, dtype=pose.dtype,
        )
        for side in ("right", "left"):
            idx = [i for i, s in enumerate(sides) if s == side]
            if not idx:
                continue
            idx_t = torch.tensor(idx, device=pose.device, dtype=torch.long)
            mano_pose = pose[idx_t]  # (K, 48) fed directly into ManoLayer
            _, joints = self._layers[side](
                mano_pose,
                th_betas=betas[idx_t],
                th_trans=trans[idx_t],
            )
            joints_out[idx_t] = joints.to(dtype=pose.dtype)
        return joints_out

    @torch.no_grad()
    def joints(
        self,
        pose: torch.Tensor,
        betas: torch.Tensor,
        trans: torch.Tensor,
        sides: List[str],
    ) -> torch.Tensor:
        return self.joints_torch(pose, betas, trans, sides)

    def verts_torch(
        self,
        pose: torch.Tensor,
        betas: torch.Tensor,
        trans: torch.Tensor,
        sides: List[str],
    ) -> torch.Tensor:
        """
        Args:
            pose:  (N, 48)  — global_aa(3) + joints_aa(45)
            betas: (N, 10)
            trans: (N, 3)   — translation in meters
            sides: list of "right"/"left", length N
        Returns:
            verts: (N, 778, 3) in mm
        """
        n_samples = pose.shape[0]
        verts_out = torch.zeros(
            n_samples, 778, 3, device=pose.device, dtype=pose.dtype,
        )
        for side in ("right", "left"):
            idx = [i for i, s in enumerate(sides) if s == side]
            if not idx:
                continue
            idx_t = torch.tensor(idx, device=pose.device, dtype=torch.long)
            mano_pose = pose[idx_t]
            verts, _ = self._layers[side](
                mano_pose,
                th_betas=betas[idx_t],
                th_trans=trans[idx_t],
            )
            verts_out[idx_t] = verts.to(dtype=pose.dtype)
        return verts_out

    @torch.no_grad()
    def verts(
        self,
        pose: torch.Tensor,
        betas: torch.Tensor,
        trans: torch.Tensor,
        sides: List[str],
    ) -> torch.Tensor:
        return self.verts_torch(pose, betas, trans, sides)

    def get_faces(self, side: str = "right") -> np.ndarray:
        """Return MANO face indices as (1538, 3) numpy int32 array."""
        f = self._layers[side].th_faces
        return f.cpu().numpy().reshape(-1, 3).astype(np.int32)


@torch.no_grad()
def compute_joints_global(
    pose: torch.Tensor,
    betas: torch.Tensor,
    trans: torch.Tensor,
    sides: List[str],
    mano_fk: MANOForwardKinematics,
) -> np.ndarray:
    """
    manopth FK → global 3D joints (metres, camera coordinate system).

    Args:
        pose:  (N, 48)  — global_aa(3) + joints_aa(45)
        betas: (N, 10)
        trans: (N, 3) in metres
        sides: length N
    Returns:
        joints_m: (N, 21, 3) float32, in metres
    """
    joints_mm = mano_fk.joints(pose, betas, trans, sides)
    return (joints_mm / 1000.0).cpu().numpy().astype(np.float32)


def compute_joints_global_torch(
    pose: torch.Tensor,
    betas: torch.Tensor,
    trans: torch.Tensor,
    sides: List[str],
    mano_fk: MANOForwardKinematics,
) -> torch.Tensor:
    """
    manopth FK → global 3D joints (metres), supports backpropagation.

    Args:
        pose:  (N, 48)
        betas: (N, 10)
        trans: (N, 3) in metres
        sides: length N
    Returns:
        joints_m: (N, 21, 3), in metres
    """
    joints_mm = mano_fk.joints_torch(pose, betas, trans, sides)
    return joints_mm / 1000.0


@torch.no_grad()
def compute_joints_global_sequence(
    pose: torch.Tensor,
    betas: torch.Tensor,
    trans: torch.Tensor,
    side: str,
    mano_fk: MANOForwardKinematics,
) -> np.ndarray:
    """Single-side sequence version: input (T, *), output (T, 21, 3)."""
    T = pose.shape[0]
    betas_seq = betas.unsqueeze(0).expand(T, -1)
    sides = [side] * T
    return compute_joints_global(pose, betas_seq, trans, sides, mano_fk)
