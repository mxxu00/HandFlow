"""
HaMeR ViT-Huge backbone wrapper (frozen).

Provides image token extraction for the 2D condition stream.
Input: (B, 3, 256, 256) float32 [0,1]
Output: (B, 192, 1280) — 16x12 patches x 1280D
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Add HaMeR repository to the import path
_HAMER_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "hamer"
if _HAMER_ROOT.exists() and str(_HAMER_ROOT) not in sys.path:
    sys.path.insert(0, str(_HAMER_ROOT))


class HaMeRBackbone(nn.Module):
    """
    Frozen HaMeR ViT-Huge backbone.

    Internal pipeline:
      1. ImageNet normalization (mean/std)
      2. Crop 32px from each horizontal side: (B,3,256,256) -> (B,3,256,192)
      3. Patch embedding: 192/16 x 256/16 = 12x16 = 192 tokens x 1280D
      4. 32-layer ViT-H Transformer
      5. Reshape output -> (B, 192, 1280)
    """

    # ImageNet normalization parameters
    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__()
        # Import the ViT backbone directly, bypassing hamer/models/__init__.py (avoids smplx dependency)
        import importlib.util
        vit_path = _HAMER_ROOT / "hamer" / "models" / "backbones" / "vit.py"
        spec = importlib.util.spec_from_file_location("hamer_vit", str(vit_path))
        vit_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vit_mod)
        ViT = vit_mod.ViT

        self.backbone = ViT(
            img_size=(256, 192),
            patch_size=16,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            ratio=1,
            use_checkpoint=False,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.55,
        )
        self.backbone.eval()

        # Load pretrained weights
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in ckpt:
                state_dict = {
                    k.replace("backbone.", ""): v
                    for k, v in ckpt["state_dict"].items()
                    if k.startswith("backbone.")
                }
            else:
                state_dict = ckpt
            self.backbone.load_state_dict(state_dict, strict=False)

        # Freeze all parameters
        self.backbone.requires_grad_(False)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Register normalization buffers
        self.register_buffer("_mean", self._IMAGENET_MEAN.view(1, 3, 1, 1))
        self.register_buffer("_std", self._IMAGENET_STD.view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 256, 256) float32 in range [0, 1]
        Returns:
            tokens: (B, 192, 1280) patch tokens
        """
        # ImageNet normalization
        x = (x - self._mean) / self._std

        # HaMeR internal crop: 32px from each side (256 -> 192)
        x = x[:, :, :, 32:-32]

        # Backbone forward
        feat_map = self.backbone.forward_features(x)  # (B, 1280, Hp, Wp)
        B, C, Hp, Wp = feat_map.shape

        # Flatten into tokens
        tokens = feat_map.flatten(2).transpose(1, 2)  # (B, Hp*Wp, 1280)

        return tokens

    def train(self, mode: bool = True):
        """Always stay in eval mode."""
        super().train(False)
        self.backbone.eval()
        return self
