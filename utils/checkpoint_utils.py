"""
Checkpoint loading utilities.
"""

import gc
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from model.flow_matching.denoiser import HandPoseDenoiser


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def _load_state_dict(
    ckpt_path: Path, device: torch.device
) -> Tuple[Dict[str, torch.Tensor], Optional[int]]:
    # Load to CPU first to avoid a large checkpoint filling GPU memory
    raw = None
    if ckpt_path.is_dir():
        state_file = ckpt_path / "mp_rank_00_model_states.pt"
        if not state_file.exists():
            raise FileNotFoundError(f"DeepSpeed checkpoint not found: {state_file}")
        raw = torch.load(state_file, map_location="cpu", weights_only=False)
    else:
        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    state_dict = raw.get("module", raw) if isinstance(raw, dict) else raw
    epoch = raw.get("epoch") if isinstance(raw, dict) else None
    del raw
    gc.collect()

    # Move state_dict to the target device
    state_dict = {k: v.to(device) for k, v in state_dict.items()}
    return _clean_state_dict(state_dict), epoch


def load_denoiser_from_ckpt(
    config,
    ckpt_path: str,
    device: torch.device,
) -> HandPoseDenoiser:
    """Load HandPoseDenoiser, supporting either a DeepSpeed directory or a .pt file."""
    denoiser = HandPoseDenoiser(config).to(device)
    ckpt = Path(ckpt_path)
    state_dict, epoch = _load_state_dict(ckpt, device)
    result = denoiser.load_state_dict(state_dict, strict=False)
    # The frozen HaMeR backbone is loaded separately from HAMER_CKPT by HaMeRBackbone
    # and is intentionally absent from the released denoiser checkpoint; filter its keys
    # out of the warning so the log stays focused on genuinely unexpected mismatches.
    missing = [k for k in result.missing_keys if not k.startswith("hamer_backbone.")]
    unexpected = [k for k in result.unexpected_keys if not k.startswith("hamer_backbone.")]
    if missing:
        print(f"[Denoiser] Missing keys: {missing}")
    if unexpected:
        print(f"[Denoiser] Unexpected keys: {unexpected}")
    epoch_str = epoch if epoch is not None else "?"
    print(f"[Denoiser] Loaded from {ckpt}  (epoch={epoch_str})")
    del state_dict
    gc.collect()
    torch.cuda.empty_cache()
    return denoiser
