"""
Inference and MPJPE evaluation utilities.

Provides:
- Overlapping-window inference (stride stitching + per-field overlap averaging)
- Sequence-level MPJPE evaluation
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from utils.eval_utils import compute_mpjpe_metrics
from utils.mano_utils import MANOForwardKinematics, compute_joints_global


def split_sequence_into_windows(
    seq_batch: dict,
    window_size: int,
    stride: int,
    device: torch.device,
) -> Tuple[dict, list[int]]:
    """
    Split a variable-length sequence batch into overlapping windows,
    returning a window batch and a list of sequence memberships.

    Args:
        seq_batch:  collated sequence batch; temporal dim shape (B, L, ...)
        window_size: window size
        stride:     window stride = window_size - overlap_size
        device:     target device

    Returns:
        win_batch:  dict of (N_wins_total, ...) tensors
        seq_ids:    list of length N_wins_total
    """
    import math

    TIME_KEYS = {
        "mano_params", "mano_trans", "padding_mask",
        "hamer_landmarks", "crop_intrinsics", "hamer_confidence",
        "images", "joint_3d", "image_tokens",
    }
    B = seq_batch["mano_params"].shape[0]

    all_windows: list[dict] = []
    seq_ids: list[int] = []

    def pad_sequence_length(nf: int, window_size: int, stride: int) -> int:
        if nf <= window_size:
            return window_size
        k = math.ceil((nf - window_size) / stride)
        return window_size + stride * k

    for b in range(B):
        pm = seq_batch.get("padding_mask")
        if pm is not None:
            nf = int((~pm[b]).sum().item())
            nf = max(nf, 1)
        else:
            nf = seq_batch["mano_params"].shape[1]

        valid_len = pad_sequence_length(nf, window_size, stride)
        num_wins = (valid_len - window_size) // stride + 1

        for w in range(num_wins):
            start = w * stride
            end = start + window_size
            win = {}
            for key, val in seq_batch.items():
                if not isinstance(val, torch.Tensor):
                    if isinstance(val, list):
                        win[key] = val[b]
                    else:
                        win[key] = val
                    continue
                if key in TIME_KEYS and val.ndim >= 2:
                    seq_len = val.shape[1]  # batched: dim 1 is time
                    if val.ndim >= 3:
                        if end <= seq_len:
                            win[key] = val[b, start:end]
                        else:
                            chunk = val[b, start:min(end, seq_len)]
                            pad_len = end - min(end, seq_len)
                            pad_shape = (pad_len,) + chunk.shape[1:]
                            pad_val = True if key == "padding_mask" else 0.0
                            pad_t = torch.full(pad_shape, pad_val, dtype=chunk.dtype, device=chunk.device)
                            win[key] = torch.cat([chunk, pad_t], dim=0)
                    else:
                        # ndim == 2: (B, L)
                        if end <= seq_len:
                            win[key] = val[b, start:end]
                        else:
                            chunk = val[b, start:min(end, seq_len)]
                            pad_len = end - min(end, seq_len)
                            pad_val = True if key == "padding_mask" else 0.0
                            pad_t = torch.full((pad_len,), pad_val, dtype=chunk.dtype, device=chunk.device)
                            win[key] = torch.cat([chunk, pad_t], dim=0)
                else:
                    win[key] = val[b] if val.ndim >= 1 else val

            # Skip windows that are all padding
            pm_win = win.get("padding_mask")
            if pm_win is not None and pm_win.all():
                continue

            all_windows.append(win)
            seq_ids.append(b)

    def collate_fn(batch):
        out = {}
        for key in batch[0]:
            vals = [b[key] for b in batch]
            if isinstance(vals[0], torch.Tensor):
                out[key] = torch.stack(vals)
            else:
                out[key] = vals
        return out

    if not all_windows:
        return collate_fn([{
            "mano_params": torch.zeros(1, window_size, 48),
            "mano_trans": torch.zeros(1, window_size, 3),
            "mano_betas": torch.zeros(1, 10),
            "padding_mask": torch.ones(1, window_size, dtype=torch.bool),
            "images": torch.zeros(1, window_size, 3, 256, 256, dtype=torch.uint8),
            "hamer_landmarks": torch.zeros(1, window_size, 21, 2),
            "crop_intrinsics": torch.zeros(1, window_size, 4),
            "hamer_confidence": torch.zeros(1, window_size),
            "side": "right",
            "source": "dexycb",
        }]), [0]

    win_batch = collate_fn(all_windows)
    return win_batch, seq_ids


@torch.no_grad()
def run_fm_inference_with_overlap(
    model_engine,
    denoiser,
    batch: Dict[str, torch.Tensor],
    window_size: int,
    overlap_size: int,
    ode_steps: int,
    device: torch.device,
    show_progress: bool = False,
    guidance=None,
    overlap_method: str = "vblend",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flow Matching inference with overlapping windows.

    overlap_method:
      "avg"    : run each window's ODE independently, then uniformly average
                 in x₁ space at the end
      "vblend" : blend velocity by center distance during ODE integration,
                 keeping a single trajectory

    Noise strategy (MagicAnimate style):
      - Initial noise is fixed per frame index: the pose/trans noise of frame f
        is globally unique, and all windows covering frame f share the same
        initial noise component.
      - beta is a single token for the whole sequence; all windows share the
        same beta noise.

    Args:
        model_engine:   DeepSpeed engine or plain model
        denoiser:       HandPoseDenoiser
        batch:          window batch (N_wins, T, ...)
        window_size:    window size
        overlap_size:   number of overlapping frames
        ode_steps:      number of ODE integration steps
        device:         torch device
        show_progress:  whether to show a progress bar
        guidance:       2D reprojection guidance
        overlap_method: "vblend" or "avg"

    Returns:
        pose_seq:   (nf, 48)  predicted pose (global_aa + joints_aa)
        trans_seq:  (nf, 3)   predicted trans
        betas_pred: (nf, 10)  predicted beta (same value for the whole sequence)
    """
    stride = window_size - overlap_size
    N_wins = batch["mano_params"].shape[0]
    T = window_size

    # Sequence length
    pm = batch.get("padding_mask")
    if pm is not None:
        last_win_valid = int((~pm[-1]).sum().item())
        nf = (N_wins - 1) * stride + last_win_valid
    else:
        nf = (N_wins - 1) * stride + T

    ms = denoiser.mano_slice
    flat_dim  = ms.flat_dim
    beta_dim  = ms.beta_dim    # 10
    pose_dim  = ms.pose_dim    # 48
    trans_dim = ms.trans_dim   # 3

    dtype = next(model_engine.parameters()).dtype

    # ── Build global initial noise per frame index ────────────────────────
    # beta: a single noise vector shared across the whole sequence
    # pose/trans: independent per frame; window w takes [w*stride, w*stride+T)
    n_noise_frames = (N_wins - 1) * stride + T   # covers all window positions
    global_noise_beta  = torch.randn(beta_dim,                    device=device, dtype=dtype)
    global_noise_pose  = torch.randn(n_noise_frames, pose_dim,   device=device, dtype=dtype)
    global_noise_trans = torch.randn(n_noise_frames, trans_dim,  device=device, dtype=dtype)

    # ── Select overlap inference path ─────────────────────────────────────
    if overlap_size > 0 and overlap_method == "vblend":
        # vblend: per-frame state; blend velocity by center distance during ODE
        x_pose  = global_noise_pose[:nf].clone()
        x_trans = global_noise_trans[:nf].clone()
        x_beta  = global_noise_beta.clone()

        center = (T - 1) / 2.0
        w_table = torch.zeros(N_wins, T, device=device, dtype=dtype)
        for w in range(N_wins):
            for t in range(T):
                w_table[w, t] = max(0.01, 1.0 - abs(t - center) / center)

        dt = 1.0 / ode_steps
        iterator = tqdm(range(ode_steps), desc="ODE", leave=False) if show_progress else range(ode_steps)

        for i in iterator:
            t_step = torch.full((N_wins,), i * dt, device=device, dtype=dtype)

            # Assemble window batch from per-frame state
            x_win = torch.empty(N_wins, flat_dim, device=device, dtype=dtype)
            for w in range(N_wins):
                s = w * stride
                e = min(s + T, nf)
                n = e - s
                x_win[w, ms.beta] = x_beta
                x_win[w, ms.poses] = torch.cat([
                    x_pose[s:e].reshape(-1),
                    global_noise_pose[s + n:s + T].reshape(-1),
                ])[:T * pose_dim]
                x_win[w, ms.trans] = torch.cat([
                    x_trans[s:e].reshape(-1),
                    global_noise_trans[s + n:s + T].reshape(-1),
                ])[:T * trans_dim]

            v_win = model_engine(x_win, t_step, batch)
            if guidance is not None and guidance.active:
                with torch.enable_grad():
                    v_win = guidance.compute_velocity_correction(v_win, x_win, t_step, batch)

            v_pose_w = v_win[:, ms.poses].reshape(N_wins, T, pose_dim)
            v_trans_w = v_win[:, ms.trans].reshape(N_wins, T, trans_dim)
            v_beta_w = v_win[:, ms.beta]

            # Weighted blending of per-frame velocity
            vp_acc = torch.zeros_like(x_pose)
            vt_acc = torch.zeros_like(x_trans)
            wt_acc = torch.zeros(nf, device=device, dtype=dtype)
            for w in range(N_wins):
                s = w * stride
                e = min(s + T, nf)
                for t in range(e - s):
                    gf = s + t
                    wgt = w_table[w, t]
                    vp_acc[gf] += v_pose_w[w, t] * wgt
                    vt_acc[gf] += v_trans_w[w, t] * wgt
                    wt_acc[gf] += wgt

            x_pose  = x_pose  + (vp_acc / wt_acc.unsqueeze(-1).clamp(min=0.01)) * dt
            x_trans = x_trans + (vt_acc / wt_acc.unsqueeze(-1).clamp(min=0.01)) * dt
            x_beta  = x_beta  + v_beta_w.mean(dim=0) * dt

        acc_pose  = x_pose
        acc_trans = x_trans
        acc_beta  = x_beta

    else:
        # avg: run each window's ODE independently, then uniformly average in x₁ space
        x = torch.empty(N_wins, flat_dim, device=device, dtype=dtype)
        for w in range(N_wins):
            s = w * stride
            x[w, ms.beta]  = global_noise_beta
            x[w, ms.poses] = global_noise_pose[s:s + T].reshape(-1)
            x[w, ms.trans] = global_noise_trans[s:s + T].reshape(-1)

        dt = 1.0 / ode_steps
        iterator = tqdm(range(ode_steps), desc="ODE", leave=False) if show_progress else range(ode_steps)

        for i in iterator:
            t_step = torch.full((N_wins,), i * dt, device=device, dtype=dtype)
            v = model_engine(x, t_step, batch)
            if guidance is not None and guidance.active:
                with torch.enable_grad():
                    v = guidance.compute_velocity_correction(v, x, t_step, batch)
            x = x + v * dt

        # Accumulate per frame in normalized x₁ space; average overlapping frames
        acc_pose  = torch.zeros(nf, pose_dim,  device=device, dtype=dtype)
        acc_trans = torch.zeros(nf, trans_dim, device=device, dtype=dtype)
        acc_beta  = torch.zeros(beta_dim,      device=device, dtype=dtype)
        cnt       = torch.zeros(nf,            device=device, dtype=dtype)

        for w in range(N_wins):
            s = w * stride
            e = min(s + T, nf)
            n = e - s
            acc_pose[s:e]  += x[w, ms.poses].reshape(T, pose_dim)[:n]
            acc_trans[s:e] += x[w, ms.trans].reshape(T, trans_dim)[:n]
            acc_beta       += x[w, ms.beta]
            cnt[s:e]       += 1

        acc_pose  /= cnt.unsqueeze(-1).clamp(min=1)
        acc_trans /= cnt.unsqueeze(-1).clamp(min=1)
        acc_beta  /= N_wins

    # ── Denormalize ──────────────────────────────────────────────────────
    pose_seq  = acc_pose  * ms.pose_std_  + ms.pose_mean_   # (nf, 48)
    trans_seq = acc_trans * ms.trans_std_ + ms.trans_mean_  # (nf, 3)
    beta_orig = acc_beta  * ms.beta_std_  + ms.beta_mean_   # (10,)
    betas_pred = beta_orig.unsqueeze(0).expand(nf, -1)      # (nf, 10)

    return pose_seq, trans_seq, betas_pred


@torch.no_grad()
def evaluate_sequence_mpjpe(
    pose_pred: torch.Tensor,
    trans_pred: torch.Tensor,
    betas_pred: torch.Tensor,
    pose_gt: torch.Tensor,
    trans_gt: torch.Tensor,
    betas_gt: torch.Tensor,
    sides: List[str],
    mano_fk: MANOForwardKinematics,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Sequence-level MPJPE evaluation.

    Args:
        pose_pred:   (T, 48) predicted pose
        trans_pred:  (T, 3)  predicted trans
        betas_pred:  (T, 10) predicted beta
        pose_gt:     (T, 48) GT pose
        trans_gt:    (T, 3)  GT trans
        betas_gt:    (T, 10) GT beta
        sides:       list of length T
        mano_fk:     MANO FK module
        valid_mask:  (T,) bool numpy array
    """
    joints_gt_seq = compute_joints_global(
        pose_gt, betas_gt, trans_gt, sides, mano_fk
    )
    joints_pred_seq = compute_joints_global(
        pose_pred, betas_pred, trans_pred, sides, mano_fk
    )
    metrics = compute_mpjpe_metrics(
        joints_gt_seq, joints_pred_seq, valid_mask=valid_mask
    )
    return {
        "a_mpjpe_mm": metrics["a_mpjpe"],
        "ra_mpjpe_mm": metrics["ra_mpjpe"],
        "pa_mpjpe_mm": metrics["pa_mpjpe"],
        "w_mpjpe_mm": metrics["w_mpjpe"],
        "wa_mpjpe_mm": metrics["wa_mpjpe"],
        "a_mpjpe_per_frame": metrics["a_mpjpe_per_frame"],
        "ra_mpjpe_per_frame": metrics["ra_mpjpe_per_frame"],
        "pa_mpjpe_per_frame": metrics["pa_mpjpe_per_frame"],
        "w_mpjpe_per_frame": metrics["w_mpjpe_per_frame"],
        "wa_mpjpe_per_frame": metrics["wa_mpjpe_per_frame"],
    }


@torch.no_grad()
def evaluate_fm_on_loader(
    model_engine,
    denoiser,
    mano_fk: MANOForwardKinematics,
    loader,
    device: torch.device,
    ode_steps: int,
    max_batches: Optional[int],
    is_master: bool,
    window_size: int,
    overlap_size: int,
    guidance=None,
    overlap_method: str = "vblend",
) -> Dict[str, float]:
    """
    Evaluate the Flow Matching model's MPJPE on a DataLoader.

    Args:
        model_engine: DeepSpeed engine or plain model
        denoiser:     HandPoseDenoiser
        mano_fk:      MANO FK module
        loader:       DataLoader (full_sequence mode)
        device:       torch device
        ode_steps:    number of ODE integration steps
        max_batches:  maximum number of batches (None = all)
        is_master:    whether this is the master process
        window_size:  window size
        overlap_size: number of overlapping frames
    """
    model_engine.eval()
    n_frames = 0
    sum_ra = 0.0
    sum_pa = 0.0
    sum_w = 0.0
    sum_wa = 0.0
    stride = window_size - overlap_size

    for batch_idx, batch in enumerate(tqdm(loader, desc="MPJPE eval", leave=False, disable=not is_master)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        B = batch["mano_params"].shape[0]

        for b in range(B):
            pm_b = batch.get("padding_mask")
            if pm_b is not None:
                nf = int((~pm_b[b]).sum().item())
                nf = max(nf, 1)
            else:
                nf = batch["mano_params"].shape[1]

            single = {k: (v[b:b+1] if isinstance(v, torch.Tensor) else v[b] if isinstance(v, list) else v)
                      for k, v in batch.items()}
            win_batch, _ = split_sequence_into_windows(single, window_size, stride, device)
            win_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in win_batch.items()}

            pose_seq, trans_seq, betas_pred_b = run_fm_inference_with_overlap(
                model_engine, denoiser, win_batch,
                window_size, overlap_size, ode_steps, device,
                show_progress=False,
                guidance=guidance,
                overlap_method=overlap_method,
            )

            betas_gt_b = batch["mano_betas"][b:b+1].expand(nf, -1)
            sides_b = [batch.get("side", ["right"] * nf)]
            if isinstance(sides_b[0], list):
                sides_b = sides_b[0]
            else:
                sides_b = [sides_b[0]] * nf
            gt_pose = batch["mano_params"][b, :nf]
            gt_trans = batch["mano_trans"][b, :nf]
            gt_pm = batch.get("padding_mask")
            gt_valid = (~gt_pm[b, :nf]) if gt_pm is not None else torch.ones(nf, dtype=torch.bool, device=device)

            metrics = evaluate_sequence_mpjpe(
                pose_seq, trans_seq, betas_pred_b,
                gt_pose, gt_trans, betas_gt_b,
                sides_b, mano_fk, valid_mask=gt_valid.cpu().numpy(),
            )

            n_valid = gt_valid.sum().item()
            sum_ra += metrics["ra_mpjpe_per_frame"].sum()
            sum_pa += metrics["pa_mpjpe_per_frame"].sum()
            sum_w += metrics["w_mpjpe_per_frame"].sum()
            sum_wa += metrics["wa_mpjpe_per_frame"].sum()
            n_frames += n_valid

    # Distributed reduction
    try:
        from deepspeed import comm as dist
        if dist.is_initialized():
            metrics_tensor = torch.tensor(
                [sum_ra, sum_pa, sum_w, sum_wa, float(n_frames)],
                dtype=torch.float32, device=device,
            )
            dist.all_reduce(metrics_tensor)
            sum_ra, sum_pa, sum_w, sum_wa, n_frames = metrics_tensor.tolist()
    except ImportError:
        pass

    if n_frames == 0:
        return {
            "ra_mpjpe_mm": float("nan"),
            "pa_mpjpe_mm": float("nan"),
            "w_mpjpe_mm": float("nan"),
            "wa_mpjpe_mm": float("nan"),
        }

    return {
        "ra_mpjpe_mm": sum_ra / n_frames,
        "pa_mpjpe_mm": sum_pa / n_frames,
        "w_mpjpe_mm": sum_w / n_frames,
        "wa_mpjpe_mm": sum_wa / n_frames,
    }
