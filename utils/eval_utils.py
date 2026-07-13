"""
Hand motion evaluation utilities.

Provides:
  - A-MPJPE, RA-MPJPE, PA-MPJPE, W-MPJPE, WA-MPJPE  (joints metrics)
  - MANO-PSNR                                (compression fidelity, dB)
  - Temporal Incremental RMSE (TI-RMSE)      (motion smoothness, mm)
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

def _compute_procrustes_transform(
    pred_c: np.ndarray,
    gt_c: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Compute the Procrustes transform (rotation + scale).

    Args:
        pred_c: (N, 3) centered predicted points
        gt_c: (N, 3) centered GT points

    Returns:
        R: (3, 3) rotation matrix
        scale: float scale factor
    """
    H = pred_c.T @ gt_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    scale = np.trace((pred_c @ R.T).T @ gt_c) / (np.linalg.norm(pred_c) ** 2 + 1e-8)
    return R, scale


def procrustes_align(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-frame Procrustes alignment (rotation + scale + translation).
    Input: (N, 21, 3). Used for PA-MPJPE."""
    pred_np = pred.cpu().numpy()
    gt_np   = gt.cpu().numpy()
    aligned = np.zeros_like(pred_np)
    for i in range(pred_np.shape[0]):
        p, g = pred_np[i], gt_np[i]
        p_mean = p.mean(axis=0)
        g_mean = g.mean(axis=0)
        R, scale = _compute_procrustes_transform(p - p_mean, g - g_mean)
        aligned[i] = scale * ((p - p_mean) @ R.T) + g_mean
    return torch.from_numpy(aligned).to(pred.device)


def first_n_frames_procrustes(
    joints_pred: np.ndarray,
    joints_gt: np.ndarray,
    valid_frames: np.ndarray,
    n_anchor: int = 2,
) -> np.ndarray:
    """
    W-MPJPE alignment: Procrustes (scale + rotation + translation) estimated
    from the first n_anchor valid frames, then applied globally to the entire
    sequence.

    The anchor frames are stacked into a single point cloud (n*21 points) and
    one optimal (s, R, t) is solved.  The same transform is applied to every
    frame — inter-frame relative motion is fully preserved.

    Args:
        joints_pred  : (T, 21, 3) metres
        joints_gt    : (T, 21, 3) metres
        valid_frames : 1-D array of valid frame indices (sorted)
        n_anchor     : number of leading valid frames to use for alignment

    Returns:
        aligned_pred : (T, 21, 3) metres
    """
    anchor_frames = valid_frames[:n_anchor]

    pred_anchor = joints_pred[anchor_frames].reshape(-1, 3)
    gt_anchor   = joints_gt[anchor_frames].reshape(-1, 3)

    pred_mean = pred_anchor.mean(axis=0)
    gt_mean   = gt_anchor.mean(axis=0)
    pred_c    = pred_anchor - pred_mean
    gt_c      = gt_anchor   - gt_mean

    R, scale = _compute_procrustes_transform(pred_c, gt_c)

    T_seq, J, _ = joints_pred.shape
    pred_flat    = joints_pred.reshape(-1, 3)
    aligned_flat = scale * (R @ (pred_flat - pred_mean).T).T + gt_mean
    return aligned_flat.reshape(T_seq, J, 3)


# Keep old name as alias for backward compatibility
def first_two_frames_procrustes(joints_pred, joints_gt, valid_frames):
    return first_n_frames_procrustes(joints_pred, joints_gt, valid_frames, n_anchor=2)


def whole_seq_procrustes(
    joints_pred: np.ndarray,
    joints_gt: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Global Procrustes alignment (rotation + scale + translation) estimated from
    all valid frames treated as a single point cloud.

    Inter-frame relative motion is 100% preserved because every frame is
    transformed by the same (R, scale, translation).

    Args:
        joints_pred : (T, 21, 3) metres
        joints_gt   : (T, 21, 3) metres
        valid_mask  : (T,) bool

    Returns:
        aligned_pred : (T, 21, 3) metres
    """
    pred_valid = joints_pred[valid_mask].reshape(-1, 3)   # (V*21, 3)
    gt_valid   = joints_gt[valid_mask].reshape(-1, 3)

    pred_mean = pred_valid.mean(axis=0)
    gt_mean   = gt_valid.mean(axis=0)
    pred_c    = pred_valid - pred_mean
    gt_c      = gt_valid   - gt_mean

    R, scale = _compute_procrustes_transform(pred_c, gt_c)

    T_seq, J, _ = joints_pred.shape
    pred_flat    = joints_pred.reshape(-1, 3)
    aligned_flat = scale * (R @ (pred_flat - pred_mean).T).T + gt_mean
    return aligned_flat.reshape(T_seq, J, 3)


def batched_procrustes_align(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Batched per-frame Procrustes alignment (rotation + scale + translation).
    GPU-accelerated replacement for the per-frame numpy loop in procrustes_align.
    Input: (N, K, 3). Returns: (N, K, 3).
    """
    pred_mean = pred.mean(dim=1, keepdim=True)
    gt_mean   = gt.mean(dim=1, keepdim=True)
    pred_c = pred - pred_mean
    gt_c   = gt - gt_mean
    H = torch.bmm(pred_c.transpose(1, 2), gt_c)          # (N, 3, 3)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2)))
    D = torch.ones_like(S)
    D[:, -1] = d.sign()
    R = torch.bmm(Vt.transpose(1, 2), torch.bmm(torch.diag_embed(D), U.transpose(1, 2)))
    pred_R = torch.bmm(pred_c, R.transpose(1, 2))      # (N, K, 3)
    scale = (pred_R * gt_c).sum(dim=(1, 2)) / ((pred_c ** 2).sum(dim=(1, 2)) + 1e-8)
    return scale.unsqueeze(-1).unsqueeze(-1) * pred_R + gt_mean


def _single_procrustes_torch(
    pred: torch.Tensor, gt: torch.Tensor
) -> tuple:
    """Procrustes (R + scale + translation) for a single point cloud pair."""
    pred_mean = pred.mean(dim=0)
    gt_mean = gt.mean(dim=0)
    pred_c = pred - pred_mean
    gt_c = gt - gt_mean
    H = pred_c.T @ gt_c
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.T @ U.T)
    D = torch.diag(torch.tensor([1.0, 1.0, d.sign().item()], device=pred.device, dtype=pred.dtype))
    R = Vt.T @ D @ U.T
    scale = ((pred_c @ R.T) * gt_c).sum() / ((pred_c ** 2).sum() + 1e-8)
    return R, scale, pred_mean, gt_mean


def first_n_frames_procrustes_torch(
    joints_pred: torch.Tensor,
    joints_gt: torch.Tensor,
    n_anchor: int = 2,
) -> torch.Tensor:
    """
    W-MPJPE alignment, torch version. Single sequence, valid frames only.
    Input: (V, 21, 3). Returns: (V, 21, 3).
    """
    anchor_pred = joints_pred[:n_anchor].reshape(-1, 3)
    anchor_gt = joints_gt[:n_anchor].reshape(-1, 3)
    R, scale, pred_mean, gt_mean = _single_procrustes_torch(anchor_pred, anchor_gt)
    flat = joints_pred.reshape(-1, 3)
    aligned = scale * (flat - pred_mean) @ R.T + gt_mean
    return aligned.reshape_as(joints_pred)


def whole_seq_procrustes_torch(
    joints_pred: torch.Tensor,
    joints_gt: torch.Tensor,
) -> torch.Tensor:
    """
    WA-MPJPE alignment, torch version. Single sequence, valid frames only.
    Input: (V, 21, 3). Returns: (V, 21, 3).
    """
    R, scale, pred_mean, gt_mean = _single_procrustes_torch(
        joints_pred.reshape(-1, 3), joints_gt.reshape(-1, 3)
    )
    flat = joints_pred.reshape(-1, 3)
    aligned = scale * (flat - pred_mean) @ R.T + gt_mean
    return aligned.reshape_as(joints_pred)


# ---------------------------------------------------------------------------
# Joints metrics
# ---------------------------------------------------------------------------

def compute_mpjpe_metrics(
    joints_gt_m: np.ndarray,
    joints_pred_m: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute five MPJPE variants.

    Alignment definitions:
      A-MPJPE  : absolute, no alignment at all — raw joint coordinate difference
      RA-MPJPE : per-frame root-relative (subtract wrist joint)
      PA-MPJPE : per-frame Procrustes (rotation + scale + translation)
      W-MPJPE  : Procrustes (scale + rotation + translation) estimated from the
                 first 2 valid frames (stacked as one 42-point cloud), then the
                 same (s, R, t) applied to the entire sequence
      WA-MPJPE : whole-sequence global Procrustes (rotation + scale + translation)
                 → all valid frames treated as one point cloud, one alignment for all
                 → inter-frame relative motion 100% preserved

    Args:
        joints_gt_m   : (T, 21, 3) GT joints in metres — FULL sequence
        joints_pred_m : (T, 21, 3) predicted joints in metres — FULL sequence
        valid_mask    : (T,) bool. If None, all frames used.

    Returns dict with scalar means (mm) and per-valid-frame arrays.
    """
    T = len(joints_gt_m)
    vm = valid_mask if valid_mask is not None else np.ones(T, dtype=bool)

    gt_v   = joints_gt_m[vm]    # (V, 21, 3)
    pred_v = joints_pred_m[vm]

    # ── A-MPJPE: absolute, no alignment ──────────────────────────────────
    a_err       = np.linalg.norm((pred_v - gt_v) * 1000.0, axis=-1)
    a_mpjpe_pf  = a_err.mean(axis=-1)

    # ── RA-MPJPE: per-frame root-relative ────────────────────────────────
    gt_rel   = gt_v   - gt_v[:, 0:1, :]
    pred_rel = pred_v - pred_v[:, 0:1, :]
    ra_err       = np.linalg.norm((pred_rel - gt_rel) * 1000.0, axis=-1)
    ra_mpjpe_pf  = ra_err.mean(axis=-1)

    # ── PA-MPJPE: per-frame Procrustes (rotation + scale + translation) ──
    pred_pa = procrustes_align(
        torch.from_numpy(pred_v), torch.from_numpy(gt_v)
    ).numpy()
    pa_err      = np.linalg.norm((pred_pa - gt_v) * 1000.0, axis=-1)
    pa_mpjpe_pf = pa_err.mean(axis=-1)

    # ── W-MPJPE: first-2-frames Procrustes (s + R + t), applied globally ─
    valid_frames = np.where(vm)[0]
    pred_w       = first_two_frames_procrustes(joints_pred_m, joints_gt_m, valid_frames)
    w_err        = np.linalg.norm((pred_w[vm] - gt_v) * 1000.0, axis=-1)
    w_mpjpe_pf   = w_err.mean(axis=-1)

    # ── WA-MPJPE: whole-sequence global Procrustes (R + scale + t) ───────
    pred_wa      = whole_seq_procrustes(joints_pred_m, joints_gt_m, vm)
    wa_err       = np.linalg.norm((pred_wa[vm] - gt_v) * 1000.0, axis=-1)
    wa_mpjpe_pf  = wa_err.mean(axis=-1)

    return {
        "a_mpjpe":             float(a_mpjpe_pf.mean()),
        "ra_mpjpe":            float(ra_mpjpe_pf.mean()),
        "pa_mpjpe":            float(pa_mpjpe_pf.mean()),
        "w_mpjpe":             float(w_mpjpe_pf.mean()),
        "wa_mpjpe":            float(wa_mpjpe_pf.mean()),
        "a_mpjpe_per_frame":   a_mpjpe_pf,
        "ra_mpjpe_per_frame":  ra_mpjpe_pf,
        "pa_mpjpe_per_frame":  pa_mpjpe_pf,
        "w_mpjpe_per_frame":   w_mpjpe_pf,
        "wa_mpjpe_per_frame":  wa_mpjpe_pf,
    }


# ---------------------------------------------------------------------------
# MANO-PSNR and Temporal Incremental RMSE
# ---------------------------------------------------------------------------

def compute_mano_psnr(
    joints_gt_m: np.ndarray,
    joints_pred_m: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    max_val_mm: float = 1000.0,
) -> float:
    """
    MANO-PSNR over valid frames (no alignment).

    MSE is computed over all valid frames × 21 joints × 3 coords in mm.
    PSNR = 10 * log10(MAX² / MSE),  MAX = 1000 mm (joint coords in [-500, 500] mm).

    Args:
        joints_gt_m:   (T, 21, 3) GT joints in metres
        joints_pred_m: (T, 21, 3) predicted joints in metres
        valid_mask:    (T,) bool. If None all frames are used.
        max_val_mm:    signal range upper bound (mm)

    Returns:
        PSNR in dB. Higher is better. Returns +inf if MSE == 0, nan if no valid frames.
    """
    if valid_mask is not None:
        if valid_mask.sum() == 0:
            return float("nan")
        joints_gt_m   = joints_gt_m[valid_mask]
        joints_pred_m = joints_pred_m[valid_mask]

    gt_mm   = joints_gt_m   * 1000.0
    pred_mm = joints_pred_m * 1000.0

    mse = float(np.mean((pred_mm - gt_mm) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(max_val_mm ** 2 / mse))


def compute_ti_rmse(
    joints_gt_m: np.ndarray,
    joints_pred_m: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Temporal Incremental RMSE (TI-RMSE) — motion smoothness metric (no alignment).

    Computes RMSE between GT and predicted frame-to-frame motion increments.
    Only consecutive pairs of valid frames are included.

    Args:
        joints_gt_m:   (T, 21, 3) GT joints in metres
        joints_pred_m: (T, 21, 3) predicted joints in metres
        valid_mask:    (T,) bool. If None all frames are used.

    Returns:
        TI-RMSE in mm. Lower is better. Returns nan if no valid pairs.
    """
    T = len(joints_gt_m)
    gt_mm   = joints_gt_m   * 1000.0
    pred_mm = joints_pred_m * 1000.0

    gt_deltas, pred_deltas = [], []
    for t in range(T - 1):
        if valid_mask is None or (valid_mask[t] and valid_mask[t + 1]):
            gt_deltas.append(gt_mm[t + 1]   - gt_mm[t])
            pred_deltas.append(pred_mm[t + 1] - pred_mm[t])

    if len(gt_deltas) == 0:
        return float("nan")

    gt_deltas   = np.stack(gt_deltas)
    pred_deltas = np.stack(pred_deltas)
    return float(np.sqrt(np.mean((pred_deltas - gt_deltas) ** 2)))


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------

def evaluate_sequence(
    x_pred_norm: np.ndarray,    # [T, 61] normalized predictions
    x_gt_norm: np.ndarray,      # [T, 61] normalized GT
    x_pred_raw: np.ndarray,     # [T, 61] unnormalized predictions
    x_gt_raw: np.ndarray,       # [T, 61] unnormalized GT
    joints_pred: np.ndarray,    # [T, 21, 3] meters
    joints_gt: np.ndarray,      # [T, 21, 3] meters
    valid_mask: np.ndarray,     # [T] bool
) -> Dict[str, float]:
    """Full evaluation of one sequence."""
    metrics = {}

    # Joints metrics
    jm = compute_mpjpe_metrics(joints_gt, joints_pred, valid_mask)
    metrics.update({k: v for k, v in jm.items() if not k.endswith("_per_frame")})

    # MANO-PSNR and TI-RMSE
    metrics["mano_psnr_db"] = compute_mano_psnr(joints_gt, joints_pred, valid_mask)
    metrics["ti_rmse_mm"]   = compute_ti_rmse(joints_gt, joints_pred, valid_mask)

    return metrics


def aggregate_metrics(list_of_dicts: list) -> Dict[str, float]:
    """Average scalar metrics across a list of per-sequence metric dicts."""
    if not list_of_dicts:
        return {}
    keys = [k for k, v in list_of_dicts[0].items() if np.isscalar(v)]
    result = {}
    for k in keys:
        vals = [d[k] for d in list_of_dicts if k in d]
        result[k] = float(np.mean(vals))
    return result


# ---------------------------------------------------------------------------
# Accel / RTE / AUC
# ---------------------------------------------------------------------------

def compute_accel_error(
    joints_gt_m: np.ndarray,
    joints_pred_m: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    fps: float = 30.0,
) -> Tuple[np.ndarray, float]:
    """
    Acceleration error (Accel Error).

    Definition: ||(a_pred - a_gt)||_2, averaged over all joints, in m/s².
    Acceleration = second-order difference * fps².
    Only frames where all three (t-1, t, t+1) are valid are considered.

    Returns:
        per_frame: (T-2,) per-frame error (valid triplets only), m/s²
        mean:      scalar mean
    """
    T = len(joints_gt_m)
    vm = valid_mask if valid_mask is not None else np.ones(T, dtype=bool)

    errors = []
    for t in range(1, T - 1):
        if vm[t - 1] and vm[t] and vm[t + 1]:
            accel_gt   = joints_gt_m[t - 1]   - 2 * joints_gt_m[t]   + joints_gt_m[t + 1]
            accel_pred = joints_pred_m[t - 1] - 2 * joints_pred_m[t] + joints_pred_m[t + 1]
            diff = np.linalg.norm(accel_pred - accel_gt, axis=-1).mean()
            errors.append(diff * fps ** 2)

    if len(errors) == 0:
        return np.array([]), float("nan")
    arr = np.array(errors)
    return arr, float(arr.mean())


def compute_rte(
    joints_gt_m: np.ndarray,
    joints_pred_m: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """
    Root Translation Error (RTE), following the HaWoR definition.

    First aligns the predicted root translation with a fixed-scale global
    rotation + translation (Umeyama, fixed_scale=True), then computes the
    per-frame root position error divided by the total GT trajectory length,
    reported as a percentage (x100).

    joints_gt_m / joints_pred_m: (T, 21, 3) metres, root = joint[0]

    Returns:
        per_frame: (V,) RTE per valid frame (%)
        mean:      scalar mean
    """
    T = len(joints_gt_m)
    vm = valid_mask if valid_mask is not None else np.ones(T, dtype=bool)

    gt_root   = joints_gt_m[vm, 0, :]    # (V, 3)
    pred_root = joints_pred_m[vm, 0, :]  # (V, 3)

    if len(gt_root) < 2:
        return np.array([]), float("nan")

    gt_t   = torch.from_numpy(gt_root).float()
    pred_t = torch.from_numpy(pred_root).float()

    # Umeyama fixed-scale alignment
    gt_mean   = gt_t.mean(0)
    pred_mean = pred_t.mean(0)
    gt_c   = gt_t   - gt_mean
    pred_c = pred_t - pred_mean

    H = pred_c.T @ gt_c
    U, _, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.T @ U.T)
    D = torch.diag(torch.tensor([1.0, 1.0, d.sign().item()]))
    R = Vt.T @ D @ U.T
    t = gt_mean - (R @ pred_mean.unsqueeze(-1)).squeeze(-1)
    pred_aligned = (R @ pred_t.T).T + t

    # Total length of the GT trajectory
    disp = float((gt_t[1:] - gt_t[:-1]).norm(dim=-1).sum())
    if disp < 1e-8:
        return np.array([]), float("nan")

    per_frame = (gt_t - pred_aligned).norm(dim=-1).numpy() / disp * 100.0
    return per_frame, float(per_frame.mean())


def compute_pck_auc(
    joints_gt_m: np.ndarray,
    joints_pred_m: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    max_thresh_mm: float = 50.0,
    n_steps: int = 50,
) -> float:
    """
    PCK-AUC (based on PA-MPJPE): area under the PCK curve over [0, max_thresh_mm],
    normalized to [0, 1].

    PCK@t = fraction of frames whose per-frame PA-MPJPE (Procrustes-aligned)
    mean is < t.
    AUC = trapz(PCK) / max_thresh_mm, range [0, 1], higher is better.
    """
    T = len(joints_gt_m)
    vm = valid_mask if valid_mask is not None else np.ones(T, dtype=bool)

    gt_v   = joints_gt_m[vm]
    pred_v = joints_pred_m[vm]

    if len(gt_v) == 0:
        return float("nan")

    # per-frame Procrustes alignment (rotation + scale + translation)
    pred_pa  = batched_procrustes_align(
        torch.from_numpy(pred_v), torch.from_numpy(gt_v)
    ).numpy()

    # per-frame mean joint error (mm)
    per_frame_err = np.linalg.norm((pred_pa - gt_v) * 1000.0, axis=-1).mean(axis=-1)  # (V,)

    thresholds = np.linspace(0, max_thresh_mm, n_steps + 1)
    pck = np.array([(per_frame_err < thr).mean() for thr in thresholds])
    auc = float(np.trapz(pck, thresholds) / max_thresh_mm)
    return auc


def compute_ra_pck_auc(
    joints_gt_m: np.ndarray,
    joints_pred_m: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    max_thresh_mm: float = 50.0,
    n_steps: int = 50,
) -> float:
    """
    PCK-AUC (based on RA-MPJPE): area under the PCK curve over [0, max_thresh_mm],
    normalized to [0, 1].

    PCK@t = fraction of frames whose per-frame RA-MPJPE (root-relative)
    mean is < t.
    AUC = trapz(PCK) / max_thresh_mm, range [0, 1], higher is better.
    """
    T = len(joints_gt_m)
    vm = valid_mask if valid_mask is not None else np.ones(T, dtype=bool)

    gt_v   = joints_gt_m[vm]
    pred_v = joints_pred_m[vm]

    if len(gt_v) == 0:
        return float("nan")

    # per-frame root-relative (wrist joint = index 0)
    gt_rel   = gt_v   - gt_v[:, 0:1, :]
    pred_rel = pred_v - pred_v[:, 0:1, :]

    # per-frame mean joint error (mm)
    per_frame_err = np.linalg.norm((pred_rel - gt_rel) * 1000.0, axis=-1).mean(axis=-1)  # (V,)

    thresholds = np.linspace(0, max_thresh_mm, n_steps + 1)
    pck = np.array([(per_frame_err < thr).mean() for thr in thresholds])
    auc = float(np.trapz(pck, thresholds) / max_thresh_mm)
    return auc
