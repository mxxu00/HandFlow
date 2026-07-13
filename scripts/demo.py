#!/usr/bin/env python3
"""demo.py — HandFlow unified demo: cam overlay + orthographic view in one run.

Two camera modes (selected by --fix_camera):
  --fix_camera  Fixed camera (e.g. DexYCB on a tripod). ViPE is skipped, c2w is
                identity, and the orthographic view assumes a level, fixed camera:
                a side orthographic view of the camera-space hand motion (the
                camera marker stays at the origin; only the hand moves).
  (default)     Moving camera (e.g. HOT3D head-mounted). ViPE SLAM estimates c2w
                and the orthographic view is a world-space trajectory view: camera
                trajectory line + per-frame hand mesh.

Both modes write two mp4s into --output_dir:
  overlay.mp4   Clean camera-space MANO mesh overlaid onto the RGB frames.
  ortho.mp4     Orthographic (side / topdown) view of the hand (+ camera motion).

Intrinsics (--intrinsics):
  fx,fy,cx,cy   Explicit pinhole intrinsics (recommended; e.g. the demo GT values).
  auto          Let ViPE estimate intrinsics (only valid without --fix_camera).
  (unset)       Generic default [600, 600, W/2, H/2].

Usage:
  # Fixed camera (no ViPE needed)
  python scripts/demo.py --input video.mp4 --fm_ckpt <ckpt> \
      --intrinsics fx,fy,cx,cy --fix_camera --output_dir output/demo

  # Moving camera (ViPE estimates c2w; requires the vipe env)
  python scripts/demo.py --input video.mp4 --fm_ckpt <ckpt> \
      --intrinsics fx,fy,cx,cy --output_dir output/demo

  # Moving camera, unknown intrinsics -> ViPE estimates both intr + c2w
  python scripts/demo.py --input video.mp4 --fm_ckpt <ckpt> \
      --intrinsics auto --output_dir output/demo

"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.checkpoint_utils import load_denoiser_from_ckpt
from utils.inference_utils import run_fm_inference_with_overlap, split_sequence_into_windows
from utils.mano_utils import MANOForwardKinematics
from utils.online_hamer import OnlineHaMeRPipeline
from utils.vipe_worker import run_vipe_slam
from visualization.renderer_p3d import PhongRenderer
from visualization.video_io import read_video_frames, write_video_ffmpeg


def resolve_intrinsics(arg_str, W: int, H: int):
    """Resolve --intrinsics into (intr(4,) | None, vipe_estimate: bool).

    None / "" / "default" -> generic [600, 600, W/2, H/2], vipe_estimate=False
    "auto"                -> (None, True)  (ViPE estimates; only without --fix_camera)
    "fx,fy,cx,cy"         -> explicit (4,), vipe_estimate=False
    """
    if arg_str in (None, "", "default"):
        return np.array([600.0, 600.0, W / 2.0, H / 2.0], dtype=np.float32), False
    if arg_str == "auto":
        return None, True
    parts = [float(x) for x in arg_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--intrinsics expects 4 values fx,fy,cx,cy or 'auto', got {arg_str!r}")
    return np.array(parts, dtype=np.float32), False


def extract_frames_to_dir(frames_bgr, img_dir: str) -> None:
    """Write frames as jpg into a directory (input for ViPE SLAM)."""
    os.makedirs(img_dir, exist_ok=True)
    for i, f in enumerate(frames_bgr):
        cv2.imwrite(str(Path(img_dir) / f"{i:06d}.jpg"), f)


def transform_verts_to_world(verts_cam_m: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """verts_cam (T,N,3) meters + c2w (T,4,4) -> verts_world (T,N,3) meters."""
    T, N, _ = verts_cam_m.shape
    homog = np.concatenate([verts_cam_m, np.ones((T, N, 1), dtype=np.float32)], axis=-1)
    return np.einsum("tij,tnj->tni", c2w.astype(np.float32), homog)[..., :3].astype(np.float32)


def get_camera_geometry(frames_bgr, fix_camera: bool, intr, vipe_estimate: bool, T: int):
    """Return (intr(4,), c2w(T,4,4), vipe_slam_time_s).

    fix_camera=True: c2w = identity (no ViPE); intr is the provided/generic value.
    fix_camera=False: run ViPE — gt_intr (c2w only) when intr is known, else default
                      (ViPE estimates intr + c2w).
    """
    if fix_camera:
        if vipe_estimate:
            print("[demo] --fix_camera ignores --intrinsics auto; using generic 600,600,W/2,H/2")
            intr = np.array([600.0, 600.0, frames_bgr[0].shape[1] / 2.0,
                             frames_bgr[0].shape[0] / 2.0], dtype=np.float32)
        c2w = np.broadcast_to(np.eye(4, dtype=np.float32), (T, 4, 4)).copy()
        return intr.astype(np.float32), c2w, 0.0

    with tempfile.TemporaryDirectory() as tmp:
        img_dir = str(Path(tmp) / "images")
        extract_frames_to_dir(frames_bgr, img_dir)
        if vipe_estimate:
            print("[demo] ViPE SLAM (default, estimating intrinsics + c2w) ...")
            slam = run_vipe_slam(img_dir, tmp, variant="default")
            intr_arr = slam["intrinsics"]
            intr = intr_arr[0] if intr_arr.ndim == 2 else intr_arr
        else:
            print("[demo] ViPE SLAM (gt_intr, estimating c2w only) ...")
            slam = run_vipe_slam(img_dir, tmp, variant="gt_intr", gt_intrinsics_4d=intr)
    c2w = slam["poses"]
    print(f"  c2w shape={c2w.shape}, intrinsics={intr}, slam_time={slam['slam_time_s']:.2f}s")
    return intr.astype(np.float32), c2w.astype(np.float32), float(slam["slam_time_s"])


def main():
    ap = argparse.ArgumentParser(description="HandFlow demo: cam overlay + orthographic view")
    ap.add_argument("--input", required=True, help="Input video path")
    ap.add_argument("--fm_ckpt", required=True, help="HandFlow checkpoint (.pt or DeepSpeed directory)")
    ap.add_argument("--config", default="configs/inference.yaml")
    ap.add_argument("--intrinsics", default=None,
                    help="fx,fy,cx,cy | auto | (unset -> default 600,600,W/2,H/2)")
    ap.add_argument("--fix_camera", action="store_true",
                    help="Fixed camera: skip ViPE (c2w=identity), ortho view assumes a level fixed camera")
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--view", default=None, choices=["topdown", "side", "third_person"],
                    help="Ortho view; default side for --fix_camera else third_person")
    ap.add_argument("--output_dir", default="output/demo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_npz", default=None, help="Optional: path to save inference results npz")
    args = ap.parse_args()

    device = torch.device(args.device)
    t0 = time.perf_counter()

    # ── config ──
    cfg = OmegaConf.load(str(_PROJECT_ROOT / args.config))
    model_cfg = OmegaConf.load(str(_PROJECT_ROOT / cfg.model_yaml))

    # ── read video ──
    print(f"[demo] Reading video {args.input} ...")
    frames_bgr, fps, (H, W) = read_video_frames(args.input)
    T = len(frames_bgr)
    print(f"  {T} frames, {W}x{H}, {fps:.1f} fps")
    t1 = time.perf_counter()

    # ── intrinsics + camera geometry (c2w) ──
    intr, vipe_estimate = resolve_intrinsics(args.intrinsics, W, H)
    intr, c2w, vipe_t = get_camera_geometry(frames_bgr, args.fix_camera, intr, vipe_estimate, T)
    intr_list = [intr] * T
    t2 = time.perf_counter()

    # ── load models ──
    print("[demo] Loading denoiser / HaMeR / MANO ...")
    mano_root = cfg.eval.mano_root or os.environ.get("MANO_ROOT")
    if not mano_root:
        raise RuntimeError("MANO_ROOT not set: please export MANO_ROOT=/path/to/mano")
    denoiser = load_denoiser_from_ckpt(OmegaConf.merge(model_cfg, cfg), args.fm_ckpt, device)
    online = OnlineHaMeRPipeline(device=str(device))
    mano_fk = MANOForwardKinematics(str(mano_root), device)
    t3 = time.perf_counter()

    # ── Online HaMeR ──
    print("[demo] Online HaMeR inference ...")
    online_result = online.process_sequence(frames_bgr, intr_list, target_side=args.side)
    det_valid = online_result["detection_valid"].numpy()
    det_sides = online_result["sides"]
    bbox_xyxy = online_result["bbox_xyxy"].numpy()
    bbox_conf = online_result["bbox_conf"].numpy()
    dominant_side = max(set(det_sides), key=det_sides.count) if det_sides else args.side
    print(f"  Detections: {int(det_valid.sum())}/{T} frames, dominant hand side={dominant_side}")
    t4 = time.perf_counter()

    bf = online_result["backbone_features"].to(device)
    image_tokens = denoiser.frame_compressor(bf.unsqueeze(0)).squeeze(0)

    seq_batch = {
        "mano_params": torch.zeros((1, T, 48), dtype=torch.float32),
        "mano_trans": torch.zeros((1, T, 3), dtype=torch.float32),
        "mano_betas": torch.zeros((1, 10), dtype=torch.float32),
        "padding_mask": torch.zeros((1, T), dtype=torch.bool),
        "images": online_result["crop_images"].unsqueeze(0),
        "hamer_landmarks": online_result["hamer_landmarks"].unsqueeze(0),
        "crop_intrinsics": online_result["crop_intrinsics"].unsqueeze(0),
        "hamer_confidence": online_result["hamer_confidence"].unsqueeze(0),
        "image_tokens": image_tokens.unsqueeze(0),
        "side": [dominant_side],
        "source": ["custom"],
    }

    # ── FM inference (overlapping windows) ──
    win = int(cfg.inference.window_size)
    overlap = int(cfg.inference.overlap_size)
    ode_steps = int(cfg.inference.ode_steps)
    stride = win - overlap
    win_batch, _ = split_sequence_into_windows(seq_batch, win, stride, device)
    win_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in win_batch.items()}

    print("[demo] FM Denoiser inference ...")
    pose_seq, trans_seq, betas_pred = run_fm_inference_with_overlap(
        denoiser, denoiser, win_batch, win, overlap, ode_steps, device,
        overlap_method=cfg.inference.get("overlap_method", "vblend"),
    )
    nf = pose_seq.shape[0]
    print(f"  Output {nf} frames of pose/trans")
    t5 = time.perf_counter()

    # ── MANO FK -> verts_cam (meters, per-frame hand side from detection) ──
    sides = (det_sides[:nf] if len(det_sides) >= nf
             else det_sides + [det_sides[-1]] * (nf - len(det_sides)))
    verts_cam_m = (mano_fk.verts(pose_seq, betas_pred, trans_seq, sides) / 1000.0
                   ).cpu().numpy().astype(np.float32)   # (nf,778,3) m
    faces = mano_fk.get_faces(args.side)
    t6 = time.perf_counter()

    # ── world verts + camera positions for the ortho view ──
    n = min(nf, T, c2w.shape[0])
    verts_world = transform_verts_to_world(verts_cam_m[:n], c2w[:n])      # identity if fix_camera
    cam_positions = c2w[:n, :3, 3]                                         # zeros if fix_camera

    # ── Output 1: cam overlay (clean verts_cam projection onto RGB) ──
    print("[demo] Rendering overlay ...")
    renderer = PhongRenderer(device)
    n_render = min(nf, T)
    out_frames = []
    for fi in range(T):
        frame = frames_bgr[fi]
        if fi < n_render and det_valid[fi]:
            x1, y1, x2, y2 = bbox_xyxy[fi].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det_sides[fi]} {bbox_conf[fi]:.2f}"
            cv2.putText(frame, label, (x1, max(8, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            frame = renderer.render_overlay(frame, verts_cam_m[fi], faces, intr, sides[fi])
        out_frames.append(frame)
    os.makedirs(args.output_dir, exist_ok=True)
    overlay_path = str(Path(args.output_dir) / "overlay.mp4")
    write_video_ffmpeg(out_frames, overlay_path, fps)
    print(f"[demo] ✅ {overlay_path}")
    t7 = time.perf_counter()

    # ── Output 2: orthographic view (side for fix_camera / level; else world topdown) ──
    view = args.view or ("side" if args.fix_camera else "third_person")
    print(f"[demo] Rendering ortho ({view}) ...")
    traj_frames = renderer.render_ortho_video(verts_world, faces, c2w[:n], view=view, side=args.side)
    ortho_path = str(Path(args.output_dir) / f"ortho_{view}.mp4")
    write_video_ffmpeg(traj_frames, ortho_path, fps)
    print(f"[demo] ✅ {ortho_path}")
    t8 = time.perf_counter()

    # ── timing breakdown ──
    det_t = float(online_result.get("det_time_s", 0.0))
    hamer_t = float(online_result.get("hamer_time_s", 0.0))
    print(
        f"[timing] read={t1-t0:.1f}s | "
        f"vipe={vipe_t:.1f}s(+frames/subprocess {t2-t1-vipe_t:.1f}s) | "
        f"load={t3-t2:.1f}s | "
        f"hamer={t4-t3:.1f}s (det={det_t:.1f} infer={hamer_t:.1f}) | "
        f"fm={t5-t4:.1f}s | mano={t6-t5:.1f}s | render={t7-t6:.1f}s | ortho={t8-t7:.1f}s | "
        f"total={t8-t0:.1f}s"
    )

    # ── optional: save inference npz ──
    if args.save_npz:
        np.savez(
            args.save_npz,
            verts_cam=verts_cam_m[:n_render], verts_world=verts_world, faces=faces,
            intrinsics=intr, c2w=c2w[:n], fix_camera=bool(args.fix_camera),
            pose=pose_seq.cpu().numpy(), trans=trans_seq.cpu().numpy(),
            betas=betas_pred.cpu().numpy(), pred_valid=det_valid[:n_render],
            side=args.side, source=args.input, fps=fps,
        )
        print(f"[demo] npz -> {args.save_npz}")


if __name__ == "__main__":
    main()
