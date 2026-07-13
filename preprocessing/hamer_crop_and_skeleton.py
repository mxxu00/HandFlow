"""
HaMeR full preprocessing (LMDB mode) — unified crop LMDB + packed npz

Reads images from the DexYCB + HOT3D RGB LMDBs, runs HaMeR inference, and saves:
  - 256x256 crop images -> unified crop LMDB (all_hamer_crop_lmdb/)
  - landmarks (21,2) normalized [0,1] + confidence -> all_hamer_packed.npz
  - crop_intrinsics (4,) -> all_crop_params.npz

Both npz formats match the legacy version, so dataset.py can switch seamlessly.

Usage (GPU node; training data preprocessing, not required for V1 demo):
  export HANDFLOW_DATA_ROOT=/path/to/dataset_v2
  python preprocessing/hamer_crop_and_skeleton.py \
      --mano_npz $HANDFLOW_DATA_ROOT/all_mano_aa_packed.npz \
      --lmdb_hot3d $HANDFLOW_DATA_ROOT/hot3d_rgb_lmdb \
      --lmdb_dexycb $HANDFLOW_DATA_ROOT/dexycb_rgb_lmdb \
      --intrinsics_hot3d $HANDFLOW_DATA_ROOT/hot3d_intrinsics.npz \
      --intrinsics_dexycb $HANDFLOW_DATA_ROOT/dexycb_intrinsics.npz \
      --output_dir $HANDFLOW_DATA_ROOT \
      --crop_lmdb $HANDFLOW_DATA_ROOT/all_hamer_crop_lmdb \
      --batch_size 32

Note: the V1 demo only uses the crop utility functions in this file (imported by utils/online_hamer.py);
    main() / run_hamer_batch_full() / setup_hamer() are for training data preprocessing and are optional.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch
from scipy.spatial.transform import Rotation as _R
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.dual_conf import select_target_box_dual_conf

# Paths are specified by environment variables (see README); HaMeR defaults to third_party/hamer
HAMER_ROOT    = Path(os.environ.get("HAMER_ROOT", str(_PROJECT_ROOT / "third_party" / "hamer")))
HAMER_DATA    = Path(os.environ.get("HAMER_DATA", str(HAMER_ROOT / "_DATA")))
HAMER_CKPT    = Path(os.environ.get("HAMER_CKPT", str(HAMER_DATA / "hamer_ckpts" / "checkpoints" / "hamer.ckpt")))
PYRENDER_STUB = HAMER_ROOT / "pyrender_stub"
DETECTOR_PT   = Path(os.environ.get("DETECTOR_CKPT", str(HAMER_ROOT / "detector.pt")))
WILOR_ROOT    = DETECTOR_PT.parent


# ── Crop utility functions ─────────────────────────────────────────────────────

def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    if target_aspect_ratio is None:
        return input_shape
    w, h = input_shape
    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    return np.array([w_new, h_new])


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height,
                            dst_width, dst_height, scale=1.0, rot=0.0):
    src_w = src_width * scale
    src_h = src_height * scale
    rot_rad = np.pi * rot / 180.0
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_center = np.array([c_x, c_y], dtype=np.float32)
    src_downdir = np.array([0, src_h * 0.5], dtype=np.float32)
    src_downdir = np.array([src_downdir[0]*cs - src_downdir[1]*sn,
                            src_downdir[0]*sn + src_downdir[1]*cs], dtype=np.float32)
    src_rightdir = np.array([src_w * 0.5, 0], dtype=np.float32)
    src_rightdir = np.array([src_rightdir[0]*cs - src_rightdir[1]*sn,
                             src_rightdir[0]*sn + src_rightdir[1]*cs], dtype=np.float32)

    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_height * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_width * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans = cv2.getAffineTransform(src, dst)
    return trans


def compute_crop_intrinsics(trans, K_orig_4):
    """K_crop = S @ K_orig"""
    fx, fy, cx, cy = K_orig_4
    K_orig = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0,  0,  1]], dtype=np.float64)
    S = np.eye(3, dtype=np.float64)
    S[:2, :] = trans.astype(np.float64)
    K_crop = S @ K_orig
    return np.array([K_crop[0, 0], K_crop[1, 1],
                     K_crop[0, 2], K_crop[1, 2]], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Model initialization
# ══════════════════════════════════════════════════════════════════════════════

def setup_hamer(device: str):
    sys.path.insert(0, str(HAMER_ROOT))
    os.environ["HAMER_DATA"] = str(HAMER_DATA)

    import torch as _t
    _orig = _t.load
    def _p(*a, **k):
        k.setdefault("weights_only", False)
        return _orig(*a, **k)
    _t.load = _p
    from ultralytics import YOLO
    detector = YOLO(str(DETECTOR_PT))
    _t.load = _orig

    import hamer.configs as _hcfg
    _hcfg.CACHE_DIR_HAMER = str(HAMER_DATA)

    from pathlib import Path as _P
    from hamer.configs import get_config
    from hamer.models.hamer import HAMER
    model_cfg_path = str(_P(HAMER_CKPT).parent.parent / "model_config.yaml")
    model_cfg = get_config(model_cfg_path, update_cachedir=True)
    if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in model_cfg.MODEL):
        model_cfg.defrost()
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        model_cfg.freeze()
    model = HAMER.load_from_checkpoint(str(HAMER_CKPT), strict=False, cfg=model_cfg,
                                        init_renderer=False)

    model = model.to(device).eval()
    detector.to(device)
    print(f"HaMeR loaded on {device}")
    return detector, model, model_cfg


# ══════════════════════════════════════════════════════════════════════════════
# Coordinate conversion helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length):
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2.0 * focal_length / bs
    tx = (2.0 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2.0 * (cy - h_2) / bs) + cam_bbox[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


def _project_full_img(points, cam_trans, focal_length, img_w, img_h):
    cx, cy = img_w / 2.0, img_h / 2.0
    K = np.array([[focal_length, 0.0, cx],
                  [0.0, focal_length, cy],
                  [0.0,         0.0,  1.0]], dtype=np.float64)
    pts = points.astype(np.float64) + cam_trans
    pts = pts / pts[:, 2:3]
    return (K @ pts.T).T[:, :2]


# ══════════════════════════════════════════════════════════════════════════════
# HaMeR batch inference + crop saving
# ══════════════════════════════════════════════════════════════════════════════

_EMPTY = (None, 0.0, None, None, None, None, None)


def run_hamer_batch_full(
    imgs_bgr: list,
    detector,
    model,
    model_cfg,
    device: str,
    det_conf: float = 0.3,
    batch_size: int = 32,
    crop_txn=None,
    crop_lmdb_seq_key: str | None = None,
    frame_indices: list | None = None,
    intrinsics_4d_list: list | None = None,
) -> list[tuple]:
    """Run HaMeR on a batch of BGR images, returning predictions + saving crops"""
    results_per_frame = [None] * len(imgs_bgr)

    # ── 1. YOLO batch detection ──
    yolo_out = detector(imgs_bgr, conf=det_conf, verbose=False)

    valid_imgs, valid_bboxes, valid_confs, valid_hw, valid_fi = [], [], [], [], []
    for fi, (img_bgr, yres) in enumerate(zip(imgs_bgr, yolo_out)):
        if len(yres.boxes) == 0:
            results_per_frame[fi] = _EMPTY
            continue

        clses   = yres.boxes.cls.cpu().numpy()
        confs   = yres.boxes.conf.cpu().numpy()
        bboxes  = yres.boxes.xyxy.cpu().numpy()

        # Pick highest-conf right-hand box; but if it heavily overlaps (IoU>thr) a left-hand box
        # with higher conf, the right box is defeated (actually the left hand) -> no detection.
        sel = select_target_box_dual_conf(clses, confs, bboxes, target_cls=1)
        if sel is None:
            results_per_frame[fi] = _EMPTY
            continue

        valid_imgs.append(img_bgr)
        valid_bboxes.append(bboxes[sel:sel+1].astype(np.float32))
        valid_confs.append(float(confs[sel]))
        valid_hw.append(img_bgr.shape[:2])
        valid_fi.append(fi)

    if not valid_fi:
        for i in range(len(results_per_frame)):
            if results_per_frame[i] is None:
                results_per_frame[i] = _EMPTY
        return results_per_frame

    # ── 1.5 Generate raw crops + affine transform + save to LMDB ──
    img_size = model_cfg.MODEL.IMAGE_SIZE
    BBOX_SHAPE = model_cfg.MODEL.get('BBOX_SHAPE', None)
    rescale_factor = 2.0

    crop_bgr_list = []
    crop_intr_list = []
    box_center_list = []
    bbox_size_list = []

    for vi, (img, bbox) in enumerate(zip(valid_imgs, valid_bboxes)):
        box = bbox[0].astype(np.float32)
        center = (box[2:4] + box[0:2]) / 2.0
        scale = rescale_factor * (box[2:4] - box[0:2]) / 200.0
        bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=BBOX_SHAPE).max()
        center_x, center_y = center
        box_center_list.append(center.astype(np.float32))
        bbox_size_list.append(float(bbox_size))

        trans = gen_trans_from_patch_cv(center_x, center_y, bbox_size, bbox_size,
                                        img_size, img_size, 1.0, 0)
        crop_bgr = cv2.warpAffine(img, trans, (img_size, img_size),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)
        crop_bgr_list.append(crop_bgr)

        fi_global = valid_fi[vi]
        if intrinsics_4d_list is not None and fi_global < len(intrinsics_4d_list):
            crop_intr = compute_crop_intrinsics(trans, intrinsics_4d_list[fi_global])
        else:
            crop_intr = np.zeros(4, dtype=np.float32)
        crop_intr_list.append(crop_intr)

        if crop_txn is not None and crop_lmdb_seq_key is not None and frame_indices is not None:
            frame_idx = frame_indices[fi_global]
            _, img_enc = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            key = f"img/{crop_lmdb_seq_key}/{frame_idx:06d}".encode("utf-8")
            crop_txn.put(key, img_enc.tobytes())

    # ── 2. HaMeR ViT batch inference (uses crop_bgr directly, skipping ViTDetDataset) ──
    MEAN = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
    STD  = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)

    # (N, 256, 256, 3) BGR uint8 -> (N, 3, 256, 256) RGB normalized
    crops_np = np.stack(crop_bgr_list)[:, :, :, ::-1].astype(np.float32)
    crops_t = torch.from_numpy(crops_np).permute(0, 3, 1, 2).to(device)
    crops_t = (crops_t - MEAN) / STD

    # metadata for camera unprojection
    box_center_t = torch.tensor(np.stack(box_center_list), dtype=torch.float32, device=device)
    bbox_size_t  = torch.tensor(bbox_size_list, dtype=torch.float32, device=device)
    img_size_t   = torch.tensor(
        [[float(w), float(h)] for h, w in valid_hw],
        dtype=torch.float32, device=device,
    )

    scaled_fl = float(
        model_cfg.EXTRA.FOCAL_LENGTH
        / model_cfg.MODEL.IMAGE_SIZE
        * img_size_t[0].max().item()
    )

    all_kp_norms, all_global_aa, all_hand_aa, all_betas_l, all_cam_t = [], [], [], [], []

    for s in range(0, len(crops_t), batch_size):
        e = min(s + batch_size, len(crops_t))
        with torch.no_grad():
            out = model({'img': crops_t[s:e]})

        pred_cam = out["pred_cam"].clone()
        joints_3d = out["pred_keypoints_3d"].detach().cpu().numpy()

        cam_t_full_np = _cam_crop_to_full(
            pred_cam, box_center_t[s:e], bbox_size_t[s:e],
            img_size_t[s:e], scaled_fl,
        ).detach().cpu().numpy()

        g_np = out["pred_mano_params"]["global_orient"].detach().cpu().numpy()
        h_np = out["pred_mano_params"]["hand_pose"].detach().cpu().numpy()
        b_np = out["pred_mano_params"]["betas"].detach().cpu().numpy()
        B = g_np.shape[0]

        g_aa = _R.from_matrix(g_np.reshape(-1, 3, 3)).as_rotvec().reshape(B, 3).astype(np.float32)
        h_aa = _R.from_matrix(h_np.reshape(-1, 3, 3)).as_rotvec().reshape(B, 45).astype(np.float32)

        for i in range(B):
            pos = s + i
            H, W = valid_hw[pos]
            kp2d = _project_full_img(joints_3d[i], cam_t_full_np[i], scaled_fl, W, H)
            kp_n = np.clip(kp2d / np.array([W, H], dtype=np.float32), 0.0, 1.0).astype(np.float32)
            all_kp_norms.append(kp_n)
            all_global_aa.append(g_aa[i])
            all_hand_aa.append(h_aa[i])
            all_betas_l.append(b_np[i].astype(np.float32))
            all_cam_t.append(cam_t_full_np[i].astype(np.float32))

    for vi, (fi, kp_n, conf, gaa, haa, bet, camt) in enumerate(zip(
            valid_fi, all_kp_norms, valid_confs,
            all_global_aa, all_hand_aa, all_betas_l, all_cam_t)):
        results_per_frame[fi] = (kp_n, conf, gaa, haa, bet, camt, crop_intr_list[vi])

    for i in range(len(results_per_frame)):
        if results_per_frame[i] is None:
            results_per_frame[i] = _EMPTY
    return results_per_frame


# ══════════════════════════════════════════════════════════════════════════════
# Main function
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HaMeR full preprocessing — unified crop LMDB + packed npz"
    )
    parser.add_argument("--mano_npz", type=str, required=True,
        help="Unified MANO npz (to obtain seq_keys and num_frames)")
    parser.add_argument("--lmdb_hot3d", type=str, default=None)
    parser.add_argument("--lmdb_dexycb", type=str, default=None)
    parser.add_argument("--intrinsics_hot3d", type=str, default=None)
    parser.add_argument("--intrinsics_dexycb", type=str, default=None)
    parser.add_argument("--output_dir", type=str,
        default=os.environ.get("HANDFLOW_DATA_ROOT", "data"))
    parser.add_argument("--crop_lmdb", type=str, default=None,
        help="Unified Crop LMDB output path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--det_conf", type=float, default=0.3)
    parser.add_argument("--seq_start", type=int, default=0)
    parser.add_argument("--seq_end", type=int, default=999999)
    parser.add_argument("--prefetch", type=int, default=8,
        help="Number of LMDB prefetch threads (higher keeps the GPU busier but uses more memory)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"

    # ── Read sequence list ──
    print(f"Loading sequence list: {args.mano_npz}")
    ref = np.load(args.mano_npz, allow_pickle=True)
    all_seq_keys   = [str(k) for k in ref["seq_keys"]]
    all_num_frames = ref["num_frames"].tolist()

    seq_keys   = all_seq_keys[args.seq_start : args.seq_end]
    num_frames = all_num_frames[args.seq_start : args.seq_end]
    print(f"Total sequences: {len(all_seq_keys)}, this shard: [{args.seq_start}, {args.seq_end}), {len(seq_keys)} sequences")

    # ── Determine which LMDB each seq belongs to ──
    def _lkey(sk):
        return "hot3d" if sk.startswith("clip-") else "dexycb"

    # ── Open LMDBs ──
    lmdb_envs, txns = {}, {}
    if args.lmdb_hot3d:
        print(f"Opening HOT3D LMDB: {args.lmdb_hot3d}")
        lmdb_envs["hot3d"] = lmdb.open(args.lmdb_hot3d, readonly=True,
                                        lock=False, readahead=False, meminit=False)
        txns["hot3d"] = lmdb_envs["hot3d"].begin(write=False)
    if args.lmdb_dexycb:
        print(f"Opening DexYCB LMDB: {args.lmdb_dexycb}")
        lmdb_envs["dexycb"] = lmdb.open(args.lmdb_dexycb, readonly=True,
                                         lock=False, readahead=False, meminit=False)
        txns["dexycb"] = lmdb_envs["dexycb"].begin(write=False)

    # ── Load intrinsics ──
    intrinsics_per_seq: dict[str, dict[int, np.ndarray]] = {}

    def _load_intr(npz_path, label):
        print(f"Loading {label} intrinsics: {npz_path}")
        d = np.load(npz_path, allow_pickle=True)
        keys = [str(k) for k in d["seq_keys"]]
        offsets = d["offsets"]
        all_intr = d["all_intrinsics"]
        for i, sk in enumerate(keys):
            s, e = int(offsets[i]), int(offsets[i + 1])
            intrinsics_per_seq[sk] = {j - s: all_intr[j] for j in range(s, e)}
        print(f"  Loaded intrinsics for {len(keys)} sequences")

    if args.intrinsics_hot3d:
        _load_intr(args.intrinsics_hot3d, "HOT3D")
    if args.intrinsics_dexycb:
        _load_intr(args.intrinsics_dexycb, "DexYCB")

    # ── Load HaMeR ──
    print("Loading HaMeR model...")
    detector, model, model_cfg = setup_hamer(device)

    # ── Open Crop LMDB ──
    crop_env = None
    if args.crop_lmdb:
        total_est = sum(num_frames)
        map_size = max(total_est * 80 * 1024, 1 << 30)  # >=80KB/crop, at least 1GB
        print(f"Crop LMDB: {args.crop_lmdb} (map_size={map_size / 1024 / 1024:.0f} MB)")
        Path(args.crop_lmdb).parent.mkdir(parents=True, exist_ok=True)
        crop_env = lmdb.open(args.crop_lmdb, map_size=map_size)

    # ── Accumulate results (unified packed format) ───────────────────────────────
    result_seq_keys: list[str] = []
    result_seq_offsets = [0]
    result_num_frames: list[int] = []

    all_inds_list: list[np.ndarray] = []
    all_landmarks_list: list[np.ndarray] = []
    all_confidence_list: list[np.ndarray] = []
    all_crop_intrinsics_list: list[np.ndarray] = []

    # ── Prefetch pipeline ──
    def _read_seq(seq_key, nf, lkey):
        """Background thread: read all frames of a sequence from the corresponding LMDB"""
        t = txns[lkey]
        imgs, fids, intrs = [], [], []
        il = intrinsics_per_seq.get(seq_key)
        for fi in range(nf):
            key = f"img/{seq_key}/{fi:06d}".encode()
            val = t.get(key)
            if val is None:
                continue
            arr = np.frombuffer(val, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                imgs.append(img)
                fids.append(fi)
                intrs.append(il.get(fi, np.zeros(4, np.float32)) if il else np.zeros(4, np.float32))
        return imgs, fids, intrs

    pending = list(zip(seq_keys, num_frames))
    total_det = total_loaded = 0
    prefetch_n = args.prefetch

    with ThreadPoolExecutor(max_workers=prefetch_n) as pool:
        futures: dict[str, Future] = {}

        def _submit_batch(start_idx):
            for idx in range(start_idx, min(start_idx + prefetch_n, len(pending))):
                sk, nf = pending[idx]
                if sk not in futures:
                    futures[sk] = pool.submit(_read_seq, sk, nf, _lkey(sk))

        _submit_batch(0)

        for pi, (seq_key, nf) in enumerate(tqdm(pending, desc="seqs")):
            _submit_batch(pi + prefetch_n)

            try:
                imgs_bgr, frame_idx_list, intr_4d_list = futures.pop(seq_key).result()

                if not imgs_bgr:
                    result_seq_keys.append(seq_key)
                    result_num_frames.append(nf)
                    result_seq_offsets.append(result_seq_offsets[-1])
                    continue

                crop_txn = crop_env.begin(write=True) if crop_env else None

                # Chunked inference
                all_results: list[tuple] = []
                for i in range(0, len(imgs_bgr), args.batch_size):
                    chunk = imgs_bgr[i:i + args.batch_size]
                    chunk_intr = intr_4d_list[i:i + args.batch_size] if intr_4d_list else None
                    all_results.extend(
                        run_hamer_batch_full(
                            chunk, detector, model, model_cfg, device,
                            args.det_conf, args.batch_size,
                            crop_txn=crop_txn,
                            crop_lmdb_seq_key=seq_key,
                            frame_indices=frame_idx_list[i:i + args.batch_size],
                            intrinsics_4d_list=chunk_intr,
                        )
                    )

                if crop_txn:
                    crop_txn.commit()

                # Aggregate this sequence
                inds_out, lm_out, conf_out, cintr_out = [], [], [], []
                for fi, (kp_n, conf, gaa, haa, bet, camt, cintr) in zip(frame_idx_list, all_results):
                    if kp_n is not None:
                        inds_out.append(fi)
                        lm_out.append(kp_n)
                        conf_out.append(conf)
                        cintr_out.append(cintr)

                n_det = len(inds_out)
                total_det += n_det
                total_loaded += len(frame_idx_list)

                # Append to unified results
                result_seq_keys.append(seq_key)
                result_num_frames.append(nf)
                result_seq_offsets.append(result_seq_offsets[-1] + n_det)

                if inds_out:
                    all_inds_list.append(np.array(inds_out, np.int32))
                    all_landmarks_list.append(np.stack(lm_out).astype(np.float32))
                    all_confidence_list.append(np.array(conf_out, np.float32))
                    all_crop_intrinsics_list.append(np.stack(cintr_out).astype(np.float32))

                if n_det == 0:
                    tqdm.write(f"  [WARN] no detection: {seq_key}")

            except Exception as e:
                tqdm.write(f"  [ERROR] {seq_key}: {e}")
                import traceback; traceback.print_exc()
                result_seq_keys.append(seq_key)
                result_num_frames.append(nf)
                result_seq_offsets.append(result_seq_offsets[-1])

    # Close LMDBs
    for env in lmdb_envs.values():
        env.close()
    if crop_env is not None:
        crop_env.close()

    # ── Write unified packed npz ──
    print("\nWriting unified packed npz...")

    all_inds = np.concatenate(all_inds_list) if all_inds_list else np.zeros(0, np.int32)
    all_landmarks = np.concatenate(all_landmarks_list) if all_landmarks_list else np.zeros((0, 21, 2), np.float32)
    all_confidence = np.concatenate(all_confidence_list) if all_confidence_list else np.zeros(0, np.float32)
    all_crop_intrinsics = np.concatenate(all_crop_intrinsics_list) if all_crop_intrinsics_list else np.zeros((0, 4), np.float32)

    hamer_path = out_dir / "all_hamer_packed.npz"
    np.savez_compressed(
        str(hamer_path),
        seq_keys=np.array(result_seq_keys, dtype=object),
        seq_offsets=np.array(result_seq_offsets, dtype=np.int64),
        all_inds=all_inds,
        all_landmarks=all_landmarks,
        all_confidence=all_confidence,
    )

    crop_params_path = out_dir / "all_crop_params.npz"
    np.savez_compressed(
        str(crop_params_path),
        seq_keys=np.array(result_seq_keys, dtype=object),
        seq_offsets=np.array(result_seq_offsets, dtype=np.int64),
        all_inds=all_inds,
        all_crop_intrinsics=all_crop_intrinsics,
    )

    hamer_mb = hamer_path.stat().st_size / 1024 / 1024
    crop_mb = crop_params_path.stat().st_size / 1024 / 1024
    print(f"\n[saved] {hamer_path} ({hamer_mb:.1f} MB)")
    print(f"[saved] {crop_params_path} ({crop_mb:.1f} MB)")

    rate = 100.0 * total_det / max(total_loaded, 1)
    print(f"Detection rate: {total_det}/{total_loaded} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
