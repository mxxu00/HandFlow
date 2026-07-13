"""
Online HaMeR pipeline — YOLO detection + full HaMeR inference.

Starting from raw images, a single forward pass simultaneously outputs:
  - backbone features (ViT-H 192 tokens × 1280D)
  - 2D landmarks (crop coordinate system [0,1])
  - crop intrinsics
  - detection confidence

Reuses the crop utility functions and coordinate conversion logic from preprocessing/hamer_crop_and_skeleton.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# Project root directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.dual_conf import select_target_box_dual_conf

# HaMeR is a submodule located at third_party/hamer; checkpoint path is set via environment variable (see README)
HAMER_ROOT = _PROJECT_ROOT / "third_party" / "hamer"
HAMER_CKPT = Path(os.environ.get(
    "HAMER_CKPT",
    str(HAMER_ROOT / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"),
))
# _DATA: prefer environment variable; otherwise inferred from HAMER_CKPT (ckpt is at <_DATA>/hamer_ckpts/checkpoints/)
HAMER_DATA = Path(os.environ.get(
    "HAMER_DATA",
    str(HAMER_CKPT.parents[2]) if os.environ.get("HAMER_CKPT") else str(HAMER_ROOT / "_DATA"),
))
DETECTOR_PT = Path(os.environ["DETECTOR_CKPT"]) if "DETECTOR_CKPT" in os.environ else None

# WiLoR hand detector class mapping: cls 0=left, 1=right
_SIDE_TO_CLS = {"left": 0, "right": 1}


# ── crop utility functions (imported from preprocessing) ─────────────────────────────────

def _import_crop_utils():
    """Lazy import to avoid import errors in non-GPU environments."""
    sys.path.insert(0, str(_PROJECT_ROOT))
    from preprocessing.hamer_crop_and_skeleton import (
        expand_to_aspect_ratio,
        gen_trans_from_patch_cv,
        compute_crop_intrinsics,
        _cam_crop_to_full,
        _project_full_img,
    )
    return expand_to_aspect_ratio, gen_trans_from_patch_cv, compute_crop_intrinsics, _cam_crop_to_full, _project_full_img


class OnlineHaMeRPipeline:
    """
    Online HaMeR pipeline: raw image -> YOLO detection -> crop -> full HaMeR inference.

    Captures backbone features via a forward hook; a single forward pass jointly produces
    backbone features + 2D landmarks + crop intrinsics + confidence.
    """

    # ImageNet normalization (x255 variant, consistent with the HAMER model)
    _IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53])
    _IMAGENET_STD = torch.tensor([58.395, 57.12, 57.375])

    def __init__(
        self,
        hamer_ckpt: str | None = None,
        detector_ckpt: str | None = None,
        device: str = "cuda",
        det_conf: float = 0.3,
        hamer_batch_size: int = 32,
        image_size: int = 256,
    ):
        self.device = device
        self.det_conf = det_conf
        self.hamer_batch_size = hamer_batch_size
        self.image_size = image_size

        # Lazy-import utility functions
        (
            self._expand_to_aspect_ratio,
            self._gen_trans_from_patch_cv,
            self._compute_crop_intrinsics,
            self._cam_crop_to_full,
            self._project_full_img,
        ) = _import_crop_utils()

        # Load YOLO + HAMER
        _hamer = hamer_ckpt or str(HAMER_CKPT)
        _detector = detector_ckpt or (str(DETECTOR_PT) if DETECTOR_PT else None)
        if _detector is None:
            raise RuntimeError(
                "YOLO detector weights not specified: please export DETECTOR_CKPT=/path/to/detector.pt, "
                "or pass the detector_ckpt argument (weights come from the WiLoR repo)"
            )
        self.detector, self.model, self.model_cfg = self._setup_models(
            _hamer, _detector, device,
        )

        # backbone hook storage
        self._backbone_features: Optional[torch.Tensor] = None
        self._hook_handle = self.model.backbone.register_forward_hook(
            self._backbone_hook
        )

        # normalization buffer
        self.register_buffer = lambda name, val: setattr(self, name, val.to(device))
        self._mean = self._IMAGENET_MEAN.view(1, 3, 1, 1).to(device)
        self._std = self._IMAGENET_STD.view(1, 3, 1, 1).to(device)

    def _backbone_hook(self, module, input, output):
        """Forward hook: captures backbone output (B, C, Hp, Wp) -> (B, 192, 1280)."""
        feat_map = output  # (B, 1280, Hp, Wp)
        B, C, Hp, Wp = feat_map.shape
        self._backbone_features = feat_map.flatten(2).transpose(1, 2)  # (B, Hp*Wp, 1280)

    # ── model initialization ────────────────────────────────────────────────

    @staticmethod
    def _setup_models(hamer_ckpt: str, detector_ckpt: str, device: str):
        """Load the YOLO detector + the full HAMER model."""
        import gc

        sys.path.insert(0, str(HAMER_ROOT))
        os.environ["HAMER_DATA"] = str(HAMER_DATA)

        # Temporarily patch torch.load for compatibility with legacy ckpts
        _orig_load = torch.load
        def _patched_load(*a, **k):
            k.setdefault("weights_only", False)
            return _orig_load(*a, **k)
        torch.load = _patched_load

        from ultralytics import YOLO
        detector = YOLO(str(detector_ckpt))
        detector.to(device)

        torch.load = _orig_load  # restore

        import hamer.configs as _hcfg
        _hcfg.CACHE_DIR_HAMER = str(HAMER_DATA)

        from hamer.configs import get_config
        from hamer.models.hamer import HAMER

        model_cfg_path = str(Path(hamer_ckpt).parent.parent / "model_config.yaml")
        model_cfg = get_config(model_cfg_path, update_cachedir=True)

        if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in model_cfg.MODEL):
            model_cfg.defrost()
            model_cfg.MODEL.BBOX_SHAPE = [192, 256]
            model_cfg.freeze()
        if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
            model_cfg.defrost()
            model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
            model_cfg.freeze()

        # Manually load to CPU to avoid PL load_from_checkpoint keeping the entire ckpt on GPU
        model = HAMER(cfg=model_cfg, init_renderer=False)
        ckpt_dict = torch.load(str(hamer_ckpt), map_location="cpu", weights_only=False)
        state_dict = ckpt_dict.get("state_dict", ckpt_dict)
        # Clean up the ckpt dict, keeping only the state_dict
        del ckpt_dict
        gc.collect()
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
        model = model.to(device).eval()
        torch.cuda.empty_cache()

        # Print actual GPU memory usage
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            print(f"[OnlineHaMeR] YOLO + HAMER loaded on {device}, "
                  f"GPU memory: {(total-free)/1024**2:.0f} / {total/1024**2:.0f} MB")

        return detector, model, model_cfg

    # ── main interface ─────────────────────────────────────────────────────

    @torch.no_grad()
    def process_sequence(
        self,
        raw_images_bgr: list[np.ndarray],
        raw_intrinsics_4d: list[np.ndarray],
        target_side: str = "right",
    ) -> dict:
        """
        Process the raw images of a full sequence.

        Args:
            raw_images_bgr: list of (H, W, 3) BGR uint8 images
            raw_intrinsics_4d: list of (4,) [fx, fy, cx, cy] raw intrinsics
            target_side: "right"/"left", lock onto a single hand -- only select the YOLO box of this hand side;
                detections of the other hand are always ignored (prevents frames with both hands co-occurring from sliding onto the wrong hand). WiLoR: 0=left, 1=right.

        Returns:
            dict with:
                crop_images:       (T, 3, 256, 256) uint8
                backbone_features: (T, 192, 1280) float32 — ViT backbone output
                hamer_landmarks:   (T, 21, 2) float32 — crop coordinate system [0,1]
                crop_intrinsics:   (T, 4) float32
                hamer_confidence:  (T,) float32
                detection_valid:   (T,) bool
                det_time_s:        float — YOLO detection time
                hamer_time_s:      float — HaMeR inference time
        """
        import time

        T = len(raw_images_bgr)
        dev = self.device
        img_size = self.image_size

        # ── 1. YOLO detection ────────────────────────────────────────────
        t0 = time.perf_counter()
        yolo_out = self.detector(raw_images_bgr, conf=self.det_conf, verbose=False)

        # Parse detection results
        valid_fi = []           # frame indices with successful detections
        valid_bboxes = []       # (1, 4) bbox
        valid_confs = []        # float
        valid_hw = []           # (H, W)
        valid_sides = []        # hand side "left"/"right" for each valid frame
        detection_valid = np.zeros(T, dtype=bool)

        for fi, yres in enumerate(yolo_out):
            if len(yres.boxes) == 0:
                continue
            clses = yres.boxes.cls.cpu().numpy()
            confs = yres.boxes.conf.cpu().numpy()
            bboxes = yres.boxes.xyxy.cpu().numpy()

            # Lock onto target_side: among boxes of this hand side only, pick the highest conf (WiLoR: 0=left, 1=right)
            # Detections of the other hand are always ignored, preventing frames with both hands co-occurring from sliding onto the wrong hand
            target_cls = _SIDE_TO_CLS[target_side]
            # Pick highest-conf box of target hand side; but if it heavily overlaps (IoU>thr)
            # a box of the other side with higher conf, the target box is defeated (actually the
            # other hand) -> skip this frame (cmask will fallback to no-info at inference).
            sel = select_target_box_dual_conf(clses, confs, bboxes, target_cls=target_cls)
            if sel is None:
                continue  # target not detected, or defeated by overlapping other-hand box
            valid_fi.append(fi)
            valid_bboxes.append(bboxes[sel:sel + 1].astype(np.float32))
            valid_confs.append(float(confs[sel]))
            valid_hw.append(raw_images_bgr[fi].shape[:2])
            valid_sides.append(target_side)  # after locking, always the target hand side
            detection_valid[fi] = True

        t_det = time.perf_counter()

        # ── 2. generate crops + affine transform ─────────────────────────────
        BBOX_SHAPE = self.model_cfg.MODEL.get("BBOX_SHAPE", None)
        rescale_factor = 2.0
        crop_bgr_list = []
        crop_intr_list = []
        box_center_list = []
        bbox_size_list = []

        for vi, fi in enumerate(valid_fi):
            img = raw_images_bgr[fi]
            bbox = valid_bboxes[vi]

            box = bbox[0].astype(np.float32)
            center = (box[2:4] + box[0:2]) / 2.0
            scale = rescale_factor * (box[2:4] - box[0:2]) / 200.0
            bbox_size = float(
                self._expand_to_aspect_ratio(scale * 200, target_aspect_ratio=BBOX_SHAPE).max()
            )
            center_x, center_y = center

            trans = self._gen_trans_from_patch_cv(
                center_x, center_y, bbox_size, bbox_size,
                img_size, img_size, 1.0, 0,
            )
            crop_bgr = cv2.warpAffine(
                img, trans, (img_size, img_size),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            )
            crop_bgr_list.append(crop_bgr)
            box_center_list.append(center.astype(np.float32))
            bbox_size_list.append(float(bbox_size))

            # crop intrinsics
            intr = raw_intrinsics_4d[fi] if fi < len(raw_intrinsics_4d) else np.zeros(4, np.float32)
            crop_intr_list.append(
                self._compute_crop_intrinsics(trans, intr)
            )

        # ── 3. HaMeR batched inference ───────────────────────────────────────
        backbone_features_all = np.zeros((T, 192, 1280), dtype=np.float32)
        landmarks_all = np.zeros((T, 21, 2), dtype=np.float32)
        crop_intrinsics_all = np.zeros((T, 4), dtype=np.float32)
        confidence_all = np.zeros(T, dtype=np.float32)
        crop_images_all = np.zeros((T, 3, 256, 256), dtype=np.uint8)

        for vi, fi in enumerate(valid_fi):
            crop_intrinsics_all[fi] = crop_intr_list[vi]
            confidence_all[fi] = valid_confs[vi]
            # crop image: BGR → RGB → (3, H, W)
            crop_images_all[fi] = crop_bgr_list[vi][:, :, ::-1].transpose(2, 0, 1)

        if len(valid_fi) > 0:
            crops_np = np.stack(crop_bgr_list)[:, :, :, ::-1].astype(np.float32)
            crops_t = torch.from_numpy(crops_np).permute(0, 3, 1, 2).to(dev)
            crops_t = (crops_t - self._mean) / self._std

            box_center_t = torch.tensor(
                np.stack(box_center_list), dtype=torch.float32, device=dev,
            )
            bbox_size_t = torch.tensor(
                bbox_size_list, dtype=torch.float32, device=dev,
            )
            img_size_t = torch.tensor(
                [[float(w), float(h)] for h, w in valid_hw],
                dtype=torch.float32, device=dev,
            )

            scaled_fl = float(
                self.model_cfg.EXTRA.FOCAL_LENGTH
                / self.model_cfg.MODEL.IMAGE_SIZE
                * img_size_t[0].max().item()
            )

            # Batched inference
            for s in range(0, len(crops_t), self.hamer_batch_size):
                e = min(s + self.hamer_batch_size, len(crops_t))
                out = self.model({"img": crops_t[s:e]})

                # backbone features (captured by the hook, output for the current sub-batch)
                bf = self._backbone_features.cpu().numpy()  # (batch, 192, 1280)
                for bi in range(e - s):
                    backbone_features_all[valid_fi[s + bi]] = bf[bi]

                # 2D landmarks: 3D joints -> full image -> normalize -> transform to crop
                pred_cam = out["pred_cam"]
                joints_3d = out["pred_keypoints_3d"].detach().cpu().numpy()

                cam_t_full = self._cam_crop_to_full(
                    pred_cam, box_center_t[s:e], bbox_size_t[s:e],
                    img_size_t[s:e], scaled_fl,
                ).detach().cpu().numpy()

                for bi in range(e - s):
                    fi = valid_fi[s + bi]
                    H, W = valid_hw[s + bi]

                    # Project onto the original image
                    kp2d = self._project_full_img(
                        joints_3d[bi], cam_t_full[bi], scaled_fl, W, H,
                    )
                    # Normalize to [0,1] (original image coordinates)
                    kp_n_orig = np.clip(
                        kp2d / np.array([W, H], dtype=np.float32), 0.0, 1.0,
                    ).astype(np.float32)

                    # Transform to crop coordinate system [0,1]
                    K_crop = crop_intrinsics_all[fi]
                    K_orig = raw_intrinsics_4d[fi] if fi < len(raw_intrinsics_4d) else np.zeros(4)
                    kp_n_crop = self._transform_landmarks_to_crop(
                        kp_n_orig, K_orig, K_crop, W, H,
                    )
                    landmarks_all[fi] = kp_n_crop

        # Save box_center/box_size/img_size/sides as full-length-T arrays (invalid frames are 0/default)
        box_center_all = np.zeros((T, 2), dtype=np.float32)
        bbox_size_all = np.zeros(T, dtype=np.float32)
        img_size_all = np.zeros((T, 2), dtype=np.float32)
        bbox_xyxy_all = np.zeros((T, 4), dtype=np.float32)
        conf_all = np.zeros(T, dtype=np.float32)
        sides_all = ["right"] * T
        for vi, fi in enumerate(valid_fi):
            box_center_all[fi] = box_center_list[vi]
            bbox_size_all[fi] = bbox_size_list[vi]
            img_size_all[fi] = [float(valid_hw[vi][1]), float(valid_hw[vi][0])]  # (W, H)
            bbox_xyxy_all[fi] = valid_bboxes[vi][0]
            conf_all[fi] = valid_confs[vi]
            sides_all[fi] = valid_sides[vi]

        t_hamer = time.perf_counter()

        return {
            "crop_images": torch.from_numpy(crop_images_all),
            "sides": sides_all,
            "bbox_xyxy": torch.from_numpy(bbox_xyxy_all),
            "bbox_conf": torch.from_numpy(conf_all),
            "backbone_features": torch.from_numpy(backbone_features_all),
            "hamer_landmarks": torch.from_numpy(landmarks_all),
            "crop_intrinsics": torch.from_numpy(crop_intrinsics_all),
            "hamer_confidence": torch.from_numpy(confidence_all),
            "detection_valid": torch.from_numpy(detection_valid),
            "box_center": torch.from_numpy(box_center_all),
            "box_size": torch.from_numpy(bbox_size_all),
            "img_size": torch.from_numpy(img_size_all),
            "det_time_s": t_det - t0,
            "hamer_time_s": t_hamer - t_det,
        }

    # ── coordinate conversion (consistent with _transform_landmarks_to_crop in dataset.py) ──

    @staticmethod
    def _transform_landmarks_to_crop(
        landmarks_orig_n: np.ndarray,
        K_orig: np.ndarray,
        K_crop: np.ndarray,
        W: int,
        H: int,
    ) -> np.ndarray:
        """Original-image normalized [0,1] -> crop normalized [0,1]."""
        fx, fy, cx, cy = K_orig
        fxc, fyc, cxc, cyc = K_crop

        sx = fxc / fx
        tx = cxc - fxc * cx / fx
        sy = fyc / fy
        ty = cyc - fyc * cy / fy

        pts_x = landmarks_orig_n[:, 0] * W
        pts_y = landmarks_orig_n[:, 1] * H

        crop_x = sx * pts_x + tx
        crop_y = sy * pts_y + ty

        result = np.stack([crop_x / 256.0, crop_y / 256.0], axis=-1)
        return result.astype(np.float32)
