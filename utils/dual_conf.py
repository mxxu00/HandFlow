"""Dual-hand confidence veto for target-box selection.

Background: the YOLO detector (WiLoR) occasionally assigns both left and right labels to the
same hand (highly overlapping boxes). In that case we arbitrate by confidence: if the
opposite-side box has higher confidence, the target box is unreliable and is vetoed (treated
as no detection). Non-overlapping left/right boxes (genuinely different hands in the frame) do
not affect the target box, so true two-hand frames are not wrongly suppressed.

Used by the box-selection logic in preprocessing/hamer_crop_and_skeleton.py and
utils/online_hamer.py. Once a frame is vetoed it has no detection (conf=0); at inference time
the cmask mechanism falls back to the no-information state automatically.
"""
import numpy as np


def iou(a, b):
    """a, b: xyxy of shape (4,). Returns the IoU."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def select_target_box_dual_conf(clses, confs, bboxes, target_cls, iou_thr=0.7):
    """Select the highest-confidence box of the target class; veto it if an opposite-side box wins.

    Mechanism: among boxes of ``target_cls`` pick the highest-confidence "main" box, then find the
    opposite-side box (cls = 1 - target_cls) with the largest IoU to it. If that IoU exceeds
    ``iou_thr`` and the opposite-side confidence is higher than the main box's, the main box is
    vetoed (the location actually belongs to the opposite-side hand) and None is returned.
    Otherwise the main box index is returned.

    Args:
        clses: (N,) YOLO box classes (WiLoR: 0=left, 1=right).
        confs: (N,) YOLO box confidences.
        bboxes: (N, 4) xyxy.
        target_cls: target hand class (1=right, 0=left).
        iou_thr: IoU threshold above which two boxes count as overlapping (same hand). Default 0.7.
    Returns:
        The main box index (int), or None (no target box / vetoed).
    """
    target_mask = clses == target_cls
    if not target_mask.any():
        return None
    sel = int(np.argmax(np.where(target_mask, confs, -1.0)))
    other_mask = clses == (1 - target_cls)
    if other_mask.any():
        main_box = bboxes[sel]
        best_iou, best_other_conf = 0.0, 0.0
        for oi in np.where(other_mask)[0]:
            v = iou(main_box, bboxes[oi])
            if v > best_iou:
                best_iou = v
                best_other_conf = float(confs[oi])
        if best_iou > iou_thr and best_other_conf > float(confs[sel]):
            return None  # vetoed by the opposite side
    return sel
