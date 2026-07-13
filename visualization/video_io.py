"""Video I/O utilities: read frames (cv2.VideoCapture) + encode mp4 (imageio/ffmpeg).

Shared by scripts/demo.py.
"""

from typing import List, Tuple

import cv2
import numpy as np


def read_video_frames(path: str) -> Tuple[List[np.ndarray], float, Tuple[int, int]]:
    """Read all video frames (BGR uint8).

    Returns:
        frames_bgr: list of (H, W, 3) BGR uint8
        fps:       frame rate (30 if unreadable)
        (H, W):    frame size
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"Video has no valid frames: {path}")

    H, W = frames[0].shape[:2]
    return frames, float(fps), (H, W)


def write_video_ffmpeg(
    frames_bgr: List[np.ndarray], path: str, fps: float, quality: int = 8
) -> None:
    """Encode a BGR frame sequence as mp4 (imageio + libx264).

    Args:
        frames_bgr: list of (H, W, 3) BGR uint8
        path:      output mp4 path
        fps:       frame rate
        quality:   imageio quality (0-10, higher is better)
    """
    import imageio.v2 as imageio

    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=quality,
                                macro_block_size=1)
    try:
        for frame in frames_bgr:
            # BGR -> RGB
            writer.append_data(frame[:, :, ::-1])
    finally:
        writer.close()
