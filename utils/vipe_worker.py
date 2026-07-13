"""ViPE SLAM worker — invoked via subprocess + conda run -n vipe.

Provides:
- run_vipe_slam(): caller side (runs in the handflow env); launches a separate vipe env via subprocess
- main():          worker side (runs in the vipe env); actually loads ViPE, runs SLAM, and writes JSON output

demo.py calls run_vipe_slam only in moving-camera mode (i.e. without --fix_camera): with
explicit/default intrinsics it uses the gt_intr variant (c2w only), and with --intrinsics auto
it uses the default variant (intrinsics + c2w).
The worker is triggered by `conda run -n vipe python utils/vipe_worker.py --worker ...`,
so vipe imports are deliberately deferred into main() (top-level imports would fail in the handflow env).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

# ViPE submodule root directory (can be overridden by the VIPE_ROOT environment variable)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIPE_ROOT = Path(os.environ.get("VIPE_ROOT", str(_PROJECT_ROOT / "third_party" / "vipe")))


def run_vipe_slam(
    images_dir: str,
    output_dir: str,
    variant: str = "default",
    gt_intrinsics_4d: Optional[np.ndarray] = None,
    vipe_env: str = "vipe",
    seq_name: Optional[str] = None,
    timeout: int = 3600,
) -> dict:
    """Caller side: run ViPE SLAM via `conda run -n <vipe_env>`.

    Args:
        images_dir:        directory of frame images (already extracted into images)
        output_dir:        output directory (slam_result.json is written here)
        variant:           "default" (ViPE estimates intrinsics + c2w)
                           | "gt_intr" (given intrinsics, estimate c2w only)
        gt_intrinsics_4d:  [fx, fy, cx, cy] used when variant=gt_intr
        vipe_env:          name of the conda environment containing ViPE (default "vipe")
    Returns:
        {"poses": (T,4,4) c2w, "intrinsics": (T,4) or (4,), "slam_time_s": float}
    """
    os.makedirs(output_dir, exist_ok=True)
    output_json = str(Path(output_dir) / "slam_result.json")
    seq_name = seq_name or Path(images_dir).name

    cmd = [
        "conda", "run", "-n", vipe_env, "--no-capture-output",
        "python", str(Path(__file__).resolve()), "--worker",
        "--img_dir", images_dir,
        "--seq_name", seq_name,
        "--variant", variant,
        "--output", output_json,
    ]
    if variant == "gt_intr" and gt_intrinsics_4d is not None:
        intr = np.asarray(gt_intrinsics_4d).ravel()[:4]
        cmd += ["--gt_intr_str", ",".join(str(float(x)) for x in intr)]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=str(_PROJECT_ROOT)
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ViPE SLAM failed (returncode={result.returncode}):\n"
            f"{result.stderr.strip()[-800:]}"
        )
    if not os.path.exists(output_json):
        raise RuntimeError(f"ViPE worker produced no output: {result.stderr[-400:]}")

    with open(output_json) as f:
        data = json.load(f)
    if not data.get("ok"):
        raise RuntimeError(f"ViPE worker error: {data.get('error')}")

    return {
        "poses": np.array(data["poses"], dtype=np.float32),           # (T, 4, 4) c2w
        "intrinsics": np.array(data["intrinsics"], dtype=np.float32),  # (T, 4) or (4,)
        "slam_time_s": float(data.get("slam_time_s", 0.0)),
    }


# ── Worker (executed inside the vipe environment) ──────────────────────────────────────────

def _worker_main() -> None:
    """Actually loads ViPE, runs SLAM, and writes results to JSON. Only runnable in the vipe environment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--seq_name", default="seq")
    parser.add_argument("--variant", default="default", choices=["default", "gt_intr"])
    parser.add_argument("--gt_intr_str", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from omegaconf import OmegaConf
    import torch
    from vipe.streams.frame_dir_stream import FrameDirStream
    from vipe.slam.system import SLAMSystem
    from vipe.streams.base import ProcessedVideoStream, StreamProcessor, FrameAttribute
    from vipe.utils.cameras import CameraType

    slam_cfg = OmegaConf.load(str(VIPE_ROOT / "configs" / "slam" / "default.yaml"))
    slam_cfg.optimize_intrinsics = (args.variant != "gt_intr")

    try:
        stream = FrameDirStream(path=Path(args.img_dir),
                                name=args.seq_name.replace("/", "_"))

        if args.variant == "gt_intr":
            fx, fy, cx, cy = [float(x) for x in args.gt_intr_str.split(",")]

            class GTIntrinsicsProcessor(StreamProcessor):
                def update_attributes(self, prev):
                    return prev | {FrameAttribute.INTRINSICS}

                def __call__(self, frame_idx, frame):
                    frame.intrinsics = torch.as_tensor([fx, fy, cx, cy]).float()
                    frame.camera_type = CameraType.PINHOLE
                    return frame

            stream = ProcessedVideoStream(stream, [GTIntrinsicsProcessor()])
        else:
            from vipe.pipeline.processors import GeoCalibIntrinsicsProcessor
            stream = ProcessedVideoStream(
                stream, [GeoCalibIntrinsicsProcessor(stream, camera_type=CameraType.PINHOLE)],
            )

        device = torch.device("cuda")
        slam_system = SLAMSystem(device=device, config=slam_cfg)
        t0 = time.perf_counter()
        slam_output = slam_system.run([stream], camera_type=CameraType.PINHOLE)
        slam_time = time.perf_counter() - t0

        traj_mat = slam_output.trajectory.matrix()
        if hasattr(traj_mat, "detach"):
            traj_mat = traj_mat.detach().cpu().numpy()
        intr_tensor = slam_output.intrinsics
        if hasattr(intr_tensor, "detach"):
            intr_tensor = intr_tensor.detach().cpu().numpy()

        n_frames = len(traj_mat)
        if len(intr_tensor) == 1 and n_frames > 1:
            intr_tensor = np.tile(intr_tensor, (n_frames, 1))
        N = min(len(traj_mat), len(intr_tensor))

        result = {
            "poses": traj_mat[:N].tolist(),
            "intrinsics": intr_tensor[:N].tolist(),
            "slam_time_s": slam_time,
            "ok": True,
        }
    except Exception as e:
        result = {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()[-500:],
            "slam_time_s": 0.0,
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    _worker_main()
