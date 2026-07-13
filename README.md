# HandFlow

**Fully Generative 4D Hand Recovery with Flow Matching.**

HandFlow reconstructs 4D (temporal 3D) hand poses from monocular RGB video. At its core is a Rectified Flow Denoiser: HaMeR ViT-H extracts image features, combined with 2D skeletons as multimodal conditioning, to directly regress a sequence of MANO parameters. ViPE SLAM can optionally supply camera geometry to reconstruct the hand trajectory in the world coordinate frame.

> This repository is the **V1 open-source release**: it includes inference + visualization demos only, without training code.

---

## Pipeline

A single entry point, `scripts/demo.py`, produces two mp4s per run: a camera-view mesh **overlay** and an **orthographic** view. The `--fix_camera` flag selects how camera geometry is obtained:

```
demo.py
┌──────────────────────────────────────────────────────────────────┐
│ Video (--input)                                                   │
│  ├─ Intrinsics: --intrinsics fx,fy,cx,cy | auto |                 │
│  │               unset → generic default 600,600,W/2,H/2          │
│  ├─ Camera geometry (c2w):                                        │
│  │    --fix_camera  → c2w = identity, ViPE skipped                │
│  │    (default)      → ViPE SLAM estimates c2w                    │
│  ├─ Online HaMeR → FM Denoiser → MANO FK → verts_cam (m)          │
│  └─ verts_world = c2w · verts_cam                                 │
│                                                                   │
│ → overlay.mp4   clean camera-space MANO mesh over the RGB frames  │
│ → ortho.mp4     --fix_camera : level side view, fixed camera       │
│                 (default)   : world topdown — camera trajectory    │
│                               line + per-frame hand mesh           │
└──────────────────────────────────────────────────────────────────┘
```

- **`--fix_camera`** (fixed camera, e.g. a tripod-mounted rig like DexYCB): ViPE is skipped, `c2w` is identity, and the orthographic view assumes a **level, fixed camera** — a side view of the camera-space hand motion (the camera stays at the origin; only the hand moves).
- **default** (moving camera, e.g. a head-mounted rig like HOT3D): ViPE SLAM estimates `c2w`; the orthographic view is a world-space trajectory view (camera trajectory line + per-frame hand mesh). Because SLAM trajectory error accumulates, the world result looks slightly more "skewed" than the overlay; the orthographic view is exactly where this camera motion becomes visible.

---

## Installation

### 1. Clone (with submodules)

```bash
git clone --recurse-submodules https://github.com/mxxu00/HandFlow.git
cd HandFlow
# If you already cloned without submodules:
git submodule update --init --recursive
```

Submodules:
- `third_party/hamer` — HaMeR ViT-H backbone ([geopavlakos/hamer](https://github.com/geopavlakos/hamer))
- `third_party/vipe` — ViPE SLAM ([nv-tlabs/vipe](https://github.com/nv-tlabs/vipe), only needed in moving-camera mode)

### 2. Inference + rendering environment

```bash
bash setup_env.sh           # creates conda env handflow (Python 3.10 + torch 2.7.0+cu128)
conda activate handflow
pip install -e third_party/hamer   # HaMeR and its dependencies
```

> `setup_env.sh` also builds **pytorch3d** (the Phong renderer) from source, which requires `nvcc`; if the build fails, see the [pytorch3d install guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). `--fix_camera` mode then works without the ViPE environment.

### 3. ViPE environment (only needed in moving-camera mode, optional)

ViPE ships a CUDA extension that must be compiled in a **separate conda environment**; `demo.py` invokes it automatically via `conda run -n vipe` whenever `--fix_camera` is **unset**.

```bash
bash setup_vipe_env.sh      # creates conda env vipe (with nvcc + Eigen, compiles vipe_ext)
```

> The first build requires internet access to download Eigen 3.4. ViPE's torch version (2.7.0+cu128) matches handflow, so the extension compiles correctly.

---

## Weights & Data

Download the following weights yourself and specify their paths via environment variables:

| Environment variable | Description | Source |
|---|---|---|
| `HAMER_CKPT` | `hamer.ckpt` (HaMeR ViT-H weights) | [HaMeR releases](https://github.com/geopavlakos/hamer) |
| `DETECTOR_CKPT` | `detector.pt` (YOLO hand detector, **from WiLoR**) | [WiLoR](https://github.com/rolpotamias/WiLoR) `pretrained_models/detector.pt` |
| `MANO_ROOT` | MANO model directory (`.pkl`) | [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) (register to download) |
| `HANDFLOW_NORMALIZATION_STATS` | `normalization_stats.npz` | [HuggingFace](https://huggingface.co/mxxu00/HandFlow) |
| `--fm_ckpt` | HandFlow FM checkpoint (`handflow_denoiser.pt`) | [HuggingFace](https://huggingface.co/mxxu00/HandFlow) (passed as a CLI argument) |

Example setup (add to `~/.bashrc`):
```bash
export HAMER_CKPT=/path/to/hamer.ckpt
export DETECTOR_CKPT=/path/to/detector.pt
export MANO_ROOT=/path/to/mano
export HANDFLOW_NORMALIZATION_STATS=/path/to/normalization_stats.npz
```

> **HandFlow weights** (`normalization_stats.npz` + `handflow_denoiser.pt`) are hosted on [HuggingFace](https://huggingface.co/mxxu00/HandFlow). Download both with:
>
> ```bash
> hf download mxxu00/HandFlow --local-dir ./weights
> ```
>
> Then point `HANDFLOW_NORMALIZATION_STATS` at `normalization_stats.npz` and pass `--fm_ckpt ./weights/handflow_denoiser.pt` to `demo.py`.

---

## Demo

Two short sample clips are bundled in `demo/` — one from [DexYCB](https://dex-ycb.github.io/) (fixed camera) and one from [HOT3D](https://hot3d.github.io/) (moving camera). The commands below run the full pipeline on each.

**DexYCB — fixed camera (no ViPE needed):**
```bash
conda activate handflow
python scripts/demo.py --input demo/dexycb_sample.mp4 \
    --fm_ckpt ./weights/handflow_denoiser.pt \
    --intrinsics 616.2495,615.87665,321.6139,244.8281 \
    --fix_camera --output_dir output/dexycb
# → output/dexycb/overlay.mp4
# → output/dexycb/ortho_side.mp4
```

**HOT3D — moving camera (requires the vipe env):**
```bash
python scripts/demo.py --input demo/hot3d_sample.mp4 \
    --fm_ckpt ./weights/handflow_denoiser.pt \
    --intrinsics 609.7035,609.7035,707.6459,704.7342 \
    --output_dir output/hot3d
# → output/hot3d/overlay.mp4
# → output/hot3d/ortho_topdown.mp4
```

---

## Usage

```bash
# Your own video, known intrinsics, fixed camera (e.g. webcam on a stand)
python scripts/demo.py --input video.mp4 --fm_ckpt <ckpt> \
    --intrinsics fx,fy,cx,cy --fix_camera --output_dir output/demo

# Your own video, known intrinsics, moving camera (ViPE estimates c2w only)
python scripts/demo.py --input video.mp4 --fm_ckpt <ckpt> \
    --intrinsics fx,fy,cx,cy --output_dir output/demo

# Your own video, unknown intrinsics, moving camera (ViPE estimates intrinsics + c2w)
python scripts/demo.py --input video.mp4 --fm_ckpt <ckpt> \
    --intrinsics auto --output_dir output/demo
```

Key options:
- `--intrinsics` `fx,fy,cx,cy` (recommended) | `auto` (ViPE estimates; moving-camera only) | unset → generic default `600,600,W/2,H/2`.
- `--fix_camera` : fixed-camera mode (skip ViPE, `c2w=identity`, level side ortho view). Incompatible with `--intrinsics auto`.
- `--view` `topdown|side` : ortho view direction. Defaults to `side` under `--fix_camera`, else `topdown`.

> Left-hand videos must be mirrored to right-hand first (the FM model is trained on right-hand data).

---

## Project Layout

```
HandFlow/
├── scripts/
│   └── demo.py             # unified demo: cam overlay + orthographic view
├── demo/                   # bundled demo clips (dexycb_sample.mp4, hot3d_sample.mp4)
├── model/                  # Rectified Flow Denoiser + feature extractors
├── utils/                  # inference / MANO FK / online HaMeR / ViPE worker
├── visualization/          # pytorch3d Phong renderer (overlay + orthographic trajectory) + video I/O
├── preprocessing/          # crop utilities (used by online_hamer)
├── configs/                # model.yaml + inference.yaml (paths via ${oc.env:})
├── third_party/
│   ├── hamer/              # submodule
│   └── vipe/               # submodule (optional, moving-camera mode only)
├── setup_env.sh            # inference environment
└── setup_vipe_env.sh       # ViPE environment (optional, moving-camera mode only)
```

---

## TODO List

The current release (V1) ships **inference + visualization only**. The following components are planned for future release:

- [ ] Training code
- [ ] Data preprocessing code
- [ ] Evaluation code
