#!/bin/bash
# HandFlow inference + render environment (training deps excluded)
# Usage: bash setup_env.sh [env_name=handflow] [python=3.10]
set -e

ENV_NAME=${1:-handflow}
PYTHON_VERSION=${2:-3.10}

echo "=========================================="
echo "HandFlow inference environment setup: $ENV_NAME (Python $PYTHON_VERSION)"
echo "=========================================="

# Create conda environment
eval "$(conda shell.bash hook)"
if ! conda env list | grep -qE "^${ENV_NAME} "; then
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"

echo ""
echo "Installing PyTorch 2.7.0 + CUDA 12.8..."
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "Installing HaMeR backbone (submodule, includes its deps smplx/yacs etc.)..."
pip install -e third_party/hamer

echo ""
echo "Installing HandFlow inference + render deps..."
pip install numpy scipy omegaconf tqdm opencv-python "imageio[ffmpeg]" matplotlib
pip install ultralytics manopth

echo ""
echo "Installing pytorch3d (Phong renderer; build from source — needs nvcc, matches torch 2.7+cu128)..."
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8"

echo ""
echo "=========================================="
echo "Inference environment ready: conda activate $ENV_NAME"
echo "=========================================="
echo ""
echo "Next — set the weight-path environment variables (see README for weight downloads):"
echo "  export HAMER_CKPT=/path/to/hamer.ckpt"
echo "  export DETECTOR_CKPT=/path/to/detector.pt        # from WiLoR"
echo "  export MANO_ROOT=/path/to/mano                    # MANO model"
echo "  export HANDFLOW_NORMALIZATION_STATS=/path/to/normalization_stats.npz"
echo ""
echo "Optional — for moving-camera world trajectory (demo.py without --fix_camera): bash setup_vipe_env.sh"
