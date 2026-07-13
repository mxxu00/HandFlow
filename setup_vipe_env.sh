#!/bin/bash
# ViPE SLAM standalone environment (invoked by demo.py via conda run -n vipe when --fix_camera is unset)
# Install only if you need the world-view trajectory (demo.py without --fix_camera); --fix_camera mode skips ViPE.
#
# ViPE's CUDA extensions are compiled against torch 2.7.0+cu128, matching the handflow environment.
# Usage: bash setup_vipe_env.sh [env_name=vipe]
set -e

ENV_NAME=${1:-vipe}

echo "=========================================="
echo "ViPE SLAM environment setup: $ENV_NAME"
echo "=========================================="

eval "$(conda shell.bash hook)"
if ! conda env list | grep -qE "^${ENV_NAME} "; then
    conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"

echo ""
echo "Installing PyTorch 2.7.0 + CUDA 12.8 (matching handflow)..."
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "Installing CUDA 12.8 build tools (nvcc, required to compile the ViPE extension)..."
conda install -y -c "nvidia/label/cuda-12.8.0" cuda-nvcc cuda-cudart-dev
conda install -y -c conda-forge gxx_linux-64=11  # gcc 11

echo ""
echo "Installing ViPE (submodule; compiles the vipe_ext CUDA extension + downloads Eigen)..."
pip install -e third_party/vipe

echo ""
echo "=========================================="
echo "ViPE environment ready: conda activate $ENV_NAME"
echo "=========================================="
echo ""
echo "demo.py will invoke this environment automatically via 'conda run -n ${ENV_NAME}' (moving-camera mode only)."
echo "The first build of vipe_ext downloads Eigen 3.4 online; ensure network access."
