#!/bin/bash
# install_uv.sh - Fast and clean installation using uv

set -e

# 1. Install uv if not found
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# 2. Create a fresh virtual environment
echo "Creating virtual environment with Python 3.11..."
uv venv --python 3.11

# 3. Use the environment
# Note: In a script, we use 'source' or just call uv run
source .venv/bin/activate

# 4. Install PyTorch
# We require torch>=2.0 for torch.compile and SDPA support.
# uv will automatically find the best version for your system.
echo "Installing PyTorch..."
uv pip install "torch>=2.0" "torchvision>=0.15"

# 5. Install dependencies from requirements.txt
echo "Installing project dependencies..."
uv pip install -r requirements.txt

# 5b. Install Ultralytics CLIP (text encoder) from GitHub
echo "Installing Ultralytics CLIP from GitHub..."
uv pip install git+https://github.com/ultralytics/CLIP.git

# 5c. Install ONNX and ONNX Runtime (GPU if available)
echo "Installing ONNX and ONNX Runtime (GPU if available)..."
uv pip install onnx onnx-simplifier onnxruntime-tools || true
# Prefer onnxruntime-gpu; fall back to cpu wheels if GPU package unavailable
if uv pip install onnxruntime-gpu; then
    echo "Installed onnxruntime-gpu"
else
    echo "onnxruntime-gpu not available via pip for this platform; installing CPU onnxruntime"
    uv pip install onnxruntime
fi

# 5d. Attempt to install TensorRT Python bindings (may require NVIDIA repo)
echo "Attempting to install TensorRT Python bindings (tensorrt)..."
if uv pip install tensorrt; then
    echo "Installed tensorrt Python package"
else
    echo "Could not install 'tensorrt' via pip. If you need TensorRT, follow NVIDIA's instructions:"
    echo "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
    echo "You may also need to run scripts/setup_cuda_libs.sh to expose TensorRT shared libraries."
fi
# 6. Install the project in editable mode
echo "Installing sgg_benchmark in editable mode..."
uv pip install -e .
source .venv/bin/activate

echo "------------------------------------------------"
echo "Installation complete!"
echo "To activate the environment: source .venv/bin/activate"
echo "To run commands: uv run <command>"
echo "------------------------------------------------"
