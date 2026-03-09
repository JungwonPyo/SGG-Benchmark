#!/bin/bash

# This script adds NVIDIA libraries installed via pip in the conda environment to LD_LIBRARY_PATH
# This is required for ONNX Runtime and other tools to find cuDNN, cuBLAS, etc.

if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: No conda environment active. Please activate sgg_benchmark first."
    if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then return 1; else exit 1; fi
fi

PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
SITE_PACKAGES="$CONDA_PREFIX/lib/python$PYTHON_VERSION/site-packages"

if [ -d "$SITE_PACKAGES" ]; then
    # 1. Find all 'lib' directories under the nvidia site-packages
    # 2. Find all directories ending in '_libs' (like tensorrt_libs, opencv_python.libs)
    NEW_PATHS=$(find "$SITE_PACKAGES/nvidia" -name "lib" -type d 2>/dev/null | tr '\n' ':')
    NEW_PATHS="${NEW_PATHS}$(find "$SITE_PACKAGES" -maxdepth 1 -name "*libs" -type d 2>/dev/null | tr '\n' ':')"
    
    # Prepend to LD_LIBRARY_PATH if not already present
    # Filter out empty paths and duplicates
    export LD_LIBRARY_PATH="${NEW_PATHS}${LD_LIBRARY_PATH}"
    
    # Ensure it's exported to the current session
    echo "Added NVIDIA and TensorRT libraries to LD_LIBRARY_PATH from $SITE_PACKAGES"
    echo "Current LD_LIBRARY_PATH now includes cuDNN/cuBLAS from conda env."
else
    echo "Warning: site-packages path not found at $SITE_PACKAGES"
fi
