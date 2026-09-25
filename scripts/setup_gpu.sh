#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Run inside your activated Python 3.11+ environment on a Linux GPU app.
for command in python nvidia-smi nvcc g++; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing $command. Use a SageMaker CUDA development image with NVIDIA drivers, CUDA toolkit and g++." >&2
    exit 1
  fi
done
python -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'
nvidia-smi
nvcc --version
python -m pip install -r requirements.txt 'cmake>=3.28,<4' ninja
# Bound build parallelism; source compilation can itself consume considerable RAM.
export CMAKE_BUILD_PARALLEL_LEVEL=2
python -m pip install --force-reinstall --no-deps --no-cache-dir \
  --no-binary=lightgbm --config-settings=cmake.define.USE_CUDA=ON 'lightgbm==4.6.0'
python run.py doctor --training-config configs/full_gpu.json
