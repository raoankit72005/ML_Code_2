#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -c 'import sys; assert sys.version_info >= (3,11), "Use Python 3.11 or newer"'
# Run in a dedicated environment with a recent NVIDIA driver.
python -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-hybrid.txt
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: check SageMaker GPU app and driver"; print(torch.cuda.get_device_name(0))'
bash scripts/setup_gpu.sh
python -m pip freeze > environment-resolved.txt
