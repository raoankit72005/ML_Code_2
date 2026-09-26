# Build on x86_64. CUDA compilation happens once here, never during a paid Training Job.
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 CMAKE_BUILD_PARALLEL_LEVEL=2
RUN apt-get update && apt-get install -y --no-install-recommends g++ git libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-hybrid.txt ./
RUN python -m pip install 'cmake>=3.28,<4' ninja && python -m pip install -r requirements-hybrid.txt
# Force source build: binary wheels ignore CUDA CMake flags. Target may be
# SM86 (A10G) or SM89 (L40S); always rebuild when changing this argument.
ARG CUDA_ARCH=89
RUN python -m pip install --force-reinstall --no-deps --no-cache-dir --no-binary=lightgbm \
    --config-settings=cmake.define.USE_CUDA=ON \
    --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} 'lightgbm==4.6.0'
COPY . /app
RUN python -m compileall -q src hybrid.py gpu_worker.py sagemaker && python -m pip freeze > /app/environment-resolved.txt
ENV OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
ENTRYPOINT ["python", "/app/sagemaker/train.py"]
