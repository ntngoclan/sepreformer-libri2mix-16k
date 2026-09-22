ARG CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG TORCH_VERSION=2.1.2
ARG TORCHVISION_VERSION=0.16.2
ARG TORCHAUDIO_VERSION=2.1.2
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        libsndfile1 \
        python3 \
        python3-dev \
        python-is-python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/SepReformer

COPY requirements-runtime.txt /tmp/requirements-runtime.txt
RUN python3 -m pip install --upgrade pip setuptools==80.9.0 wheel \
    && python3 -m pip install \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url "${TORCH_INDEX_URL}" \
    && python3 -m pip install -r /tmp/requirements-runtime.txt

COPY . .

RUN mkdir -p run_logs runs initializations

CMD ["python3", "scripts/preflight.py"]
