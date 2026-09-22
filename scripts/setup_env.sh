#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.1.2}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.16.2}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.1.2}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools==80.9.0 wheel
python -m pip install \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"
python -m pip install -r requirements-runtime.txt

echo "Environment installed at ${VENV_DIR}"
echo "Next: source ${VENV_DIR}/bin/activate && python scripts/preflight.py"
