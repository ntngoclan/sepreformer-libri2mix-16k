#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-SepReformer_PARR_Libri2Mix_16K}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export CUBLAS_WORKSPACE_CONFIG=:4096:8

case "${MODEL}" in
  SepReformer_Base_Libri2Mix_16K|SepReformer_Base_VnSpeechMix_16K|SepReformer_PARR_Libri2Mix_16K) ;;
  *) echo "Unsupported MODEL=${MODEL}" >&2; exit 2 ;;
esac

cd "${ROOT_DIR}"
mkdir -p run_logs
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="run_logs/test_${MODEL}_${STAMP}.log"

python -u run.py --model "${MODEL}" --engine-mode test "$@" 2>&1 | tee -a "${LOG_PATH}"
