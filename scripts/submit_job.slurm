#!/usr/bin/env bash
#SBATCH --job-name=sepreformer
#SBATCH --output=run_logs/slurm_%j.out
#SBATCH --error=run_logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=mps:2
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# UIT-specific allocation adapter, based on HuongDanSuDungSlurm.docx.
# Submit FROM the repo root, after mkdir -p run_logs. Arguments are the command.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script using sbatch, not bash on the login node}"
: "${SLURM_SUBMIT_DIR:?Missing Slurm submission directory}"
if [[ $# -eq 0 ]]; then
    echo 'Usage: sbatch scripts/submit_job.slurm python -B scripts/...' >&2
    exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "${PROJECT_DIR}"
if [[ ! -f run.py || ! -d models ]]; then
    echo 'Submit from the SepReformer repository root (or set PROJECT_DIR).' >&2
    exit 2
fi

# Optional: exact Python module name verified with module avail on UIT.
# Otherwise inherit the environment exported by sbatch (the default).
if [[ -n "${PYTHON_MODULE:-}" ]]; then
    module clear -f
    module load shared "${PYTHON_MODULE}"
fi
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

# MiB of requested free VRAM, NOT CPU RAM and not a guarantee of model fit.
# Initial smoke-test request: 24 GiB; adjust using measured peak usage.
REQUIRED_VRAM="${REQUIRED_VRAM:-24576}"
if [[ ! "${REQUIRED_VRAM}" =~ ^[1-9][0-9]*$ ]]; then
    echo 'REQUIRED_VRAM must be a positive integer in MiB.' >&2
    exit 2
fi
if [[ ! -x /usr/local/bin/gpu_check.sh ]]; then
    echo 'UIT GPU helper is unavailable; compare the current campus job template.' >&2
    exit 1
fi

# This exception is explicitly required by the UIT MPS allocation workflow.
# Generic train/evaluate wrappers then preserve the helper-selected device.
unset CUDA_VISIBLE_DEVICES
if CHECK_OUT=$(/usr/local/bin/gpu_check.sh "${REQUIRED_VRAM}" "${SLURM_JOB_ID}"); then
    EXIT_CODE=0
else
    EXIT_CODE=$?
fi
case "${EXIT_CODE}" in
    0) ;;
    10) printf '%s\n' "${CHECK_OUT}"; exit 0 ;; # helper requeues the job
    *) printf '%s\n' "${CHECK_OUT}" >&2; exit 1 ;; # includes documented code 11
esac
if [[ -z "${CHECK_OUT//[[:space:]]/}" ]]; then
    echo 'GPU helper returned an empty device selection.' >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${CHECK_OUT}"

# Keep the job-specific paths used by the campus template so site hooks agree.
# Do not delete existing pipes or issue global daemon commands from this adapter.
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-job${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-job${SLURM_JOB_ID}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

echo "Job: ${SLURM_JOB_ID}; GPU: ${CUDA_VISIBLE_DEVICES}; requested free VRAM: ${REQUIRED_VRAM} MiB"
echo "Project: ${PROJECT_DIR}; Python: $(command -v python)"
printf 'Command:'; printf ' %q' "$@"; printf '\n'
"$@"
