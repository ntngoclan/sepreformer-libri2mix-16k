# Deployment on a CUDA training resource

For the UIT cluster's site-specific MPS/helper workflow, see [UIT_SLURM.md](UIT_SLURM.md)
and `scripts/submit_job.slurm`.

The deployment filenames are hardware-neutral. They can be used on a workstation,
cloud GPU VM, container or Slurm GPU allocation with a compatible NVIDIA CUDA
runtime and enough memory. Training/evaluation still require CUDA; CPU regression
and selected smoke checks do not. This change does not add ROCm, MPS or distributed
training support.

Use [common protocol v2](COMMON_16K_PROTOCOL.md) for the four baseline/LTRR configs,
dataset preparation, pilots, checkpoints and resume rules.

## Environment

The default reference environment remains Python 3.10/3.11, PyTorch 2.1.2 and its
CUDA 12.1 build, with pinned dependencies. It is not tied to a GPU model and is
not guaranteed compatible with every GPU generation or driver.

On Linux, provision build tools, libsndfile and Python venv support, then:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The setup script accepts PYTHON_BIN and VENV_DIR. For a resource needing a different
PyTorch/CUDA build, select a mutually compatible set using TORCH_VERSION,
TORCHVISION_VERSION, TORCHAUDIO_VERSION and TORCH_INDEX_URL environment variables.
If the cluster already supplies an environment, activate it and install only the
needed dependencies from requirements-runtime.txt instead of recreating it.
Record the chosen versions; use the same environment for a controlled model pair.

## Resource allocation

Run inside the GPU allocation on Slurm. tmux keeps a shell session alive but does
not allocate a GPU or extend a scheduler time limit. The wrappers preserve
CUDA_VISIBLE_DEVICES as supplied by the scheduler, container or user, including
an explicitly empty value. They do not reset it to physical GPU 0.

Default engine.gpuid is "0": logical GPU 0 among the visible devices. If multiple
GPUs are visible, the default still uses only the first. Merely exposing several
devices does not create distributed training. Changing device topology, batch,
workers or software must be recorded and applied consistently within model pairs.

## Checks and execution

For baseline/LTRR on either dataset, select the model explicitly:

```bash
python -B scripts/preflight_vnspeechmix_baseline.py --root . --model SepReformer_Base_Libri2Mix_16K --device cuda --samples 64000 --steps 2 --workers 12 --report run_logs/libri_base_gpu.json

MODEL=SepReformer_Base_Libri2Mix_16K bash scripts/train.sh --seed 0 --run-id libri_base_common_v2_s0
MODEL=SepReformer_Base_Libri2Mix_16K bash scripts/evaluate.sh --seed 0 --run-id libri_base_common_v2_s0
```

Use the corresponding LTRR or VN model name as needed. Prepare data and its manifest
first, as described in the common protocol guide. Run the models separately.

The optional `python scripts/preflight.py` checks the Libri2Mix baseline/PARR pair,
including GPU operations, model gradients, data, evaluator and serialization.
It tests logical GPU 0 without a GPU-name or compute-capability whitelist; other
visible GPUs are not certified. Its report records the installed Torch/CUDA
versions, GPU details and visible-device count. It defaults to
run_logs/preflight_report.json. Skip/quick options produce a partial report.
Passing a smoke check does not establish convergence or full-epoch memory safety.

## Container

```bash
docker build -t sepreformer:runtime .
mkdir -p runs run_logs initializations

docker run --rm --gpus all --shm-size=8g \
  -v "$PWD/data/Libri2Mix:/workspace/SepReformer/data/Libri2Mix:ro" \
  -v "$PWD/runs:/workspace/SepReformer/runs" \
  -v "$PWD/run_logs:/workspace/SepReformer/run_logs" \
  -v "$PWD/initializations:/workspace/SepReformer/initializations" \
  -e MODEL=SepReformer_Base_Libri2Mix_16K \
  sepreformer:runtime bash scripts/train.sh --seed 0 --run-id libri_base_common_v2_s0
```

The extracted dataset mount must include its prepared manifest. Mount the
VN-SpeechMix directory instead when using a VN config. Docker excludes local data,
environments and checkpoints from the image. The Dockerfile accepts CUDA_IMAGE
and the four TORCH_* build arguments above; choose compatible images/builds.
Request only resources granted to you by the host/scheduler.

Persist runs/, initializations/, run_logs/, data and config/source snapshots.
Checkpoints live in runs/<model>/seed_<seed>/<run-id>/checkpoints/ with latest,
best and periodic milestones. Back up complete run directories using
scripts/backup_run.py. Resume explicitly with the same code/config/manifest;
see the common protocol guide before continuing an old experiment.
