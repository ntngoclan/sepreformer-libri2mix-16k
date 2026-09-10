# NVIDIA L40 deployment

This setup targets one visible NVIDIA L40, Python 3.10, PyTorch 2.1.2 and
CUDA 12.1. The architecture, objectives and training hyperparameters are not
changed by the deployment files.

## Persistent files

Keep the following outside ephemeral container storage:

- `data/Libri2Mix.zip`, or extracted `data/Libri2Mix/`;
- `initializations/` (the common epoch-0 baseline state used by both runs);
- `models/SepReformer_Base_Libri2Mix_16K/log/`;
- `models/SepReformer_PARR_Libri2Mix_16K/log/`;
- `run_logs/` and the evaluation output directories.

Training writes two bounded checkpoint files:

- `scratch_weights/latest.pth`: model, optimizer and schedulers from the most
  recently completed epoch; this is selected when training resumes;
- `scratch_weights/best.pth`: validation-best state; this is selected by test
  mode for the final held-out evaluation.

Both files also contain Python/NumPy/PyTorch/CUDA RNG states and DataLoader
generator states. Resume is at a completed epoch boundary, not mid-epoch; keep
the same software, device topology and loader configuration. Nonpersistent
workers are required. CUDA numerical nondeterminism can still prevent bitwise
reproduction. Old checkpoints without RNG states cannot provide this guarantee.

Both files are written atomically. Legacy `epoch.XXXX.pth` checkpoints remain
selectable for evaluation when neither of the new files exists; resuming a
controlled paired run requires the initialization metadata described below.

For the controlled one-seed comparison, both train commands automatically use
the same baseline epoch-0 initialization. Every common state-dict tensor must
match exactly; only `separator.parr.*` may be additional in PARR, whose residual
stage scales must start at exactly zero. The initialization file and shared
configuration are fingerprinted in each checkpoint. A legacy scratch checkpoint
without these fingerprints is deliberately rejected, as is a checkpoint in
`log/pretrain_weights`, because either would invalidate this from-scratch pair.

## Native setup

Install the required OS library first on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libsndfile1 python3-venv
```

Then run:

```bash
bash scripts/setup_l40.sh
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/preflight_l40.py --require-l40
```

Run preflight on the actual GPU before starting a long training run. It verifies
the CUDA kernel, shared configuration, PARR identity and gradient invariants,
and decodes every mixture and source in all three splits (59,700 WAV streams).
It runs TWO forward/loss/backward/AdamW updates per model at batch 2 x 64,000
samples, using natural gamma=0 initialization, and saves the real optimizer and
scheduler states. A separate stochastic test compares resumed vs uninterrupted
training with 0 and 2 workers. It also checks inference equivalence without
auxiliary heads, evaluator CSV/JSON output and training peak allocated VRAM.
This is not an epoch of training, a convergence test or a throughput benchmark.
`--quick-data`, `--skip-data`, `--skip-backward`, or a reduced workload yield only
`status: partial`; exit code zero alone is insufficient for the full check.
Its machine-readable report is written to
`run_logs/preflight_l40_report.json`.

Train PARR inside `tmux`:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/train_l40.sh
```

Train the controlled baseline by changing only the model package:

```bash
source .venv/bin/activate
MODEL=SepReformer_Base_Libri2Mix_16K CUDA_VISIBLE_DEVICES=0 bash scripts/train_l40.sh
```

After training, test the validation-best checkpoint:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/test_l40.sh
MODEL=SepReformer_Base_Libri2Mix_16K CUDA_VISIBLE_DEVICES=0 bash scripts/test_l40.sh
```

## Docker setup

The image deliberately excludes the 14 GB archive and all checkpoints. Build it
from the repository root:

```bash
docker build -f Dockerfile.l40 -t sepreformer:l40 .
```

Prepare persistent directories on the host:

```bash
mkdir -p persistent/initializations persistent/parr_log persistent/run_logs
```

Run preflight with the ZIP mounted read-only:

```bash
docker run --rm --gpus 'device=0' --shm-size=8g \
  -v "$PWD/data/Libri2Mix.zip:/workspace/SepReformer/data/Libri2Mix.zip:ro" \
  -v "$PWD/persistent/initializations:/workspace/SepReformer/initializations" \
  -v "$PWD/persistent/parr_log:/workspace/SepReformer/models/SepReformer_PARR_Libri2Mix_16K/log" \
  -v "$PWD/persistent/run_logs:/workspace/SepReformer/run_logs" \
  sepreformer:l40 \
  python3 scripts/preflight_l40.py --require-l40
```

Use an extracted dataset for substantially better long-run I/O. Mount its
`Libri2Mix` directory at `/workspace/SepReformer/data/Libri2Mix` instead of
mounting the ZIP.

Train with the same persistent mounts:

```bash
docker run --rm --gpus 'device=0' --shm-size=8g \
  -v "$PWD/data/Libri2Mix:/workspace/SepReformer/data/Libri2Mix:ro" \
  -v "$PWD/persistent/initializations:/workspace/SepReformer/initializations" \
  -v "$PWD/persistent/parr_log:/workspace/SepReformer/models/SepReformer_PARR_Libri2Mix_16K/log" \
  -v "$PWD/persistent/run_logs:/workspace/SepReformer/run_logs" \
  sepreformer:l40 \
  bash scripts/train_l40.sh
```

Do not delete the instance until `latest.pth`, `best.pth`, the training log and
`run_logs/preflight_l40_report.json` have been copied to durable storage.

## First-run checklist

1. `nvidia-smi` identifies one L40 and shows no unwanted process.
2. At least 60 GiB of persistent disk is free; more is needed if both models
   and extracted audio are retained.
3. Preflight exits with status zero AND its report has `status: passed`.
4. The report says `dataset.train.num_mixtures = 13900`, and dev/test both equal
   3000.
5. Begin with the configured batch size 2. Do not increase only one model's
   batch size when making the scientific comparison.
6. Verify `latest.pth` and `best.pth` after the first completed epoch.
7. Back up the whole model `log/` directory and `initializations/` regularly.
