# VN-SpeechMix long-run procedure

For the dedicated VN-SpeechMix baseline package and its matching PARR config, see
[SepReformer_Base_VnSpeechMix_16K](../models/SepReformer_Base_VnSpeechMix_16K/README.md).
The commands below continue to document the earlier package names.

The two existing `*_Libri2Mix_16K` Python package names are retained for compatibility.
Their default configs now select `data/VnSpeechMix/rendered/{train,valid,test}`
(18,000 / 3,000 / 5,000 mixtures). Architecture, losses, optimizer, crop policy and
training schedule are unchanged. Both models use `initializations/vnspeechmix/`.
The published split has two speakers shared by validation/test; training has no
speaker overlap with either. Report this limitation when using the published split.

## Prepare and verify

Use Python 3.10/3.11 and the existing L40 setup script. Setuptools is pinned to
80.9.0 for TensorBoard compatibility. CUDA determinism uses `:4096:8` before Python
loads CUDA. Deterministic algorithms remain in warn-only mode: adaptive average
pooling backward on CUDA can still be nondeterministic. Do not claim bitwise CUDA
reproducibility; use the same seed set for baseline and PARR.

```bash
bash scripts/setup_l40.sh
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m scripts.runtime_checks
python -m scripts.test_parr_fixes
python -m scripts.test_run_management
python -m models.SepReformer_PARR_Libri2Mix_16K.smoke_test_parr
python scripts/preflight_l40.py --require-l40
```

Require the preflight JSON to report complete success on the actual training GPU.
CPU regression checks do not substitute for this check. Full preflight reads all
78,000 WAV streams. Preserve `rendered_metadata.csv` byte-for-byte when migrating:
its SHA-256 is part of the resume contract. Old absolute paths inside that file do
not affect the training loader. Use a separate relocated copy for toolkit audits.
This is a **manifest fingerprint**, not a hash of every WAV's contents; retain the
dataset validation report and revalidate audio after transfer.

## Start and resume

```bash
python run.py --model SepReformer_Base_Libri2Mix_16K --seed 0 --run-id vn_base_s0
python run.py --model SepReformer_PARR_Libri2Mix_16K --seed 0 --run-id vn_parr_s0
```

Use the same seed for each pair (for example 0, 1, 2). `--seed` updates the experiment
and DataLoader seeds together. A timestamp with microseconds is generated if
`--run-id` is omitted. `--output-dir /persistent/runs` chooses persistent storage.
An existing training run directory is always rejected. The L40 shell wrappers
forward these arguments, e.g. `bash scripts/train_l40.sh --seed 0 --run-id vn_parr_s0`.

```text
runs/<model>/seed_0000/<run-id>/
  checkpoints/{latest.pth,best.pth,epoch_0010.pth,...}
  tensorboard/
  logs/system.log
  config.yaml
  environment.txt
  provenance.json
  source.patch
  source/...
```

Checkpoints are saved atomically. Config `engine.checkpoint_epochs` defaults to
10, 20, 50, 100, 150, 200. Milestones use the **same full payload** as latest:
model, AdamW, both schedulers, RNG/loader states, epoch, best loss, initialization
identity and run contract. No automatic scan of old model `log/` directories is
used by the CLI.

```bash
python run.py --model SepReformer_Base_Libri2Mix_16K --seed 0 \
  --run-id vn_base_s0_continued \
  --resume runs/SepReformer_Base_Libri2Mix_16K/seed_0000/vn_base_s0/checkpoints/latest.pth
```

Resume explicitly continues the optimizer/scheduler/RNG state into a **new run
directory**. Keep the same training config and seed; use `--config path/to/config.yaml`
when required. Historical best is copied if available at that resume boundary. If
resuming an older milestone after its historical best has been overwritten, the
new run retains the recorded best loss, but has no `best.pth` until it improves that
loss; evaluate an explicit archived historical best in that case. Do not silently
substitute latest for best. Preserve parent runs for TensorBoard/history and backup.

The pilot source before this migration was saved locally in
`run_logs/pilot_source_before_run_management.zip`. Existing pilot checkpoints/logs
were not moved or overwritten. Extract that archive to a separate checkout to
continue the pilot with the old code. New source fingerprints intentionally prevent
resuming old pilot checkpoints with this implementation. Start official paired
experiments from epoch zero. Commit reviewed code before an official run; this
change does not automatically create a Git commit.

## TensorBoard and provenance

Both models record time/frequency train/validation losses, main LR (after the
epoch's scheduler update), and train/validation seconds. PARR also records four
gamma values, gate mean/std/fractions below .05 and above .95, applied correction
norm divided by original feature norm, and the mean pre-clipping shared-core
gradient L2 norm over training batches.

Gate statistics aggregate all gate elements over all stages; correction ratio is
the mean of per-stage/per-example ratios `||gamma * delta|| / ||feature||` (computed
from output minus input). Diagnostics use the first `engine.diagnostic_examples`
validation examples (default four), deterministic center crops, eval mode, no
gradients and no DataLoader iterator. RNG states and model mode are restored even
on failure. Keys are saved in provenance. TensorBoard writer closes on exceptions.

Every run stores Git commit/status, pip freeze, Python/Torch/CUDA/cuDNN, visible
GPUs/driver/UUID, CUBLAS configuration, paired initialization hash, dataset manifest
hash, exact config, CLI arguments and fingerprinted source snapshots. Resume and
evaluation also record the input checkpoint SHA-256. Untracked source files are
included in snapshots even though `git diff` cannot include them.

## Evaluate a specific checkpoint

```bash
python run.py --model SepReformer_Base_Libri2Mix_16K --engine-mode test --seed 0 \
  --checkpoint runs/SepReformer_Base_Libri2Mix_16K/seed_0000/vn_base_s0/checkpoints/epoch_0050.pth
```

Alternatively `--run-id vn_base_s0` without `--checkpoint` selects that run's
`checkpoints/best.pth`. Every evaluation writes into a separate timestamped directory
and reports the checkpoint hash. Choose checkpoint using validation; reserve test
for the final comparison rather than selecting an epoch using test results.

## Backup and restore verification

At milestones, flush/stop writers or choose a quiet boundary before copying the
run to a mounted persistent volume. A run that changes during copying is rejected;
a `.partial` artifact is retained for diagnosis. Use a new destination when retrying.

```bash
python scripts/backup_run.py create \
  runs/SepReformer_Base_Libri2Mix_16K/seed_0000/vn_base_s0 \
  /persistent/backups/vn_base_s0_epoch0050.zip
python scripts/backup_run.py verify /persistent/backups/vn_base_s0_epoch0050.zip
```

The archive includes checkpoints, TensorBoard, logs, snapshots and provenance,
with per-file SHA-256 and an external `.zip.sha256`. Copy **both files** to Drive or
another persistent destination and run `verify` again after download, before
extracting. The script uses local/mounted storage; it does not upload to an account.
Also preserve the dataset and `initializations/vnspeechmix/` separately, plus parent
runs if resuming. Dataset audio and generated runs/backups are excluded from Git
and Docker build context.
