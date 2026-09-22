# SepReformer-LTRR on VN-SpeechMix, 16 kHz

The default config now follows [shared protocol v2](../../docs/COMMON_16K_PROTOCOL.md)
with the Libri2Mix experiments. The absolute plateau threshold and new initialization
directory require fresh official runs for both baseline and LTRR.

LTRR (Lightweight Temporal Residual Refinement) is the original course-project
NoGate design, renamed and placed in the corrected VN-SpeechMix pipeline.
It is not PARR with its gate disabled, and does not load old trained weights.

## Architecture

One shared refinement block processes final separator features for each speaker
after the reconstruction decoder and before OutputLayer. Auxiliary outputs are
unchanged. The NoGate block is copied from the supplied SourceCode.zip; its
unused speaker-guided gate path is removed, preserving all active operations:

```text
Z = bottleneck(Y)
H = Z + Dropout(ConvU(Z) * DenseDilatedMemory(ConvV(Z)))
Y_out = Y + alpha * projection(H)
```

- Input/output: 128 channels; bottleneck: 64.
- Conv-U depthwise kernel: 3; memory dilations: [1, 2, 4, 8].
- Dropout: 0.05; ChannelLayerNorm epsilon: 1e-8.
- Learnable scalar alpha starts at 0.05, exactly as old NoGate.
- There is no speaker-guided gate. Multiplication of the two temporal branches remains.
- Extra parameters: 89,031. With the current backbone: 14,805,191 total versus
  14,716,160 baseline (about 0.605% more).
- Because alpha is nonzero, initial model outputs need not equal baseline outputs.

## Shared experimental configuration

All common settings match `SepReformer_Base_VnSpeechMix_16K/configs.yaml`:
dataset/splits, 16 kHz, crops, batch size, loss, optimizer, warmup, scheduler,
epoch budget, evaluation and checkpoint selection. Only project description,
candidate routing and `model.module_ltrr` differ. These are the selected thesis
settings, not empirically demonstrated optimal hyperparameters.

The local pairing adapter copies every backbone tensor from the same epoch-zero
artifact used by the existing baseline. It works whether baseline or LTRR runs
first. Old PARR routing fields are normalized ONLY when validating that legacy
artifact; they do not instantiate a PARR network. The actual LTRR config/source
are recorded in its own run contract. Data/training mismatches still fail.

LTRR uses the same backbone code as the baseline. Protocol v2 changes the common
config for both models; old pilot artifacts cannot be reused as v2 initialization
or resume checkpoints. Keep the same protocol and initialization artifact within
each new baseline/LTRR pair.

## Checks and GPU run

From the repository root in the installed environment:

```bash
python -B -m scripts.test_ltrr
mkdir -p run_logs
python -B scripts/preflight_vnspeechmix_baseline.py --root . --model SepReformer_LTRR_VnSpeechMix_16K --device cuda --samples 64000 --steps 2 --workers 12 --report run_logs/ltrr_gpu_check.json
```

The smoke check is not a full epoch or evidence of convergence. CPU tests verify
original NoGate forward/input gradients/parameter gradients in train and eval,
backbone pairing in both run orders, auxiliary invariance, true-length handling,
and exact next-update checkpoint resume on CPU. CUDA reproducibility is subject
to the same nondeterministic-operation limitations as baseline.

Production training (from scratch), after the GPU pilot succeeds:

```bash
MODEL=SepReformer_LTRR_VnSpeechMix_16K bash scripts/train.sh --seed 0 --run-id vn_ltrr_s0
```

Results: `runs/SepReformer_LTRR_VnSpeechMix_16K/seed_0000/vn_ltrr_s0/`.

New two-epoch pilot, using the same v2 settings as the baseline pilot:

```bash
python - <<'PY'
from pathlib import Path
import yaml
path = Path('run_logs/ltrr_pilot.yaml')
path.parent.mkdir(parents=True, exist_ok=True)
c = yaml.safe_load(Path('models/SepReformer_LTRR_VnSpeechMix_16K/configs.yaml').read_text())
c['config']['engine']['max_epoch'] = 2
c['config']['engine']['checkpoint_epochs'] = [1, 2]
c['config']['paired_initialization']['directory'] = 'initializations/vnspeechmix_common_v2_pilot'
with path.open('x') as f:
    yaml.safe_dump(c, f, sort_keys=False)
PY
MODEL=SepReformer_LTRR_VnSpeechMix_16K bash scripts/train.sh --config run_logs/ltrr_pilot.yaml --seed 0 --run-id vn_ltrr_pilot_s0
```

Resume the official run into a new output directory with unchanged configuration:

```bash
python run.py --model SepReformer_LTRR_VnSpeechMix_16K --seed 0 --run-id vn_ltrr_s0_resume --resume runs/SepReformer_LTRR_VnSpeechMix_16K/seed_0000/vn_ltrr_s0/checkpoints/latest.pth
```

Evaluate the validation-best checkpoint after the final protocol is frozen:

```bash
python run.py --model SepReformer_LTRR_VnSpeechMix_16K --engine-mode test --seed 0 --run-id vn_ltrr_s0
```

For pilot resume/evaluation include the same `--config run_logs/ltrr_pilot.yaml`.
Never resume a pilot checkpoint with the 200-epoch production configuration.
