# Shared 16 kHz protocol, revision 2

Use the root `run.py` with these four packages for NEW from-scratch experiments:

| Dataset | Baseline | LTRR |
| --- | --- | --- |
| Libri2Mix | SepReformer_Base_Libri2Mix_16K | SepReformer_LTRR_Libri2Mix_16K |
| VN-SpeechMix | SepReformer_Base_VnSpeechMix_16K | SepReformer_LTRR_VnSpeechMix_16K |

Their `configs.yaml` files share all training settings except dataset identity
and paired-initialization routing. LTRR adds only `model.module_ltrr` to the
model config. `python -B -m scripts.test_libri2mix_protocol` checks this invariant
and the same pairing, gradients and exact CPU resume checks used for VN LTRR.

## Frozen common settings

- Mono 16 kHz, two clean sources; 64,000-sample random training crops.
- Deterministic center crops for validation; full utterances for test, with
  at most seven trailing samples removed for stride alignment.
- Train batch 2; validation/test batch 1. 12 loader workers, pin_memory enabled.
- Four separator stages, feature width 128, encoder width 256, eight heads.
- Waveform encoder AND decoder kernel 32, stride 8 (2 ms / 0.5 ms).
- Spectral auxiliary loss frame 1024 / hop 256 (64 ms / 16 ms).
- AdamW LR 0.001, weight decay 0.01; warmup 1,000 optimizer updates; clip norm 5.
- ReduceLROnPlateau: min mode, factor 0.8, patience 2, min_lr 1e-10;
  **absolute** threshold 1e-4. It first observes validation after epoch 51
  (`epoch > 50`), subject to warmup completion.
- 200 epochs inclusive; best checkpoint selected by validation time loss.
- Auxiliary loss weight 0.4 through epoch 100, then multiplied by 0.8 per
  five-epoch interval, starting at epoch 101. This schedule lives in engine.py.
- No intermediate test-set evaluation for model selection.
- Same seed and identical backbone epoch-zero tensors within each dataset pair.
- LTRR retains original NoGate operations and alpha initialization 0.05.

This is the chosen experimental protocol, not an empirically optimal config and
not a reproduction of the paper's 8 kHz experiment. Same epochs on different
dataset sizes do not imply the same number of updates or compute budget.

## Dataset-specific settings

| Item | Libri2Mix | VN-SpeechMix |
| --- | --- | --- |
| Extracted root | data/Libri2Mix/wav16k/min | data/VnSpeechMix/rendered |
| ZIP fallback | data/Libri2Mix.zip | data/VnSpeechMix/rendered.zip |
| Root inside ZIP | Libri2Mix/wav16k/min | rendered |
| Train / valid / test folders | train-100 / dev / test | train / valid / test |
| Expected mixture counts | 13900 / 3000 / 3000 | 18000 / 3000 / 5000 |
| Source folders | mix_clean, s1, s2 | mix_clean, s1, s2 |
| Initialization directory | initializations/libri2mix_common_v2 | initializations/vnspeechmix_common_v2 |

Use matching filenames in all three source folders. The Libri2Mix counts refer
to this project's train-100 clean/min protocol, not every LibriMix variant.
Do not silently substitute train-360, max-length mixtures or noisy mixtures.
If a ZIP has a different prefix, set `dataset.archive_root` in both configs.
Extracting the WAVs is recommended for long runs to reduce ZIP I/O overhead.

## Server preparation and checks

Run from the repository root in the existing Python environment / GPU allocation.
Place Libri2Mix at the paths above; cloning Git does not download the dataset.

```bash
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -B -m scripts.test_libri2mix_protocol

# Checks all triplet keys, counts, mono/sample rates and lengths; hashes every WAV.
# Run once after uploading/extracting the dataset. Can take time: use tmux.
python -B scripts/prepare_libri2mix_manifest.py

# Existing preflight now supports both datasets; sampled WAV checks by default.
python -B scripts/preflight_vnspeechmix_baseline.py --root . --model SepReformer_Base_Libri2Mix_16K --device cuda --samples 64000 --steps 2 --workers 12 --report run_logs/libri_base_gpu.json
python -B scripts/preflight_vnspeechmix_baseline.py --root . --model SepReformer_LTRR_Libri2Mix_16K --device cuda --samples 64000 --steps 2 --workers 12 --report run_logs/libri_ltrr_gpu.json
```

The manifest is `data/Libri2Mix/manifest_16k_min.csv`, excluded from Git. Its
rows contain split/key/length/sample rate and SHA-256 for all three WAV files.
The run contract hashes that manifest; regenerate it after an intentional data
change. Hashing proves content identity, not speaker-disjointness or absence of
audio corruption beyond header/ZIP checks. Preflight `--full-audio` reads every
WAV and checks finiteness and mixture/source sums.

For a separate two-epoch pilot, create configs without editing production YAML:

```bash
python - <<'PY'
from pathlib import Path
import yaml
Path('run_logs').mkdir(exist_ok=True)
for variant in ('Base', 'LTRR'):
    name = f'SepReformer_{variant}_Libri2Mix_16K'
    doc = yaml.safe_load(Path(f'models/{name}/configs.yaml').read_text())
    cfg = doc['config']
    cfg['engine']['max_epoch'] = 2
    cfg['engine']['checkpoint_epochs'] = [1, 2]
    cfg['paired_initialization']['directory'] = 'initializations/libri2mix_common_v2_pilot'
    with Path(f'run_logs/libri_{variant.lower()}_pilot.yaml').open('x') as f:
        yaml.safe_dump(doc, f, sort_keys=False)
PY
MODEL=SepReformer_Base_Libri2Mix_16K bash scripts/train.sh --config run_logs/libri_base_pilot.yaml --seed 0 --run-id libri_base_v2_pilot_s0
MODEL=SepReformer_LTRR_Libri2Mix_16K bash scripts/train.sh --config run_logs/libri_ltrr_pilot.yaml --seed 0 --run-id libri_ltrr_v2_pilot_s0
```

After pilot verification, start production runs separately (do not run both
simultaneously on one GPU unless you have verified capacity):

```bash
MODEL=SepReformer_Base_Libri2Mix_16K bash scripts/train.sh --seed 0 --run-id libri_base_common_v2_s0
MODEL=SepReformer_LTRR_Libri2Mix_16K bash scripts/train.sh --seed 0 --run-id libri_ltrr_common_v2_s0
```

## Compatibility and historical results

Changing scheduler semantics changes the full config hash. The new initialization
directories prevent collisions with v1 artifacts. **Start all four new official
runs from epoch zero**; old checkpoints are not interchangeable with this
protocol. Preserve previous run config/source snapshots to continue old runs.
For resume of a v2 run, retain exactly its config, manifest and code version.

`legacy/libri2mix_16k/` preserves the actual course-project config (batch 12,
kernel 16, STFT 512/128, scheduling after 20); historical scores must continue to
be attributed to that protocol. The author's WSJ0 reference stays unchanged.

The optional PARR default config is aligned to the Libri2Mix baseline so its
existing shared-initialization tests remain valid. For PARR on VN-SpeechMix use
`models/SepReformer_PARR_Libri2Mix_16K/configs_vnspeechmix.yaml` with the dedicated
VN baseline. PARR routing labels in baseline configs identify the existing
initialization format; LTRR's local adapter shares those backbone weights and
does not construct a PARR model.

## Local verification

The 8 Libri2Mix protocol tests and 38 existing LTRR/runtime/PARR/run-management
tests passed on CPU. These include original NoGate gradient equivalence, shared
initialization in both run orders, exact next-update resume, negative plateau
scheduling, WAV-directory/ZIP equivalence, and CLI preflight against synthetic
WAV triplets with a real optimizer step and checkpoint roundtrip. The two-worker
resume test was rerun through `python -m unittest` because Windows workers cannot
spawn an entry point supplied through stdin. The original 28 imported archive
files retain their recorded hashes. Full Libri2Mix audio and CUDA were unavailable
locally; actual-dataset GPU preflight and the pilot remain server-side checks.
