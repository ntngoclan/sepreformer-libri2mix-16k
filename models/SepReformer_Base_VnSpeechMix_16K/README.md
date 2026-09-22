# SepReformer-B on VN-SpeechMix (16 kHz)

The default config uses [common 16 kHz protocol v2](../../docs/COMMON_16K_PROTOCOL.md).
Start fresh runs with its absolute plateau threshold and new initialization directory;
previous pilot/checkpoint configs are not interchangeable with this revision.

Dedicated baseline copied from `SepReformer_Base_Libri2Mix_16K`, with the same
architecture and training recipe. This package has its own model identity,
configuration, run directory and paired initialization filename.

Data: `data/VnSpeechMix/rendered/{train,valid,test}/{mix_clean,s1,s2}`.
Input: mono 16 kHz. Training crops: up to 64,000 samples (4 seconds).
Dataset audio is not included in Git. Transfer the rendered directory and
`rendered_metadata.csv` to the location above on the GPU machine.

For the pinned CUDA 12.1 environment on Linux with Python 3.10:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python -B scripts/preflight_vnspeechmix_baseline.py --root . --device cuda --samples 64000 --steps 2 --workers 12 --report vn_gpu_check.json
```

This baseline-specific check uses real audio, the configured model/loss/batch,
two optimizer updates and a checkpoint serialization roundtrip. It is not two
epochs or a full production-engine test. Add `--full-audio` for exhaustive audio
validation; otherwise all file names are checked but waveforms are sampled.
Inspect the report's package availability and limitations as well as its status.

To run a separate two-epoch pilot through the production engine:

```bash
python -c "import yaml; p='models/SepReformer_Base_VnSpeechMix_16K/configs.yaml'; c=yaml.safe_load(open(p)); c['config']['engine']['max_epoch']=2; c['config']['engine']['checkpoint_epochs']=[1,2]; c['config']['paired_initialization']['directory']='initializations/vnspeechmix_common_v2_pilot'; yaml.safe_dump(c,open('vn_pilot.yaml','x'),sort_keys=False)"
python run.py --model SepReformer_Base_VnSpeechMix_16K --config vn_pilot.yaml --seed 0 --run-id vn_base_pilot_s0
```

Keep pilot initialization separate because its config fingerprint differs.
The official run below starts fresh with the original 200-epoch configuration;
do not resume the pilot checkpoint into that configuration.

Run from the repository root in the CUDA training environment:

```bash
python run.py --model SepReformer_Base_VnSpeechMix_16K --seed 0 --run-id vn_base_s0
```

Results: `runs/SepReformer_Base_VnSpeechMix_16K/seed_0000/vn_base_s0/`.
Evaluate the validation-best checkpoint:

```bash
python run.py --model SepReformer_Base_VnSpeechMix_16K --engine-mode test --seed 0 --run-id vn_base_s0
```

Resume into a new run:

```bash
python run.py --model SepReformer_Base_VnSpeechMix_16K --seed 0 --run-id vn_base_s0_resume \
  --resume runs/SepReformer_Base_VnSpeechMix_16K/seed_0000/vn_base_s0/checkpoints/latest.pth
```

Start this package from epoch zero: checkpoints from the older package have a
different model/source identity and are incompatible with strict resume.

The commands above load `configs.yaml` by default. No separate PARR configuration
is needed to train this baseline. The paired-initialization settings in
`configs.yaml` still reference the existing PARR candidate; they do not add PARR
to the baseline architecture. The new [LTRR package](../SepReformer_LTRR_VnSpeechMix_16K/README.md)
reuses this baseline's initialization through its own compatibility adapter;
the baseline config and training source do not need to change.

The optional configuration for running PARR with this baseline now lives in
the PARR package: [configs_vnspeechmix.yaml](../SepReformer_PARR_Libri2Mix_16K/configs_vnspeechmix.yaml).
See that package's [instructions](../SepReformer_PARR_Libri2Mix_16K/README.md).

See [run management](../../docs/VNSPEECHMIX_RUNS.md) for checkpoints, diagnostics,
provenance and backup. The preflight script above supports both VN-SpeechMix and
Libri2Mix baseline/LTRR packages; run it on the actual training GPU.
