# SepReformer-PARR for Libri2Mix 16 kHz

Default training now uses [common protocol v2](../../docs/COMMON_16K_PROTOCOL.md).
Prepare the Libri2Mix manifest before starting a new English run. For Vietnamese
data, explicitly select the config below.

## VN-SpeechMix configuration

`configs_vnspeechmix.yaml` pairs this PARR implementation with
`SepReformer_Base_VnSpeechMix_16K`. It is the configuration previously kept as
`configs_parr.yaml` in the VN-SpeechMix baseline directory.
Run from the repository root:

```bash
python run.py --model SepReformer_PARR_Libri2Mix_16K --config models/SepReformer_PARR_Libri2Mix_16K/configs_vnspeechmix.yaml --seed 0 --run-id vn_parr_s0
```

Use the same `--config` for resume and evaluation. Shared initialization is saved
to `initializations/vnspeechmix_common_v2/sepreformer_base_vnspeechmix_16k_seed_0000.pth`
for seed 0. This optional configuration runs PARR, not LTRR; baseline training
continues to use its own `configs.yaml`.

## Architecture

This package is the Libri2Mix 16 kHz SepReformer baseline with one architectural
change: Progressive Adaptive Residual Refinement (PARR) is applied after each of
the four reconstruction decoder stages. No PARR block is applied at the
bottleneck.

For decoder stage `r`, the implementation is:

```text
N_r       = ChannelLayerNorm_r(H_r)
Z_r       = ChannelLayerNorm(PReLU(Conv1x1(N_r)))
U_r       = ConvU_theta_u(Z_r)
V_r       = ConvU_theta_v(Z_r)
M_r       = DenseDilatedFSMN_theta(V_r)
R_r       = ChannelLayerNorm(U_r * M_r)
A_r       = sigmoid(Conv1x1_theta(concat(Z_r, R_r)))
Delta_r   = Conv1x1_theta(A_r * R_r)
H'_r      = H_r + gamma_r * Delta_r
```

`A_r` has shape `[B*J, 64, T_r]`, so gating is channel-temporal. The temporal
core parameters are shared by all stages and speaker branches. Stage
normalizations and `gamma_r` are stage-specific. Every `gamma_r` is initialized
to zero, making PARR an exact identity mapping at initialization. The controller
weights and bias are also zero-initialized, giving a neutral initial gate of
`0.5`.

The two Conv-U branches have separate weights inside the shared core. The
output projection has a bias, so the gate modulates the input-dependent
correction; it is not a calibrated error probability or a complete off switch.

The unchanged SepReformer training objectives are used: time-domain PIT SI-SNR
for the final output and STFT-magnitude PIT supervision for four auxiliary
outputs. PARR does not add cross-scale evidence, stage-consistent PIT, mixture
consistency, MR-STFT, or early-exit logic.

## Data and training

Expected archive layout:

```text
data/Libri2Mix.zip
└── Libri2Mix/wav16k/min/{train-100,dev,test}/{mix_clean,s1,s2}
```

The loader can read the archive directly, but extracting it to
`data/Libri2Mix/wav16k/min` is faster.

```bash
python run.py --model SepReformer_PARR_Libri2Mix_16K --engine-mode train
```

New checkpoints are stored under `runs/<model>/seed_<seed>/<run-id>/checkpoints/`.
Use explicit `--resume path/to/latest.pth` with the same config to restore model,
optimizer, scheduler and RNG state. The launcher does not automatically resume
from old `log/scratch_weights` or load `log/pretrain_weights`. A full run contract
records config, architecture, source fingerprints and pipeline revision;
incompatible checkpoints are rejected. Start protocol v2 from epoch zero.

The length-aware spectral loss and validation policy changed in this revision.
Start fresh baseline/PARR runs together; see
[fix details and verification](../../docs/PARR_FIXES_2026_09_10.md).

Do not use the author's WSJ0-2Mix 8 kHz checkpoint as the reported Libri2Mix
16 kHz baseline. Its waveform front-end has different dimensions, so it can only
serve as a documented partial-transfer initialization and is not an equivalent
training condition.

## Verification

```bash
python -m models.SepReformer_PARR_Libri2Mix_16K.smoke_test_parr
python -m scripts.runtime_checks
python -m scripts.test_parr_fixes
```

The smoke test checks dimensions, neutral controller initialization, exact
identity at `gamma=0`, and gradient flow after enabling a residual scale. It
requires the dependencies in the repository `requirements.txt`, including
PyTorch.

Run the complete held-out evaluation after training:

```bash
python run.py --model SepReformer_PARR_Libri2Mix_16K --engine-mode test
```

Results are written under `evaluation/checkpoint_epoch_XXXX/` as an
utterance-level CSV and a JSON summary containing bootstrap 95% confidence
intervals. The suite includes SI-SNR/i, BSS-Eval SDR/SIR/SAR and improvements,
scale-dependent SNR/i, mixture-consistency error, WB-PESQ, STOI, ESTOI, latency,
RTF, peak VRAM, parameter count and MAC estimates.
All quality metrics reuse the speaker assignment selected by SI-SNR.

For a valid baseline comparison, use the matching
`SepReformer_Base_Libri2Mix_16K` package with the same data split, crop length,
batch size, optimizer, scheduler, maximum epoch, seed policy, and evaluation
code. Report multiple seeds when resources permit.
