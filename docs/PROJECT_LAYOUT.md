# Project layout after cleanup

## Active thesis experiments

- `models/SepReformer_Base_Libri2Mix_16K/` and `SepReformer_LTRR_Libri2Mix_16K/`:
  NEW English experiments using the [shared 16 kHz protocol](COMMON_16K_PROTOCOL.md).

- `models/SepReformer_Base_VnSpeechMix_16K/`: current VN-SpeechMix baseline.
- `models/SepReformer_LTRR_VnSpeechMix_16K/`: old NoGate architecture in the
  corrected, shared VN-SpeechMix pipeline.
- `scripts/setup_env.sh`, `train.sh`, `evaluate.sh`: deployment wrappers.
- `scripts/preflight_vnspeechmix_baseline.py`: baseline/LTRR checks.
- `scripts/test_ltrr.py`, `scripts/fixtures/legacy_nogate.py`: reference equivalence
  tests; the frozen fixture is required, not a redundant model copy.

## Historical English experiments

`legacy/libri2mix_16k/` contains baseline and LTRR (formerly NoGate) imported from
the course-project SourceCode.zip, including their old config and private utils.
Use its own README and launcher. Do not confuse these with the root package
`SepReformer_Base_Libri2Mix_16K`, whose default config now targets Libri2Mix
under the new shared protocol. Historical results still belong to the legacy config.

## Files deliberately retained

| Area | Why retained |
| --- | --- |
| Root Base_Libri2Mix and PARR packages | Referenced by existing tests/preflight and optional PARR comparison |
| Base_WSJ0 and upstream data-preparation scripts | Original author's reference implementation and supported entry points |
| `utils/` | Shared runtime dependencies; removing them breaks experiments |
| `docs/` and audit probe | Historical rationale and evidence, not alternate training configs |
| `initializations/`, `run_logs/`, checkpoints/backups | Reproducibility and resume evidence; not disposable cache |
| Dataset ZIP, source and rendered metadata | Data assets, excluded from Git; not deleted in cleanup |
| `.venv-runchecks/` | Working CPU verification environment |
| LICENSE and upstream README | Attribution and licensing |

## Removed local artifacts

- `.venv-audit/`: broken environment whose base interpreter is absent
  (1,687,015,683 bytes of files before removal).
- `models/SepReformer_LTRR_VnSpeechMix_16K.zip`: every file matched the live package
  byte-for-byte; retain the editable folder instead (27,595 bytes).
- Generated Python `__pycache__` directories under root/models/utils/scripts.

Removed paths were recorded in ignored `run_logs/cleanup_legacy_import.json`.
No dataset, checkpoint, experiment result, or original external SourceCode.zip
was deleted. No trained-model behavior or VN baseline fingerprint changed.
