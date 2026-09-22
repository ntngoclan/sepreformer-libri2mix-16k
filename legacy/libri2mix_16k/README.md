# Libri2Mix 16 kHz: historical course-project baseline and LTRR

This isolated directory contains the code used in the supplied `SourceCode.zip`.
NoGate is now named **LTRR** (Lightweight Temporal Residual Refinement). This is
an archival import of the old experiment, not a migration to the new VN-SpeechMix
training protocol. Original trained checkpoints and audio are not in this import.

## Layout and naming

```text
legacy/libri2mix_16k/
  run.py
  prepare_scp.py
  utils/                         # original course-project utilities
  models/
    SepReformer_Base_Libri2Mix_16K/
    SepReformer_LTRR_Libri2Mix_16K/
  SOURCE_PROVENANCE.json
```

Use this directory's `run.py`, not the root project's launcher. Its private
`utils/` is required: mixing old engines with the current root utilities would
change the experiment or cause imports to fail.

Changes relative to the source archive:

- Package `SepReformer_SGLR_NoGate_Libri2Mix_16K` is renamed
  `SepReformer_LTRR_Libri2Mix_16K`; block class `SGLR` is renamed `LTRR`.
- The original `module_sglr` configuration key and `self.sglr` parameter prefix
  are deliberately retained for existing checkpoint compatibility. The config
  still specifies `use_speaker_gate: false`. No operation or weight shape is changed.
- Add `utils/functions.py` as a re-export of the original implementation in
  `utils/implements/functions.py`: the ZIP omitted this import path.
- Adapt the original SCP preparation script to the two imported models and
  this local project directory. Normalize source line endings to LF.
- Do not copy duplicate config snapshots, SGLR-full, notebooks, results, logs,
  or duplicate server-script evidence. The external original ZIP is preserved.

`SOURCE_PROVENANCE.json` records the original ZIP member, SHA-256 and imported
file hash. It distinguishes source import/naming changes from the original run.

Import verification: all 28 imported source files were checked against the ZIP
and recorded hashes. Both configs preserve the original `config` values. After
reversing the class/package rename and ignoring documentation, LTRR's Python AST
matches NoGate's original executable code. Both engines import with the isolated
utilities; both full models pass a CPU forward check on a 2048-sample input with
finite outputs. This does not constitute a full training or GPU validation.

## Historical configuration (preserved)

| Setting | Baseline | LTRR, formerly NoGate |
| --- | --- | --- |
| Data | Libri2Mix wav16k/min, clean | Same |
| Sample rate / max crop | 16000 / 64000 | Same |
| Train/validation batch | 12 | 12 |
| Encoder/decoder kernel / stride | 16 / 8 | 16 / 8 |
| Spectral frame / hop | 512 / 128 | 512 / 128 |
| AdamW LR / weight decay | 0.001 / 0.01 | Same |
| Warmup steps / clip norm | 1000 / 5 | Same |
| Scheduler start | 20 | 20 |
| Configured max_epoch | 200 | 200 |
| GPU IDs | 0,1 | 0,1 |
| DataLoader workers / pin_memory | 16 / true | 4 / false |

The old epoch loop has an exclusive upper bound, so max_epoch=200 ends at 199.
The original loader shuffles validation/test and uses random validation crops.
The old pipeline does not have the new paired initialization/run management or
full RNG/scheduler resume protections. These historical behaviors are preserved
for traceability, not recommended as the new thesis protocol.

## Preparing data and using the historical launcher

Reuse a compatible Python environment; `requirements_minimal.txt` is the original
incomplete, unpinned dependency list, not a reproducibility lock. The root project's
environment has been used for CPU verification of these imported models.

From the repository root, activate that environment first, then:

```bash
cd legacy/libri2mix_16k
python prepare_scp.py --data-root /absolute/path/to/Libri2Mix/wav16k/min
python run.py --model SepReformer_Base_Libri2Mix_16K --engine-mode train
python run.py --model SepReformer_LTRR_Libri2Mix_16K --engine-mode train
```

The preparation script writes absolute SCP paths and updates only the dataset
path by default. It expects train-100/dev/test; confirm this matches the dataset
being used. Do not invoke it unless configuring a historical rerun.

The checked-in configs retain the original server's absolute SCP path. They
cannot access your data until you prepare paths. GPU IDs 0,1 require two visible
devices; adapting to one GPU changes the historical execution setting and must
be recorded. These commands are references, not commands to start on the login node.

The old launcher accepts `--checkpoint`, but does NOT support the new `--config`,
`--run-id`, `--seed` or `--resume` flags. For current VN-SpeechMix experiments use
the packages under the root `models/`, as described in the repository guide.
