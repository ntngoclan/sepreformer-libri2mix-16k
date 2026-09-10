# Paired epoch-0 initialization

Training either `SepReformer_Base_Libri2Mix_16K` or
`SepReformer_PARR_Libri2Mix_16K` automatically creates or loads the same
seed-specific baseline state in this directory. All shared tensors are copied
strictly into both models; only `separator.parr.*` remains independently
initialized in PARR.

Generated `.pth` files are intentionally ignored by Git because they are large.
Their deterministic model-state SHA-256, file SHA-256, and the
shared-configuration SHA-256 are recorded for audit (the stable model-state and
configuration hashes are stored in every training checkpoint). Keep this
directory together with the run checkpoints. If a shared
configuration or seed changes, a different compatible file must be generated;
the loader fails instead of silently accepting an incompatible file.

Epoch-0 sharing and training resume have separate identities. The shared hash
intentionally excludes PARR; every new training checkpoint also stores a
`run_contract` covering the full configuration (including dilation/dropout),
architecture, source fingerprints and pipeline revision. Resume rejects a
missing or changed contract. Evaluation may change evaluation options but still
requires the original architecture and source contract.

After the length-aware loss/padding fixes, start new baseline/PARR runs. Existing
compatible epoch-0 weights can be reused; older training checkpoints must not
silently continue under the changed loss/validation semantics.
