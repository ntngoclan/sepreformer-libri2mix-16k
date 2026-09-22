# SepReformer-LTRR, Libri2Mix 16 kHz

LTRR is the original NoGate refinement, placed in the corrected baseline pipeline.
Its model and backbone match the VN-SpeechMix LTRR implementation; only the dataset
and local baseline identity differ. Its common config matches the Libri2Mix baseline.

See [shared protocol, preparation, GPU checks and commands](../../docs/COMMON_16K_PROTOCOL.md).
Run `python -B -m scripts.test_libri2mix_protocol` for paired initialization,
original NoGate gradient equivalence, length handling and exact CPU resume tests.

Use the root `run.py --model SepReformer_LTRR_Libri2Mix_16K`, not the legacy launcher.
No old trained weights are loaded. The historical config/checkpoint naming scheme
is preserved separately under `legacy/libri2mix_16k/`.
