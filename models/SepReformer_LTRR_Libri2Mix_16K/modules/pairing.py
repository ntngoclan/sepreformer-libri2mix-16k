"""Share the Libri2Mix baseline epoch-zero artifact with LTRR.

The baseline's v1 artifact hashes the historical PARR routing fields as part of
its shared config. Normalize only those routing fields and remove module_ltrr
when accessing that artifact. All data/training/seed settings remain checked.
The actual LTRR config and source retain their own full run contract.
"""
import copy
import importlib
import hashlib
import json
from pathlib import Path

import torch

from utils.paired_initialization import (
    apply_paired_initialization as apply_baseline_initialization,
    paired_initialization_enabled,
)


BASELINE = "SepReformer_Base_Libri2Mix_16K"
CANDIDATE = "SepReformer_LTRR_Libri2Mix_16K"


def make_run_contract(model, config):
    """Include the baseline constructor used to regenerate epoch-zero weights."""
    from utils.run_contract import make_run_contract as make_common_contract
    contract = make_common_contract(model, config)
    root = Path(__file__).resolve().parents[3]
    baseline_dir = root / 'models' / BASELINE
    paths = [baseline_dir / 'model.py', *sorted((baseline_dir / 'modules').glob('*.py'))]
    for path in paths:
        contract['source_files'][path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_text(encoding='utf-8-sig').encode('utf-8')).hexdigest()
    contract['source_sha256'] = hashlib.sha256(json.dumps(
        contract['source_files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return contract


def baseline_artifact_config(config):
    shared = copy.deepcopy(config)
    shared["model"].pop("module_ltrr")
    settings = shared["paired_initialization"]
    if (settings["baseline_model"] != BASELINE
            or settings["candidate_model"] != CANDIDATE
            or settings["allowed_extra_prefix"] != "ltrr."):
        raise ValueError("LTRR requires the declared Libri2Mix baseline and ltrr. parameter prefix.")
    # Compatibility with already created baseline initialization, not a PARR model.
    settings["candidate_model"] = "SepReformer_PARR_Libri2Mix_16K"
    settings["allowed_extra_prefix"] = "separator.parr."
    return shared


def apply_paired_initialization(model, model_name, config, workspace_root):
    if not paired_initialization_enabled(config):
        return None
    if model_name != CANDIDATE:
        raise ValueError(f"Unexpected model for LTRR initialization: {model_name}")
    shared = baseline_artifact_config(config)
    baseline_cls = importlib.import_module(f"models.{BASELINE}.model").Model
    # Construct a CPU reference without consuming the candidate's random stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(shared["experiment"]["seed"]))
        baseline = baseline_cls(**shared["model"])
    metadata = apply_baseline_initialization(
        baseline, BASELINE, shared, workspace_root)
    state = baseline.state_dict()
    target = model.state_dict()
    extra = set(target) - set(state)
    if not extra or any(not key.startswith("ltrr.") for key in extra):
        raise RuntimeError("Only ltrr.* tensors may be added to the baseline.")
    if any(key not in target or target[key].shape != value.shape
           for key, value in state.items()):
        raise RuntimeError("LTRR backbone does not match baseline tensor names/shapes.")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or set(result.missing_keys) != extra:
        raise RuntimeError("Unexpected state mismatch while copying the baseline.")
    current = model.state_dict()
    if any(not torch.equal(current[key].detach().cpu(), value.cpu())
           for key, value in state.items()):
        raise RuntimeError("Backbone tensors were not copied exactly.")
    # The old NoGate alpha is 0.05, not PARR's zero-initialized gamma.
    expected_alpha = model.ltrr.alpha.new_tensor(config["model"]["module_ltrr"]["residual_scale_init"])
    if not torch.equal(model.ltrr.alpha.detach(), expected_alpha):
        raise RuntimeError("LTRR residual alpha differs from its configured initialization.")
    metadata = dict(metadata, model_only_tensor_count=len(extra))
    model.paired_initialization_metadata = metadata
    return metadata
