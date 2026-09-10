"""Deterministic, auditable initialization shared by baseline and PARR."""

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path

import torch
from loguru import logger


FORMAT_VERSION = 1
KIND = "sepreformer_paired_initialization"


def paired_initialization_enabled(config):
    return bool(config.get("paired_initialization", {}).get("enabled", False))


def _settings(config):
    settings = config.get("paired_initialization", {})
    required = ("baseline_model", "candidate_model", "directory", "allowed_extra_prefix")
    missing = [key for key in required if not settings.get(key)]
    if missing:
        raise RuntimeError(f"Paired initialization is missing settings: {missing}")
    return settings


def _shared_config(config):
    shared = copy.deepcopy(config)
    shared["model"]["module_separator"].pop("parr", None)
    return shared


def _config_sha256(config):
    encoded = json.dumps(
        _shared_config(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_dict_sha256(state_dict):
    """Hash tensor names, metadata, and raw values independent of torch.save bytes."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        header = json.dumps(
            [key, str(tensor.dtype), list(tensor.shape)],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _initialization_path(config, workspace_root):
    settings = _settings(config)
    directory = Path(settings["directory"])
    if not directory.is_absolute():
        directory = Path(workspace_root) / directory
    seed = int(config.get("experiment", {}).get("seed", 0))
    stem = settings["baseline_model"].lower()
    return directory / f"{stem}_seed_{seed:04d}.pth"


def _baseline_state_from_seed(config, current_model_name):
    settings = _settings(config)
    seed = int(config.get("experiment", {}).get("seed", 0))
    if current_model_name == settings["baseline_model"]:
        raise RuntimeError("Internal error: an existing baseline model should provide its own state.")

    baseline_module = importlib.import_module(f"models.{settings['baseline_model']}.model")
    baseline_config = copy.deepcopy(config["model"])
    baseline_config["module_separator"].pop("parr", None)
    # Constructing the helper model must not alter the candidate model's RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        baseline = baseline_module.Model(**baseline_config)
    return baseline.state_dict()


def _cpu_state_dict(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _save_initialization(path, state_dict, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = _cpu_state_dict(state_dict)
    payload = {
        "format_version": FORMAT_VERSION,
        "kind": KIND,
        "seed": int(config.get("experiment", {}).get("seed", 0)),
        "baseline_model": _settings(config)["baseline_model"],
        "shared_config_sha256": _config_sha256(config),
        "model_state_sha256": _state_dict_sha256(cpu_state),
        "torch_version": torch.__version__,
        "model_state_dict": cpu_state,
    }
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _validate_payload(payload, config, path):
    expected = {
        "format_version": FORMAT_VERSION,
        "kind": KIND,
        "seed": int(config.get("experiment", {}).get("seed", 0)),
        "baseline_model": _settings(config)["baseline_model"],
        "shared_config_sha256": _config_sha256(config),
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Paired initialization {path} is incompatible with this run: {mismatches}. "
            "Do not reuse it after changing a shared configuration; move it aside and regenerate."
        )
    if not isinstance(payload.get("model_state_dict"), dict):
        raise RuntimeError(f"Paired initialization {path} has no valid model_state_dict.")
    actual_state_hash = _state_dict_sha256(payload["model_state_dict"])
    if payload.get("model_state_sha256") != actual_state_hash:
        raise RuntimeError(
            f"Paired initialization {path} failed its model-state SHA-256 integrity check."
        )


def apply_paired_initialization(model, model_name, config, workspace_root):
    """Create/load epoch-0 baseline weights and strictly copy all common tensors."""
    if not paired_initialization_enabled(config):
        return None

    settings = _settings(config)
    allowed_models = (settings["baseline_model"], settings["candidate_model"])
    if model_name not in allowed_models:
        raise RuntimeError(
            f"Paired initialization only supports {allowed_models}, not {model_name}."
        )

    path = _initialization_path(config, workspace_root)
    if not path.is_file():
        if model_name == settings["baseline_model"]:
            baseline_state = model.state_dict()
        else:
            baseline_state = _baseline_state_from_seed(config, model_name)
        _save_initialization(path, baseline_state, config)
        logger.info(f"Created paired epoch-0 initialization: {path}")

    payload = torch.load(path, map_location="cpu")
    _validate_payload(payload, config, path)
    baseline_state = payload["model_state_dict"]
    target_state = model.state_dict()

    shape_mismatches = {
        key: (tuple(value.shape), tuple(target_state[key].shape))
        for key, value in baseline_state.items()
        if key in target_state and value.shape != target_state[key].shape
    }
    absent = [key for key in baseline_state if key not in target_state]
    if shape_mismatches or absent:
        raise RuntimeError(
            "Shared model tensors are incompatible with the paired initialization: "
            f"absent={absent[:5]}, shape_mismatches={dict(list(shape_mismatches.items())[:5])}."
        )

    if model_name == settings["baseline_model"]:
        model.load_state_dict(baseline_state, strict=True)
        extra_keys = []
    else:
        result = model.load_state_dict(baseline_state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"Unexpected baseline keys while initializing PARR: {result.unexpected_keys}")
        extra_keys = list(result.missing_keys)
        prefix = settings["allowed_extra_prefix"]
        invalid_extra = [key for key in extra_keys if not key.startswith(prefix)]
        if not extra_keys or invalid_extra:
            raise RuntimeError(
                "PARR may differ from the baseline only by its declared module: "
                f"missing={extra_keys[:10]}, invalid={invalid_extra[:10]}."
            )
        stage_scales = [
            parameter.detach()
            for name, parameter in model.named_parameters()
            if name.startswith(prefix + "stage_scales.")
        ]
        if not stage_scales or any(torch.count_nonzero(scale).item() != 0 for scale in stage_scales):
            raise RuntimeError("PARR stage scales must all be exactly zero at paired initialization.")

    current_state = model.state_dict()
    unequal = [key for key, value in baseline_state.items() if not torch.equal(value, current_state[key].cpu())]
    if unequal:
        raise RuntimeError(f"Common tensors were not copied exactly: {unequal[:10]}")

    metadata = {
        "format_version": FORMAT_VERSION,
        "kind": KIND,
        "seed": payload["seed"],
        "baseline_model": payload["baseline_model"],
        "initialization_file": path.name,
        "initialization_sha256": payload["model_state_sha256"],
        "initialization_file_sha256": _file_sha256(path),
        "shared_config_sha256": payload["shared_config_sha256"],
        "common_tensor_count": len(baseline_state),
        "model_only_tensor_count": len(extra_keys),
    }
    model.paired_initialization_metadata = metadata
    logger.info(
        f"Applied paired initialization to {model_name}: common={len(baseline_state)}, "
        f"model_only={len(extra_keys)}, sha256={metadata['initialization_sha256']}"
    )
    return metadata


def validate_resume_initialization(checkpoint, model, checkpoint_path):
    """Prevent a paired run from resuming an unpaired or differently initialized run."""
    from utils.run_contract import validate_run_contract
    contract = getattr(model, "run_contract", None)
    if contract is not None:
        validate_run_contract(checkpoint, contract, checkpoint_path)
    expected = getattr(model, "paired_initialization_metadata", None)
    if expected is None:
        return
    actual = checkpoint.get("paired_initialization_metadata")
    if actual is None:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} predates paired initialization metadata and cannot "
            "resume this controlled run. Move it aside to start from the paired epoch-0 state."
        )
    identity_fields = (
        "format_version", "kind", "seed", "baseline_model",
        "initialization_sha256", "shared_config_sha256",
    )
    mismatches = {
        key: (actual.get(key), expected.get(key))
        for key in identity_fields
        if actual.get(key) != expected.get(key)
    }
    if mismatches:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} uses a different paired initialization: {mismatches}."
        )
