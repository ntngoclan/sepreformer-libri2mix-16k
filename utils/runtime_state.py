"""Epoch-boundary RNG state for reproducible training resume."""

import random

import numpy as np
import torch


def _loader_spec(loader):
    if loader.persistent_workers:
        raise RuntimeError("Reproducible resume requires persistent_workers=False.")
    return {
        "dataset_size": len(loader.dataset), "batch_size": loader.batch_size,
        "num_workers": loader.num_workers, "drop_last": loader.drop_last,
        "sampler": type(loader.sampler).__name__,
    }


def capture_runtime_state(dataloaders=None):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loaders": {
            name: loader.generator.get_state()
            for name, loader in (dataloaders or {}).items()
            if loader.generator is not None
        },
        "loader_specs": {name: _loader_spec(loader)
                         for name, loader in (dataloaders or {}).items()},
    }


def restore_runtime_state(state, dataloaders=None):
    for name, spec in state.get("loader_specs", {}).items():
        if name not in (dataloaders or {}) or _loader_spec(dataloaders[name]) != spec:
            raise RuntimeError(f"DataLoader configuration changed on resume: {name}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Visible GPU count differs from the saved RNG state.")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    for name, value in state["loaders"].items():
        if name not in (dataloaders or {}) or dataloaders[name].generator is None:
            raise RuntimeError(f"Cannot restore DataLoader generator: {name}")
        dataloaders[name].generator.set_state(value.cpu())
