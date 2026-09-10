"""Fail-fast deployment check for SepReformer-PARR on one NVIDIA L40."""

import argparse
import json
import os
import shutil
import tempfile
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path




ROOT = Path(__file__).resolve().parents[1]
# Direct execution sets sys.path[0] to scripts/, not the repository.
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
EXPECTED_PARTITION_SIZES = {"train": 13900, "valid": 3000, "test": 3000}


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def load_config(model_name):
    path = ROOT / "models" / model_name / "configs.yaml"
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if "config" not in document:
        raise RuntimeError(f"Missing top-level config mapping in {path}")
    return document["config"]


def check_shared_configuration(base_config, parr_config):
    comparable_parr = json.loads(json.dumps(parr_config))
    parr_block = comparable_parr["model"]["module_separator"].pop("parr", None)
    if parr_block is None:
        raise RuntimeError("PARR configuration block is missing.")
    if base_config != comparable_parr:
        raise RuntimeError(
            "Baseline and PARR shared configurations differ. This invalidates a controlled comparison."
        )
    return parr_block


def check_dataset(config, full=True):
    from models.SepReformer_PARR_Libri2Mix_16K.dataset import Libri2MixDataset

    report = {}
    for partition in ("train", "valid", "test"):
        dataset = Libri2MixDataset(config["dataset"], partition)
        expected = EXPECTED_PARTITION_SIZES[partition]
        if len(dataset) != expected:
            raise RuntimeError(
                f"Libri2Mix {partition} contains {len(dataset)} mixtures; expected {expected}."
            )
        # Read raw audio before crop/stride trimming so mismatches stay visible.
        indices = range(len(dataset)) if full else [0]
        for index in indices:
            key, mixture_ref, source_refs = dataset.examples[index]
            audio = [dataset._load_audio(ref) for ref in (mixture_ref, *source_refs)]
            lengths = [len(value) for value in audio]
            if len(set(lengths)) != 1:
                raise RuntimeError(f"{partition}/{key}: raw lengths differ: {lengths}")
            if index % 1000 == 0:
                print(f"Dataset {partition}: checked {index + 1}/{len(dataset)}", flush=True)
        example = dataset[0]
        report[partition] = {
            "files_decoded": len(indices) * (1 + config["model"]["num_spks"]),
            "full_scan": full,
            "storage": dataset.storage,
            "num_mixtures": len(dataset),
            "sample_key": example["key"],
            "sample_num_samples": int(example["num_sample"]),
        }
    return report


def check_checkpoint_roundtrip():
    from scripts.runtime_checks import check_resume
    return {"single_worker": check_resume(0), "multi_worker": check_resume(2)}


def check_evaluator(config):
    from models.SepReformer_PARR_Libri2Mix_16K.dataset import Libri2MixDataset
    from utils.evaluation_metrics import SeparationMetricsEvaluator

    example = Libri2MixDataset(config["dataset"], "test")[0]
    length = min(int(example["num_sample"]), 32000)
    references = [source[:length] for source in example["src"]]
    rng = np.random.default_rng(0)
    estimates = [
        reference + 1.0e-5 * rng.standard_normal(reference.shape)
        for reference in references
    ]
    evaluator = SeparationMetricsEvaluator(
        sampling_rate=config["dataset"]["sampling_rate"],
        num_spks=config["model"]["num_spks"],
        bootstrap_samples=20,
        bootstrap_seed=0,
    )
    row = evaluator.add_utterance(
        key=example["key"],
        mixture=example["mix"][:length],
        references=references,
        estimates=estimates,
        length=length,
        latency_seconds=0.01,
        peak_vram_mb=0.0,
    )
    required = ("si_snri", "bss_sdri", "pesq_wb", "stoi", "estoi", "rtf")
    invalid = [name for name in required if not np.isfinite(row[name])]
    if invalid:
        raise RuntimeError(f"Evaluator returned non-finite required metrics: {invalid}")
    with tempfile.TemporaryDirectory() as directory:
        summary = evaluator.save(directory)
        if summary["num_utterances"] != 1:
            raise RuntimeError("Evaluator summary utterance count is invalid.")
        if not (Path(directory) / "metrics_per_utterance.csv").is_file():
            raise RuntimeError("Evaluator CSV was not written.")
        if not (Path(directory) / "metrics_summary.json").is_file():
            raise RuntimeError("Evaluator JSON was not written.")
    return {name: float(row[name]) for name in required}


def check_gpu(require_l40):
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, found {torch.cuda.device_count()}. "
            "Set CUDA_VISIBLE_DEVICES to one device."
        )
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if not torch.__version__.startswith("2.1.2"):
        raise RuntimeError(
            f"Expected PyTorch 2.1.2 for the controlled experiment, found {torch.__version__}."
        )
    if torch.version.cuda is None or not torch.version.cuda.startswith("12.1"):
        raise RuntimeError(
            f"Expected the CUDA 12.1 PyTorch build, found CUDA {torch.version.cuda}."
        )
    if require_l40 and "L40" not in name.upper():
        raise RuntimeError(f"Expected an NVIDIA L40, found '{name}'.")
    if require_l40 and capability != (8, 9):
        raise RuntimeError(
            f"Expected L40 compute capability 8.9, found {capability[0]}.{capability[1]}."
        )
    test_tensor = torch.randn(256, 256, device="cuda")
    test_value = (test_tensor @ test_tensor).mean()
    torch.cuda.synchronize()
    if not torch.isfinite(test_value):
        raise RuntimeError("Basic CUDA matrix multiplication produced a non-finite value.")
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": name,
        "compute_capability": list(capability),
        "vram_gib": properties.total_memory / (1024 ** 3),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "compiled_architectures": torch.cuda.get_arch_list(),
    }


def check_paired_model_identity(base_config, parr_config, samples=2048):
    """Verify the complete baseline and zero-gamma PARR outputs are bit-identical."""
    from models.SepReformer_Base_Libri2Mix_16K.model import Model as BaselineModel
    from models.SepReformer_PARR_Libri2Mix_16K.model import Model as ParrModel
    from utils.paired_initialization import apply_paired_initialization
    from utils.util_system import set_random_seed

    names = ("SepReformer_Base_Libri2Mix_16K", "SepReformer_PARR_Libri2Mix_16K")
    model_classes = (BaselineModel, ParrModel)
    configs = (base_config, parr_config)
    models, metadata = [], []
    for name, model_class, config in zip(names, model_classes, configs):
        set_random_seed(**config["experiment"])
        model = model_class(**config["model"])
        metadata.append(
            apply_paired_initialization(
                model, model_name=name, config=config, workspace_root=ROOT
            )
        )
        models.append(model.cuda().eval())

    if metadata[0]["initialization_sha256"] != metadata[1]["initialization_sha256"]:
        raise RuntimeError("Baseline and PARR report different epoch-0 model-state hashes.")
    set_random_seed(**base_config["experiment"])
    waveform = torch.randn(1, int(samples), device="cuda")
    with torch.inference_mode():
        outputs = [model(waveform) for model in models]
    tensors = [
        output[0] + [tensor for stage in output[1] for tensor in stage]
        for output in outputs
    ]
    if len(tensors[0]) != len(tensors[1]):
        raise RuntimeError("Baseline and PARR return different numbers of waveform tensors.")
    for index, (baseline_tensor, parr_tensor) in enumerate(zip(*tensors)):
        try:
            torch.testing.assert_close(baseline_tensor, parr_tensor, rtol=0, atol=0)
        except AssertionError as error:
            raise RuntimeError(
                f"Paired epoch-0 output tensor {index} is not bit-identical."
            ) from error
    result = {
        "identity": "bit_exact",
        "probe_samples": int(samples),
        "output_tensor_count": len(tensors[0]),
        "initialization_sha256": metadata[0]["initialization_sha256"],
        "initialization_file_sha256": metadata[0]["initialization_file_sha256"],
        "shared_config_sha256": metadata[0]["shared_config_sha256"],
        "baseline_common_tensors": metadata[0]["common_tensor_count"],
        "parr_only_tensors": metadata[1]["model_only_tensor_count"],
    }
    del outputs, tensors, waveform, models
    torch.cuda.empty_cache()
    return result


def check_forward_backward(model_name, config, forward_samples, batch_size):
    if model_name == "SepReformer_PARR_Libri2Mix_16K":
        from models.SepReformer_PARR_Libri2Mix_16K.model import Model
    elif model_name == "SepReformer_Base_Libri2Mix_16K":
        from models.SepReformer_Base_Libri2Mix_16K.model import Model
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    from utils.paired_initialization import apply_paired_initialization
    from utils.util_system import set_random_seed

    samples = max(256, int(forward_samples))
    multiple = int(config["dataset"].get("sample_length_multiple", 1))
    samples -= samples % multiple
    batch_size = int(batch_size or config["dataloader"]["batch_size"])
    set_random_seed(**config["experiment"])
    model = Model(**config["model"])
    from utils.run_contract import make_run_contract
    model.run_contract = make_run_contract(model, config)
    paired_metadata = apply_paired_initialization(
        model, model_name=model_name, config=config, workspace_root=ROOT
    )
    # Make preflight independent of whether it created or loaded the init file.
    set_random_seed(**config["experiment"])
    # Match Engine._train: loader targets/lengths start on CPU and the loss
    # transfers them; only the mixture is moved before the model forward.
    sources = [
        torch.randn(batch_size, samples)
        for _ in range(config["model"]["num_spks"])
    ]
    waveform = torch.stack(sources).sum(dim=0).cuda()
    input_sizes = torch.full(
        (batch_size,), samples, dtype=torch.float32
    )
    model = model.cuda().train()
    from utils.util_implement import CriterionFactory, OptimizerFactory, SchedulerFactory

    spectral_loss, time_loss, _, _ = CriterionFactory(
        config["criterion"], torch.device("cuda:0")
    ).get_criterions()
    optimizer = OptimizerFactory(config["optimizer"], model.parameters()).get_optimizers()[0]
    schedulers = SchedulerFactory(config["scheduler"], [optimizer]).get_schedulers()
    torch.cuda.reset_peak_memory_stats(0)
    step_reports = []
    for step_index in range(2):
        optimizer.zero_grad(set_to_none=True)
        estimates, auxiliaries = model(waveform, input_sizes=input_sizes)
        if len(estimates) != config["model"]["num_spks"]:
            raise RuntimeError("Unexpected number of separated outputs.")
        tensors = list(estimates) + [tensor for stage in auxiliaries for tensor in stage]
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise RuntimeError("Forward pass produced NaN or infinity.")
        if any(tensor.shape[0] != batch_size for tensor in estimates):
            raise RuntimeError("Separated output batch dimension is invalid.")
        prepared_targets = spectral_loss.prepare_targets(sources, input_sizes)
        spectral_losses = [
            spectral_loss(
                estims=stage,
                idx=index,
                input_sizes=input_sizes,
                target_attr=sources,
                prepared_targets=prepared_targets,
            )
            for index, stage in enumerate(auxiliaries)
        ]
        waveform_loss = time_loss(
            estims=estimates,
            input_sizes=input_sizes,
            target_attr=sources,
        )
        alpha = 0.4
        loss = (
            (1.0 - alpha) * waveform_loss
            + alpha * sum(spectral_losses) / len(spectral_losses)
        ) / config["model"]["num_spks"]
        if not torch.isfinite(loss):
            raise RuntimeError("Configured training objective is NaN or infinity.")
        loss.backward()
        trainable_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not trainable_gradients or not all(torch.isfinite(grad).all() for grad in trainable_gradients):
            raise RuntimeError("Backward pass is missing gradients or produced non-finite gradients.")
        if hasattr(model.separator, "parr"):
            stage_scale_gradients = [
                parameter.grad for parameter in model.separator.parr.stage_scales
            ]
            if any(
                grad is None or not torch.isfinite(grad).all()
                for grad in stage_scale_gradients
            ):
                raise RuntimeError("PARR stage scales did not receive finite gradients.")
        if step_index == 1 and hasattr(model.separator, "parr"):
            gate_grad = model.separator.parr.shared_temporal_core.temporal_controller.weight.grad
            if gate_grad is None or torch.count_nonzero(gate_grad) == 0:
                raise RuntimeError("Gate has no loss gradient after gamma opened naturally.")
        before = next(model.parameters()).detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["engine"]["clip_norm"],
                                       error_if_nonfinite=True)
        optimizer.step()
        schedulers[1].step()
        if torch.equal(before, next(model.parameters()).detach()):
            raise RuntimeError("optimizer.step() did not update model weights.")
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("Non-finite parameters after optimizer.step().")
        step_reports.append(float(loss.detach().cpu()))
        del before, trainable_gradients, tensors
        if step_index == 0:
            del estimates, auxiliaries, loss, spectral_losses, waveform_loss
    torch.cuda.synchronize()
    result = {
        "input_samples": samples,
        "batch_size": batch_size,
        "output_shapes": [list(tensor.shape) for tensor in estimates],
        "auxiliary_stages": len(auxiliaries),
        "optimizer_steps": len(step_reports),
        "step_losses": step_reports,
        "loss": float(loss.detach().cpu()),
        "waveform_loss": float(waveform_loss.detach().cpu()),
        "spectral_losses": [
            float(value.detach().cpu()) for value in spectral_losses
        ],
        "peak_vram_mib": torch.cuda.max_memory_allocated(0) / (1024 ** 2),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model": model_name,
        "paired_initialization": paired_metadata,
    }
    del estimates, auxiliaries, loss, spectral_losses, waveform_loss
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        with_aux, _ = model(waveform)
        without_aux, empty_aux = model(waveform, return_aux=False)
        if empty_aux:
            raise RuntimeError("Inference unexpectedly computed auxiliary waveforms.")
        for left, right in zip(with_aux, without_aux):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    # Saving real model + AdamW state + both schedulers catches serialization failures.
    from utils.util_engine import save_latest_checkpoint
    with tempfile.TemporaryDirectory() as directory:
        save_latest_checkpoint(1.0, 1.0, 1, model, optimizer, directory, schedulers=schedulers)
        checkpoint = torch.load(Path(directory) / "latest.pth", map_location="cpu")
        if not checkpoint["optimizer_state_dict"]["state"]:
            raise RuntimeError("AdamW state was not saved.")
    del checkpoint, model, optimizer, schedulers, waveform, sources
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-l40", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--quick-data", action="store_true", help="Sample-only scan; not a full release check.")
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument("--forward-samples", type=int, default=64000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to the configured training batch size (currently 2).",
    )
    parser.add_argument(
        "--report", default=str(ROOT / "run_logs" / "preflight_l40_report.json")
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    # Invalidate a previous success before any new check can fail.
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "status": "incomplete", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "A check failed or is still running unless replaced by a completed report.",
    }, indent=2), encoding="utf-8")
    base_config = load_config("SepReformer_Base_Libri2Mix_16K")
    parr_config = load_config("SepReformer_PARR_Libri2Mix_16K")
    parr_block = check_shared_configuration(base_config, parr_config)

    from models.SepReformer_PARR_Libri2Mix_16K.smoke_test_parr import main as parr_smoke

    from utils.util_system import set_random_seed
    set_random_seed(**parr_config["experiment"])
    parr_smoke()
    disk = shutil.disk_usage(ROOT)
    report = {
        "status": "partial" if (args.skip_data or args.quick_data or args.skip_backward
                                or args.forward_samples != parr_config["dataset"]["max_len"]
                                or args.batch_size not in (None, parr_config["dataloader"]["batch_size"])) else "passed",
        "root": str(ROOT),
        "gpu": check_gpu(args.require_l40),
        "packages": {
            name: package_version(name)
            for name in (
                "numpy", "scipy", "torch", "torchvision", "torchaudio", "librosa",
                "soundfile", "mir-eval", "pesq", "pystoi", "PyYAML",
                "tensorboard", "ptflops", "thop", "torchinfo",
            )
        },
        "disk_free_gib": disk.free / (1024 ** 3),
        "parr_config": parr_block,
        "checkpoint": check_checkpoint_roundtrip(),
        "paired_initialization": check_paired_model_identity(base_config, parr_config),
    }
    if not args.skip_data:
        report["dataset"] = check_dataset(parr_config, full=not args.quick_data)
        report["evaluator"] = check_evaluator(parr_config)
    if not args.skip_backward:
        report["forward_backward"] = {
            "baseline": check_forward_backward(
                "SepReformer_Base_Libri2Mix_16K",
                base_config,
                args.forward_samples,
                args.batch_size,
            ),
            "parr": check_forward_backward(
                "SepReformer_PARR_Libri2Mix_16K",
                parr_config,
                args.forward_samples,
                args.batch_size,
            ),
        }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Preflight {report['status']}. Report: {report_path}")


if __name__ == "__main__":
    main()
