"""CPU regression checks; also used by the deployment preflight.

Run: python -m scripts.runtime_checks
"""

import copy
import csv
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger

from utils.implements.schedulers import WarmupConstantSchedule
from utils.runtime_state import restore_runtime_state
from utils.util_engine import save_checkpoint_per_best, save_latest_checkpoint


class RandomCropProbe(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return torch.tensor([index, random.random(), np.random.random(), torch.rand(()).item()])


def check_resume(workers=0):
    """Actual AdamW/scheduler checkpoint; next epoch must match uninterrupted."""
    from models.SepReformer_PARR_Libri2Mix_16K.engine import (
        _load_training_resume, _select_scratch_checkpoint,
    )
    from models.SepReformer_PARR_Libri2Mix_16K.dataset import _seed_worker
    torch.manual_seed(4)
    np.random.seed(4)
    random.seed(4)

    def create():
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        schedulers = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0),
            WarmupConstantSchedule(optimizer, 4),
        )
        loader = DataLoader(RandomCropProbe(), batch_size=2, shuffle=True,
                            generator=torch.Generator().manual_seed(7),
                            num_workers=workers, worker_init_fn=_seed_worker)
        return model, optimizer, schedulers, {"train": loader}

    def epoch(model, optimizer, schedulers, loaders, epoch_number):
        batches, losses = [], []
        for batch in loaders["train"]:
            batches.append(batch.clone())
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch).square().mean()
            loss.backward()
            optimizer.step()
            if epoch_number == 1:
                schedulers[1].step()
            losses.append(loss.detach().clone())
        schedulers[0].step(1.0)
        return batches, losses

    model, optimizer, schedulers, loaders = create()
    epoch(model, optimizer, schedulers, loaders, 1)
    with tempfile.TemporaryDirectory() as directory:
        save_checkpoint_per_best(float("inf"), 1.0, 2.0, 1, model, optimizer,
                                 directory, schedulers=schedulers, dataloaders=loaders)
        save_latest_checkpoint(1.0, 2.0, 1, model, optimizer, directory,
                               schedulers=schedulers, dataloaders=loaders)
        assert Path(_select_scratch_checkpoint(directory, "test")).name == "best.pth"
        latest = _select_scratch_checkpoint(directory, "train")
        expected_batches, expected_losses = epoch(model, optimizer, schedulers, loaders, 2)
        expected_weights = copy.deepcopy(model.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_lr = optimizer.param_groups[0]["lr"]
        resumed, opt2, sch2, loaders2 = create()
        start, best, runtime = _load_training_resume(latest, resumed, opt2, sch2, "cpu")
        assert start == 2 and best == 1.0
        restore_runtime_state(runtime, loaders2)
        actual_batches, actual_losses = epoch(resumed, opt2, sch2, loaders2, 2)
        for expected, actual in zip(expected_batches + expected_losses,
                                    actual_batches + actual_losses):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        for key, expected in expected_weights.items():
            torch.testing.assert_close(expected, resumed.state_dict()[key], rtol=0, atol=0)
        assert opt2.param_groups[0]["lr"] == expected_lr
        for key, state in expected_optimizer["state"].items():
            for field, expected in state.items():
                actual = opt2.state_dict()["state"][key][field]
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        assert not list(Path(directory).glob("*.tmp"))
    return {"workers": workers, "next_epoch_exact": True, "scheduler_save_load": True}


class RuntimeRegressionTests(unittest.TestCase):
    def test_resume_no_workers(self):
        check_resume(0)

    def test_warmup_lr_sequence_and_order(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = WarmupConstantSchedule(optimizer, 1000)
        used = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones(1, 1)).sum().backward()
            used.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        np.testing.assert_allclose(used, [1e-6, 2e-6, 3e-6], rtol=0, atol=1e-15)

    def test_resume_two_workers(self):
        check_resume(2)

    def test_paired_metadata_is_saved_and_mismatch_is_rejected(self):
        from utils.paired_initialization import validate_resume_initialization

        model = torch.nn.Linear(2, 2)
        model.paired_initialization_metadata = {
            "format_version": 1,
            "kind": "sepreformer_paired_initialization",
            "seed": 0,
            "baseline_model": "baseline",
            "initialization_sha256": "init-a",
            "shared_config_sha256": "config-a",
        }
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            save_latest_checkpoint(1, 1, 1, model, optimizer, directory)
            path = Path(directory) / "latest.pth"
            checkpoint = torch.load(path, map_location="cpu")
            self.assertEqual(
                checkpoint["paired_initialization_metadata"],
                model.paired_initialization_metadata,
            )
            validate_resume_initialization(checkpoint, model, path)
            checkpoint["paired_initialization_metadata"]["initialization_sha256"] = "init-b"
            with self.assertRaisesRegex(RuntimeError, "different paired initialization"):
                validate_resume_initialization(checkpoint, model, path)

    def test_natural_gamma_opening(self):
        from models.SepReformer_PARR_Libri2Mix_16K.modules.module import ProgressiveAdaptiveResidualRefinement
        torch.manual_seed(11)
        parr = ProgressiveAdaptiveResidualRefinement(
            num_stages=4, in_channels=128, bottleneck_channels=64,
            kernel_size=3, dilations=(1, 2, 4, 8), dropout_rate=0.0, gamma_init=0.0,
        )
        optimizer = torch.optim.AdamW(parr.parameters(), lr=1e-3)
        x, target = torch.randn(2, 128, 65), torch.randn(2, 128, 65)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            outputs = [parr(x, index) for index in range(4)]
            if step == 0:
                for output in outputs:
                    torch.testing.assert_close(output, x, rtol=0, atol=0)
            sum((output - target).square().mean() for output in outputs).backward()
            grad = parr.shared_temporal_core.temporal_controller.weight.grad
            self.assertIsNotNone(grad)
            self.assertEqual(bool(torch.count_nonzero(grad)), step == 1)
            optimizer.step()
        self.assertTrue(all(scale.detach().abs() > 0 for scale in parr.stage_scales))

    def test_paired_initialization_and_full_model_identity(self):
        import importlib
        import yaml
        from utils.paired_initialization import apply_paired_initialization
        from utils.util_system import set_random_seed

        names = (
            "SepReformer_Base_Libri2Mix_16K",
            "SepReformer_PARR_Libri2Mix_16K",
        )
        configs = []
        for name in names:
            with open(Path("models") / name / "configs.yaml") as stream:
                configs.append(yaml.safe_load(stream)["config"])

        with tempfile.TemporaryDirectory() as directory:
            models = []
            metadata = []
            for name, config in zip(names, configs):
                config["paired_initialization"]["directory"] = directory
                set_random_seed(**config["experiment"])
                model_class = importlib.import_module(f"models.{name}.model").Model
                model = model_class(**config["model"])
                metadata.append(
                    apply_paired_initialization(model, name, config, Path.cwd())
                )
                models.append(model.eval())

            baseline_state, parr_state = (model.state_dict() for model in models)
            common_keys = set(baseline_state) & set(parr_state)
            self.assertEqual(common_keys, set(baseline_state))
            for key in common_keys:
                torch.testing.assert_close(
                    baseline_state[key], parr_state[key], rtol=0, atol=0
                )
            self.assertEqual(
                metadata[0]["initialization_sha256"],
                metadata[1]["initialization_sha256"],
            )
            self.assertGreater(metadata[1]["model_only_tensor_count"], 0)

            waveform = torch.randn(1, 2048)
            with torch.inference_mode():
                baseline_output = models[0](waveform)
                parr_output = models[1](waveform)
            baseline_tensors = baseline_output[0] + [
                tensor for stage in baseline_output[1] for tensor in stage
            ]
            parr_tensors = parr_output[0] + [
                tensor for stage in parr_output[1] for tensor in stage
            ]
            self.assertEqual(len(baseline_tensors), len(parr_tensors))
            for baseline_tensor, parr_tensor in zip(baseline_tensors, parr_tensors):
                torch.testing.assert_close(
                    baseline_tensor, parr_tensor, rtol=0, atol=0
                )

    def test_comparison_rejects_missing_keys(self):
        from utils.compare_separation_metrics import compare
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("base.csv", "parr.csv")]
            for path, keys in zip(paths, [("a", "b"), ("a",)]):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["key", "num_samples", "si_snri"])
                    writer.writeheader()
                    writer.writerows(dict(key=key, num_samples=16000, si_snri=1) for key in keys)
            with self.assertRaisesRegex(ValueError, "key sets differ"):
                compare(*paths, 20, .95, 0)

    def test_evaluator_without_optional_pesq(self):
        from utils.evaluation_metrics import SeparationMetricsEvaluator
        rng = np.random.default_rng(3)
        references = [rng.standard_normal(16000), rng.standard_normal(16000)]
        mixture = references[0] + references[1]
        estimates = [value + 1e-3 * rng.standard_normal(16000) for value in references]
        evaluator = SeparationMetricsEvaluator(
            sampling_rate=16000, num_spks=2, compute_pesq=False,
            bootstrap_samples=20, bootstrap_seed=0,
        )
        row = evaluator.add_utterance("probe", mixture, references, estimates,
                                      16000, latency_seconds=.01, peak_vram_mb=0)
        for metric in ("si_snri", "bss_sdri", "stoi", "estoi", "rtf"):
            self.assertTrue(np.isfinite(row[metric]), metric)
        with tempfile.TemporaryDirectory() as directory:
            summary = evaluator.save(directory)
            self.assertEqual(summary["num_utterances"], 1)
            self.assertTrue((Path(directory) / "metrics_per_utterance.csv").is_file())
            self.assertTrue((Path(directory) / "metrics_summary.json").is_file())

    def test_loader_rejects_raw_length_mismatch(self):
        from models.SepReformer_PARR_Libri2Mix_16K.dataset import Libri2MixDataset
        dataset = object.__new__(Libri2MixDataset)
        dataset.examples = [("bad.wav", "mix", ("s1", "s2"))]
        dataset._load_audio = lambda ref: np.ones(2048 if ref == "mix" else 2040)
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            dataset[0]

    def test_full_model_inference_equivalence(self):
        import importlib
        import yaml
        for name in ("SepReformer_Base_Libri2Mix_16K", "SepReformer_PARR_Libri2Mix_16K"):
            with open(Path("models") / name / "configs.yaml") as stream:
                config = yaml.safe_load(stream)["config"]
            model = importlib.import_module(f"models.{name}.model").Model(**config["model"]).eval()
            waveform = torch.randn(1, 2048)
            calls = []
            hook = model.decoder_bn[0].register_forward_hook(lambda *args: calls.append(1))
            with torch.no_grad():
                original, auxiliary = model(waveform)
                self.assertTrue(calls)
                calls.clear()
                fast, empty = model(waveform, return_aux=False)
                self.assertEqual(empty, [])
                self.assertEqual(calls, [])
                for left, right in zip(original, fast):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
            hook.remove()

    def test_evaluation_does_not_construct_optimizer(self):
        from types import SimpleNamespace
        from utils.run_contract import make_run_contract
        from models.SepReformer_PARR_Libri2Mix_16K.engine import Engine
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "log" / "scratch_weights"
            model = torch.nn.Linear(4, 4)
            config = {"check_computations": {"dummy_len": 8, "metrics": []},
                      "engine": {"clip_norm": 5}}
            model.run_contract = make_run_contract(model, config)
            optimizer = torch.optim.AdamW(model.parameters())
            save_checkpoint_per_best(float("inf"), 1, 1, 1, model, optimizer, str(weights))
            engine = Engine(SimpleNamespace(engine_mode="test", out_wav_dir=None),
                            config, model, {}, [], [], [],
                            (0,), torch.device("cpu"), work_dir=directory)
            self.assertFalse(hasattr(engine, "main_optimizer"))
            self.assertEqual(engine.start_epoch, 2)

    def test_full_model_two_updates(self):
        import importlib
        import yaml
        from utils.util_implement import CriterionFactory, OptimizerFactory, SchedulerFactory
        for name in ("SepReformer_Base_Libri2Mix_16K", "SepReformer_PARR_Libri2Mix_16K"):
            with open(Path("models") / name / "configs.yaml") as stream:
                config = yaml.safe_load(stream)["config"]
            model = importlib.import_module(f"models.{name}.model").Model(**config["model"]).train()
            spectral, temporal, _, _ = CriterionFactory(config["criterion"], torch.device("cpu")).get_criterions()
            optimizer = OptimizerFactory(config["optimizer"], model.parameters()).get_optimizers()[0]
            schedulers = SchedulerFactory(config["scheduler"], [optimizer]).get_schedulers()
            sources = [torch.randn(2, 2048), torch.randn(2, 2048)]
            sizes = torch.full((2,), 2048.0)
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                outputs, auxiliary = model(sources[0] + sources[1])
                loss_t = temporal(estims=outputs, input_sizes=sizes, target_attr=sources)
                loss_f = [spectral(estims=stage, idx=i, input_sizes=sizes, target_attr=sources)
                          for i, stage in enumerate(auxiliary)]
                loss = (.6 * loss_t + .4 * sum(loss_f) / len(loss_f)) / 2
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None))
                if step == 1 and hasattr(model.separator, "parr"):
                    grad = model.separator.parr.shared_temporal_core.temporal_controller.weight.grad
                    self.assertTrue(torch.count_nonzero(grad) > 0)
                before = next(model.parameters()).detach().clone()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["engine"]["clip_norm"],
                                               error_if_nonfinite=True)
                optimizer.step()
                schedulers[1].step()
                self.assertFalse(torch.equal(before, next(model.parameters()).detach()))
                del outputs, auxiliary, loss, loss_t, loss_f
            with tempfile.TemporaryDirectory() as directory:
                save_latest_checkpoint(1, 1, 1, model, optimizer, directory, schedulers=schedulers)
                checkpoint = torch.load(Path(directory) / "latest.pth", map_location="cpu")
                self.assertTrue(checkpoint["optimizer_state_dict"]["state"])
                self.assertEqual(len(checkpoint["scheduler_state_dicts"]), 2)


if __name__ == "__main__":
    logger.remove()
    torch.set_num_threads(2)
    unittest.main()
