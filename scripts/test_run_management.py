"""Regression tests for run isolation, archival checkpoints, diagnostics and backup."""

import copy
import contextlib
import importlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts.backup_run import backup, verify
from utils.run_management import prepare_run
from utils.run_contract import make_run_contract, validate_run_contract
from utils.util_engine import save_latest_checkpoint
from utils.parr_diagnostics import collect_parr_diagnostics


class RunManagementTests(unittest.TestCase):
    def test_managed_resume_carries_full_state_and_historical_best(self):
        from utils.util_engine import save_checkpoint_per_best
        from utils.implements.schedulers import WarmupConstantSchedule
        module = importlib.import_module('models.SepReformer_Base_Libri2Mix_16K.engine')
        with tempfile.TemporaryDirectory() as directory:
            config = {'check_computations': {'dummy_len': 8, 'metrics': []}, 'engine': {'clip_norm': 5}}

            def create():
                model = torch.nn.Linear(2, 1)
                model.run_contract = make_run_contract(model, config)
                optimizer = torch.optim.AdamW(model.parameters())
                schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                              WarmupConstantSchedule(optimizer, 100)]
                return model, optimizer, schedulers

            model, optimizer, schedulers = create()
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            schedulers[1].step()
            parent = str(Path(directory) / 'parent')
            save_checkpoint_per_best(float('inf'), 1, 1, 5, model, optimizer, parent, schedulers=schedulers)
            save_latest_checkpoint(2, 2, 10, model, optimizer, parent, schedulers=schedulers,
                                   best_valid_loss=1, milestone_epochs=[10])
            output = str(Path(directory) / 'continued')
            resumed, new_optimizer, new_schedulers = create()
            args = SimpleNamespace(engine_mode='train', out_wav_dir=None, run_dir=output,
                                   resume=str(Path(parent) / 'epoch_0010.pth'))
            engine = module.Engine(args, config, resumed, {}, [None]*4, [new_optimizer],
                                   new_schedulers, (), torch.device('cpu'), work_dir=output)
            self.assertEqual(engine.start_epoch, 11)
            self.assertEqual(engine.best_valid_loss, 1)
            self.assertEqual(new_schedulers[1].last_epoch, schedulers[1].last_epoch)
            self.assertTrue(new_optimizer.state_dict()['state'])
            self.assertEqual(torch.load(Path(output) / 'checkpoints/best.pth')['epoch'], 5)
            torch.testing.assert_close(resumed.weight, model.weight, rtol=0, atol=0)

    def test_explicit_evaluation_uses_selected_checkpoint(self):
        for name in ('SepReformer_Base_Libri2Mix_16K', 'SepReformer_PARR_Libri2Mix_16K'):
            module = importlib.import_module(f'models.{name}.engine')
            with tempfile.TemporaryDirectory() as directory:
                model = torch.nn.Linear(2, 1)
                config = {'check_computations': {'dummy_len': 8, 'metrics': []},
                          'engine': {'clip_norm': 5}}
                model.run_contract = make_run_contract(model, config)
                model.paired_initialization_metadata = {'initialization_sha256': 'identity'}
                optimizer = torch.optim.AdamW(model.parameters())
                weights = Path(directory) / 'parent'
                save_latest_checkpoint(1, 1, 10, model, optimizer, str(weights), milestone_epochs=[10])
                with torch.no_grad():
                    model.weight.add_(100)
                save_latest_checkpoint(1, 1, 20, model, optimizer, str(weights), milestone_epochs=[20])
                output = Path(directory) / 'evaluation'
                args = SimpleNamespace(engine_mode='test', out_wav_dir=None,
                                       checkpoint=str(weights / 'epoch_0010.pth'), run_dir=str(output))
                engine = module.Engine(args, config, model, {}, [], [], [], (), torch.device('cpu'), work_dir=str(output))
                selected = torch.load(weights / 'epoch_0010.pth')
                torch.testing.assert_close(model.weight, selected['model_state_dict']['weight'])
                self.assertEqual(engine.start_epoch, 11)
                self.assertEqual(engine.evaluation_provenance['checkpoint'], 'epoch_0010.pth')
                self.assertEqual(model.paired_initialization_metadata['initialization_sha256'], 'identity')
                self.assertFalse(hasattr(engine, 'main_optimizer'))

    def test_engine_writes_scalar_tags_and_milestone(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from utils.implements.schedulers import WarmupConstantSchedule
        for name in ('SepReformer_Base_Libri2Mix_16K', 'SepReformer_PARR_Libri2Mix_16K'):
            module = importlib.import_module(f'models.{name}.engine')
            with tempfile.TemporaryDirectory() as directory:
                engine = object.__new__(module.Engine.__wrapped__)
                engine.device, engine.work_dir = 'cpu', directory
                engine.engine_mode, engine.start_epoch, engine.best_valid_loss = 'train', 10, float('inf')
                engine.model = torch.nn.Linear(1, 1)
                engine.checkpoint_path = str(Path(directory) / 'checkpoints')
                engine.config = {'engine': {'max_epoch': 10, 'start_scheduling': 50,
                                            'test_epochs': [], 'mvn': False, 'checkpoint_epochs': [10]}}
                engine.main_optimizer = torch.optim.AdamW(engine.model.parameters())
                engine.main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(engine.main_optimizer)
                engine.warmup_scheduler = WarmupConstantSchedule(engine.main_optimizer, 1000)
                from torch.utils.data import DataLoader, TensorDataset
                loader = DataLoader(TensorDataset(torch.ones(1, 1)), generator=torch.Generator())
                engine.dataloaders = {'train': loader, 'valid': loader}
                engine._train = lambda *args: (1., 2., 1)
                engine._validate = lambda *args: (3., 4., 1)
                with patch.object(torch.cuda, 'device', return_value=contextlib.nullcontext()):
                    engine.run()
                self.assertTrue((Path(directory) / 'checkpoints/epoch_0010.pth').is_file())
                events = EventAccumulator(str(Path(directory) / 'tensorboard')).Reload()
                for tag in ('loss/train_time', 'loss/valid_time', 'loss/train_frequency',
                            'loss/valid_frequency', 'learning_rate/main',
                            'duration/train_seconds', 'duration/valid_seconds'):
                    self.assertEqual(events.Scalars(tag)[0].step, 10)

    def test_run_collision_and_explicit_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {'experiment': {'seed': 0}, 'dataloader': {'seed': 0}}
            args = SimpleNamespace(model='test', engine_mode='train', run_id='first',
                                   seed=7, output_dir=directory, resume=None, checkpoint=None)
            run = prepare_run(args, config)
            self.assertEqual(run.parent.name, 'seed_0007')
            self.assertEqual(config['dataloader']['seed'], 7)
            with self.assertRaises(FileExistsError):
                prepare_run(args, config)
            checkpoint = run / 'checkpoints/latest.pth'
            checkpoint.write_bytes(b'original')
            args.run_id, args.resume = 'continued', str(checkpoint)
            continued = prepare_run(args, config)
            self.assertNotEqual(run, continued)
            self.assertEqual(checkpoint.read_bytes(), b'original')
            args.run_id = '../escape'
            with self.assertRaises(ValueError):
                prepare_run(args, config)

    def test_milestone_payload_and_next_rng_draw(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters())
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
            model.run_contract = {'identity': 'test'}
            model.paired_initialization_metadata = {'sha256': 'test'}
            save_latest_checkpoint(1, 2, 10, model, optimizer, directory,
                                   schedulers=[scheduler], milestone_epochs=[10])
            latest = torch.load(Path(directory) / 'latest.pth')
            milestone = torch.load(Path(directory) / 'epoch_0010.pth')
            self.assertEqual(set(latest), set(milestone))
            for key in ['epoch', 'best_valid_loss', 'run_contract', 'paired_initialization_metadata', 'scheduler_state_dicts']:
                self.assertEqual(latest[key], milestone[key])
            self.assertTrue(milestone['optimizer_state_dict']['state'])
            for key in latest['model_state_dict']:
                torch.testing.assert_close(latest['model_state_dict'][key], milestone['model_state_dict'][key], rtol=0, atol=0)
            torch.testing.assert_close(latest['runtime_state']['torch_cpu'], milestone['runtime_state']['torch_cpu'])
            original = (Path(directory) / 'epoch_0010.pth').read_bytes()
            save_latest_checkpoint(3, 4, 11, model, optimizer, directory, milestone_epochs=[10])
            self.assertEqual((Path(directory) / 'epoch_0010.pth').read_bytes(), original)

    def test_manifest_change_blocks_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.csv'
            path.write_text('first')
            config = {'dataset': {'manifest': str(path)}}
            model = torch.nn.Linear(1, 1)
            first = make_run_contract(model, config)
            path.write_text('changed')
            second = make_run_contract(model, config)
            with self.assertRaises(RuntimeError):
                validate_run_contract({'run_contract': first}, second, 'test.pth')

    def test_diagnostics_preserve_rng_parameters_and_hooks(self):
        from models.SepReformer_PARR_Libri2Mix_16K.modules.module import ProgressiveAdaptiveResidualRefinement

        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.separator = torch.nn.Module()
                self.separator.parr = ProgressiveAdaptiveResidualRefinement(
                    num_stages=2, in_channels=4, bottleneck_channels=2)

            def forward(self, waveform, **kwargs):
                feature = waveform[:, None].expand(-1, 4, -1)
                for i in range(2):
                    feature = self.separator.parr(feature, i)
                return feature

        class Data:
            def __len__(self):
                return 2

            def __getitem__(self, index):
                random.random(); np.random.random(); torch.rand(1)
                return {'key': str(index), 'mix': np.sin(np.arange(32))}

        model = Probe().train()
        before = copy.deepcopy(model.state_dict())
        from utils.runtime_state import capture_runtime_state, restore_runtime_state
        state = capture_runtime_state()
        expected = (random.random(), np.random.random(), torch.rand(1))
        restore_runtime_state(state)
        metrics = collect_parr_diagnostics(model, Data(), 'cpu', 2)
        self.assertEqual(metrics['parr/gate_mean'], .5)
        self.assertEqual(metrics['parr/gate_std'], 0)
        self.assertEqual(metrics['parr/correction_feature_ratio'], 0)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        torch.testing.assert_close(torch.rand(1), expected[2], rtol=0, atol=0)
        self.assertTrue(model.training)
        self.assertFalse(model.separator.parr._forward_hooks)
        self.assertFalse(model.separator.parr.shared_temporal_core.temporal_controller._forward_hooks)
        for key, value in before.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_backup_verifies_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'run'
            run.mkdir()
            (run / 'config.yaml').write_text('config: {}')
            (run / 'checkpoint.pth').write_bytes(b'checkpoint')
            destination = Path(directory) / 'backup.zip'
            backup(run, destination)
            self.assertEqual(verify(destination), 2)
            with self.assertRaises(FileExistsError):
                backup(run, destination)
            with destination.open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaises(ValueError):
                verify(destination)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
