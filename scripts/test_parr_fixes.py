"""Regression tests for the PARR review fixes. Run: python -m scripts.test_parr_fixes"""

import copy
import csv
import importlib
import tempfile
import unittest
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml
from loguru import logger

from utils.implements.criterions import PIT_SISNR_mag, STFT
from utils.evaluation_metrics import SeparationMetricsEvaluator, si_snr, _bootstrap_interval
from utils.compare_separation_metrics import compare
from utils.run_contract import make_run_contract, validate_run_contract
from utils.paired_initialization import validate_resume_initialization
from utils.util_engine import save_latest_checkpoint
from utils.implements.schedulers import WarmupConstantSchedule


def spectral(mel=False, scale_inv=True):
    return PIT_SISNR_mag(torch.device('cpu'), 1024, 256, 'hann', 4, 2, scale_inv, mel)


def cropped_reference_loss(criterion, estimates, targets, lengths):
    """Slow waveform-scale-before-STFT oracle, each utterance independently."""
    per_utterance = []
    for batch, length in enumerate(lengths.tolist()):
        length = int(length)
        costs = []
        for perm in permutations(range(2)):
            total = 0
            for speaker, target in enumerate(perm):
                est = estimates[speaker][batch:batch+1, :length]
                ref = targets[target][batch:batch+1, :length]
                est, ref = est - est.mean(-1, keepdim=True), ref - ref.mean(-1, keepdim=True)
                if criterion.scale_inv:
                    scale = ((est * ref).sum(-1, keepdim=True) / (ref.square().sum(-1, keepdim=True) + 1e-12)).clamp_min(.01)
                    ref = scale * ref
                a, b = criterion.stft[0](est)[0], criterion.stft[0](ref)[0]
                if criterion.mel_opt:
                    a, b = criterion.mel_fb(a), criterion.mel_fb(b)
                total = total - 20 * torch.log10(1e-12 + torch.linalg.vector_norm(b)
                                                / (torch.linalg.vector_norm(a-b) + 1e-12))
            costs.append(total)
        per_utterance.append(torch.stack(costs).min())
    return torch.stack(per_utterance).mean()


class SpectralFixTests(unittest.TestCase):
    def test_matches_cropped_oracle_value_and_gradients(self):
        torch.manual_seed(21)
        targets = [torch.randn(3, 2560) for _ in range(2)]
        # Positive, negative and very small scale exercise the clamp and epsilon.
        estimates = [(targets[s] * torch.tensor([.8, -.3, .001])[:, None]
                      + .1 * torch.randn(3, 2560)).requires_grad_() for s in range(2)]
        lengths = torch.tensor([1536, 2560, 512])
        for mel, scale_inv in ((False, True), (True, True), (False, False)):
            with self.subTest(mel=mel, scale_inv=scale_inv):
                criterion = spectral(mel, scale_inv)
                actual = criterion(estims=estimates, target_attr=targets, input_sizes=lengths, idx=0)
                expected = cropped_reference_loss(criterion, estimates, targets, lengths)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
                ga = torch.autograd.grad(actual, estimates)
                ge = torch.autograd.grad(expected, estimates)
                for a, b in zip(ga, ge):
                    torch.testing.assert_close(a, b, rtol=3e-4, atol=2e-6)
                    for index, length in enumerate(lengths):
                        self.assertEqual(torch.count_nonzero(a[index, length:]).item(), 0)

    def test_padding_invariance_and_target_cache(self):
        torch.manual_seed(23)
        criterion = spectral()
        targets = [torch.randn(1, 1536) for _ in range(2)]
        estimates = [t + .2 * torch.randn_like(t) for t in targets]
        lengths = torch.tensor([1536])
        expected = criterion(estims=estimates, target_attr=targets, input_sizes=lengths, idx=0)
        # Nonzero garbage in the padded tail must not leak into the objective.
        targets = [torch.cat([t, torch.randn(1, 1024)], -1) for t in targets]
        estimates = [torch.cat([t, torch.randn(1, 1024)], -1) for t in estimates]
        prepared = criterion.prepare_targets(targets, lengths)
        for stage in range(4):
            actual = criterion(estims=estimates, input_sizes=lengths, idx=stage, prepared_targets=prepared)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_short_stft_and_spectral_backward(self):
        criterion = spectral()
        for length in (8, 512, 1016, 1024):
            with self.subTest(length=length):
                targets = [torch.randn(2, length) for _ in range(2)]
                estimates = [torch.randn(2, length, requires_grad=True) for _ in range(2)]
                value = criterion(estims=estimates, target_attr=targets,
                                  input_sizes=torch.tensor([length, length]), idx=0)
                value.backward()
                self.assertTrue(torch.isfinite(value))
                self.assertTrue(all(torch.isfinite(x.grad).all() for x in estimates))

    def test_fft_work_reused_across_heads(self):
        criterion = spectral()
        targets = [torch.randn(2, 2048) for _ in range(2)]
        estimates = [torch.randn(2, 2048) for _ in range(2)]
        lengths = torch.tensor([2048, 1536])
        calls = []
        hooks = [stft.register_forward_hook(lambda *args: calls.append(1)) for stft in criterion.stft]
        prepared = criterion.prepare_targets(targets, lengths)
        for stage in range(4):
            criterion(estims=estimates, input_sizes=lengths, idx=stage, prepared_targets=prepared)
        for hook in hooks:
            hook.remove()
        self.assertEqual(len(calls), 5)  # one reference batch + four estimate batches


class MetricFixTests(unittest.TestCase):
    def test_silence_dc_and_small_gain(self):
        rng = np.random.default_rng(5)
        ref = rng.normal(size=1024)
        est = ref + .1 * rng.normal(size=1024)
        self.assertEqual(si_snr(np.zeros_like(ref), ref), -80)
        self.assertEqual(si_snr(np.ones_like(ref), ref), -80)
        with self.assertRaisesRegex(ValueError, 'reference'):
            si_snr(est, np.ones_like(ref))
        self.assertAlmostEqual(si_snr(est, ref), si_snr(-est * 1e-100, ref * 1e-80), places=10)

    def test_silence_is_retained_and_reported(self):
        evaluator = SeparationMetricsEvaluator(16000, 2, False, False, False, False, bootstrap_samples=0)
        sources = [np.random.default_rng(i).normal(size=1024) for i in (3, 4)]
        row = evaluator.add_utterance('silent', sum(sources), sources,
                                      [np.zeros(1024), np.zeros(1024)], 1024, .1)
        self.assertEqual(row['si_snr'], -80)
        self.assertEqual(len(evaluator.failures), 2)
        summary = evaluator.summarize()
        self.assertEqual(summary['metrics']['si_snr']['n_valid'], 1)
        self.assertTrue(np.isnan(summary['metrics']['si_snr']['ci_low']))

    def test_disabled_ci_and_confidence_labels(self):
        ci = _bootstrap_interval([1, 3], 0, .95, np.random.default_rng(0))
        self.assertTrue(np.isnan(ci).all())
        evaluator = SeparationMetricsEvaluator(16000, 2, False, False, False, False,
                                               bootstrap_samples=20, confidence=.9)
        evaluator.rows = [{'si_snr': 1}, {'si_snr': 3}]
        result = evaluator.summarize()['metrics']['si_snr']
        self.assertTrue(np.isfinite(result['ci_low']))
        self.assertTrue(np.isnan(result['ci95_low']))

    def test_reject_short_array_and_metric_protocol_mismatch(self):
        evaluator = SeparationMetricsEvaluator(16000, 2, False, False, False, False)
        with self.assertRaisesRegex(ValueError, 'length'):
            evaluator.add_utterance('short', np.ones(5), [np.ones(5)]*2, [np.ones(4)]*2, 5, .1)
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder)/name for name in ('a.csv', 'b.csv')]
            for path, protocol in zip(paths, ('v1', 'v2')):
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['key','num_samples','si_snri','metric_protocol'])
                    writer.writeheader()
                    writer.writerow(dict(key='a',num_samples=1024,si_snri=1,metric_protocol=protocol))
            with self.assertRaisesRegex(ValueError, 'protocol'):
                compare(*paths, 0, .95, 0)


class ContractFixTests(unittest.TestCase):
    def test_checkpoint_roundtrip_and_semantic_changes(self):
        model = torch.nn.Linear(2, 2)
        config = {'model': {'module_separator': {'parr': {'dilations': [1,2,4,8], 'dropout_rate': .05}}},
                  'optimizer': {'lr': .001}, 'evaluation': {'bootstrap_samples': 2000}}
        model.run_contract = make_run_contract(model, config)
        with tempfile.TemporaryDirectory() as directory:
            save_latest_checkpoint(1, 1, 1, model, torch.optim.AdamW(model.parameters()), directory)
            payload = torch.load(Path(directory)/'latest.pth')
            validate_resume_initialization(payload, model, 'latest.pth')
            for field, value in (('dilations',[1,2,4,16]),('dropout_rate',.2)):
                changed = copy.deepcopy(config)
                changed['model']['module_separator']['parr'][field] = value
                with self.assertRaisesRegex(RuntimeError, 'run contract differs'):
                    validate_run_contract(payload, make_run_contract(model, changed), 'latest.pth')
            changed = copy.deepcopy(config)
            changed['optimizer']['lr'] = .002
            with self.assertRaises(RuntimeError):
                validate_run_contract(payload, make_run_contract(model, changed), 'latest.pth')
            # Evaluation options are allowed to change, never the architecture.
            changed = copy.deepcopy(config)
            changed['evaluation']['bootstrap_samples'] = 0
            validate_run_contract(payload, make_run_contract(model, changed), 'latest.pth', evaluation=True)
            with self.assertRaisesRegex(RuntimeError, 'lacks a run contract'):
                validate_run_contract({}, model.run_contract, 'legacy.pth')


class EngineLoopFixTests(unittest.TestCase):
    def test_weighted_validation_and_warmup_across_epochs(self):
        class Toy(torch.nn.Module):
            num_stages = 1
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))
            def forward(self, waveform, **kwargs):
                output = [self.weight * waveform, self.weight * waveform]
                return output, [output]

        class Loss:
            def prepare_targets(self, *args):
                return None
            def __call__(self, **kwargs):
                return kwargs['estims'][0].mean()

        def batch(values):
            signal = torch.tensor(values, dtype=torch.float32)[:, None].expand(-1, 8)
            return torch.full((len(values),), 8), signal, [signal, signal], ['key']*len(values)

        for name in ('SepReformer_Base_Libri2Mix_16K', 'SepReformer_PARR_Libri2Mix_16K'):
            module = importlib.import_module(f'models.{name}.engine')
            engine = object.__new__(module.Engine.__wrapped__)
            engine.model = Toy()
            engine.device, engine.gpuid = torch.device('cpu'), ()
            engine.config = {'engine': {'mvn': False, 'clip_norm': 5}, 'model': {'num_spks': 2}}
            engine.PIT_SISNR_mag_loss = engine.PIT_SISNR_time_loss = Loss()
            engine.main_optimizer = torch.optim.AdamW(engine.model.parameters(), lr=.004)
            engine.warmup_scheduler = WarmupConstantSchedule(engine.main_optimizer, 4)
            with patch.object(module, 'tqdm'), patch.object(torch.nn.parallel, 'data_parallel',
                    side_effect=lambda model, waveform, **kw: model(waveform, **kw.get('module_kwargs', {}))):
                time_loss, spectral_loss, count = engine._validate([batch([1,3]), batch([9])])
                self.assertAlmostEqual(time_loss, 13/6)
                self.assertAlmostEqual(spectral_loss, 13/6)
                self.assertEqual(count, 2)
                engine._train([batch([1,3])], 1)
                engine._train([batch([1,3])], 2)
            self.assertEqual(engine.warmup_scheduler.last_epoch, 2)
            self.assertAlmostEqual(engine.main_optimizer.param_groups[0]['lr'], .003)


class ModelBoundaryFixTests(unittest.TestCase):
    def test_length_aware_forward_and_backward(self):
        for name in ('SepReformer_Base_Libri2Mix_16K', 'SepReformer_PARR_Libri2Mix_16K'):
            config = yaml.safe_load((Path('models')/name/'configs.yaml').read_text())['config']
            model = importlib.import_module(f'models.{name}.model').Model(**config['model']).eval()
            waveform = torch.randn(3, 2048, requires_grad=True)
            lengths = torch.tensor([1536, 2048, 1536])
            together, aux = model(waveform, input_sizes=lengths)
            with torch.inference_mode():
                for index, length in enumerate(lengths):
                    single, single_aux = model(waveform[index:index+1, :length])
                    for grouped, alone in zip(together + [t for stage in aux for t in stage],
                                              single + [t for stage in single_aux for t in stage]):
                        torch.testing.assert_close(grouped[index:index+1, :length], alone, rtol=1e-4, atol=1e-5)
                        self.assertEqual(torch.count_nonzero(grouped[index, length:]).item(), 0)
            sum(t.square().mean() for t in together).backward()
            self.assertTrue(torch.isfinite(waveform.grad).all())
            self.assertEqual(torch.count_nonzero(waveform.grad[0, 1536:]).item(), 0)
            self.assertTrue(all(module.attn is None for module in model.modules() if hasattr(module, 'attn')))
            model.train()
            # Each group has B=1, including one shorter than a waveform kernel.
            small = torch.randn(2, 512, requires_grad=True)
            out, _ = model(small, return_aux=False, input_sizes=torch.tensor([8, 512]))
            sum(t.square().mean() for t in out).backward()
            self.assertTrue(torch.isfinite(small.grad).all())

    def test_model_preserves_short_and_unaligned_lengths(self):
        for name in ('SepReformer_Base_Libri2Mix_16K', 'SepReformer_PARR_Libri2Mix_16K'):
            config = yaml.safe_load((Path('models')/name/'configs.yaml').read_text())['config']
            model = importlib.import_module(f'models.{name}.model').Model(**config['model']).eval()
            for length in (8, 31, 2049):
                with self.subTest(model=name, length=length), torch.inference_mode():
                    output, aux = model(torch.randn(1, length))
                    for tensor in output + [t for stage in aux for t in stage]:
                        self.assertEqual(tensor.shape, (1, length))
                        self.assertTrue(torch.isfinite(tensor).all())

    def test_validation_uses_individual_utterances(self):
        for name in ('SepReformer_Base_Libri2Mix_16K', 'SepReformer_PARR_Libri2Mix_16K'):
            module = importlib.import_module(f'models.{name}.dataset')
            with patch.object(module, 'Libri2MixDataset', return_value=list(range(4))):
                loaders = module.get_dataloaders(SimpleNamespace(engine_mode='train'), {},
                                                 dict(batch_size=2, pin_memory=False, num_workers=0, drop_last=False))
            self.assertEqual(loaders['train'].batch_size, 2)
            self.assertEqual(loaders['valid'].batch_size, 1)
            self.assertEqual(loaders['test'].batch_size, 1)


if __name__ == '__main__':
    logger.remove()
    torch.set_num_threads(2)
    unittest.main()
