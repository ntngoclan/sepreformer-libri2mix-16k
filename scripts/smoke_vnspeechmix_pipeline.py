"""Exercise real VN audio/model/Engine, resume, metrics, comparison and backup.

CPU mode substitutes ONLY CUDA dispatch/timing calls; it is not GPU validation.
Uses tiny subsets and separate initialization artifacts, never official runs.
"""
import argparse
import contextlib
import copy
import gc
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from loguru import logger

from scripts.backup_run import backup, verify
from utils.compare_separation_metrics import compare, _write_markdown
from utils.evaluation_metrics import json_safe
from utils.run_management import prepare_run, record_provenance
from utils.runtime_dependencies import check_dependencies
from utils.util_implement import CriterionFactory, OptimizerFactory, SchedulerFactory
from utils.util_system import set_random_seed

MODELS = ('SepReformer_Base_VnSpeechMix_16K', 'SepReformer_LTRR_VnSpeechMix_16K')


@contextlib.contextmanager
def dispatch_context(device):
    with contextlib.ExitStack() as stack:
        if device.type == 'cpu':
            stack.enter_context(patch('torch.nn.parallel.data_parallel',
                side_effect=lambda model, inputs, device_ids=None, module_kwargs=None:
                    model(inputs, **(module_kwargs or {}))))
            stack.enter_context(patch('torch.cuda.device', side_effect=lambda *a: contextlib.nullcontext()))
            for name in ('synchronize', 'reset_peak_memory_stats'):
                stack.enter_context(patch(f'torch.cuda.{name}', return_value=None))
            stack.enter_context(patch('torch.cuda.max_memory_allocated', return_value=0))
            stack.enter_context(patch('torch.cuda.get_device_name', return_value='CPU smoke (CUDA boundary substituted)'))
        yield


def loaders(module, config, workers, mode):
    result = {}
    for split, count in [('train', 2), ('valid', 1), ('test', 2)]:
        if mode != 'train' and split != 'test':
            continue
        dataset = module.VnSpeechMixDataset(config['dataset'], split)
        dataset.examples = dataset.examples[:count]
        result[split] = torch.utils.data.DataLoader(
            dataset, batch_size=2 if split == 'train' else 1,
            shuffle=split == 'train', num_workers=workers,
            collate_fn=module._collate, worker_init_fn=module._seed_worker,
            generator=torch.Generator().manual_seed(0))
    return result


def run_one(name, config, output, device, workers, run_id, mode='train', resume=None,
            checkpoint=None):
    main = importlib.import_module(f'models.{name}.main')
    dataset = importlib.import_module(f'models.{name}.dataset')
    args = SimpleNamespace(model=name, engine_mode=mode, seed=0, run_id=run_id,
        output_dir=str(output / 'runs'), resume=str(resume) if resume else None,
        checkpoint=str(checkpoint) if checkpoint else None, out_wav_dir=None)
    config = copy.deepcopy(config)
    run = prepare_run(args, config)
    set_random_seed(**config['experiment'])
    data = loaders(dataset, config, workers, mode)
    model = main.Model(**config['model'])
    if mode == 'train':
        main.apply_paired_initialization(model, name, config, ROOT)
        set_random_seed(**config['experiment'])
    model.to(device)
    criteria, optimizers, schedulers = [], [], []
    if mode == 'train':
        criteria = CriterionFactory(config['criterion'], device).get_criterions()
        optimizers = OptimizerFactory(config['optimizer'], model.parameters()).get_optimizers()
        schedulers = SchedulerFactory(config['scheduler'], optimizers).get_schedulers()
    with dispatch_context(device):
        engine = main.Engine(args, config, model, data, criteria, optimizers, schedulers,
            (0,) if device.type == 'cuda' else (), device, work_dir=str(run))
        # Record real machine metadata outside the substituted CUDA context below.
        engine.run()
    record_provenance(run, args, model, data)
    del engine, model, data, criteria, optimizers, schedulers
    gc.collect()
    return run


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right, (left, right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New, isolated directory')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--samples', type=int, default=2048)
    parser.add_argument('--skip-pesq', action='store_true', help='Explicit partial metric check')
    parser.add_argument('--compact-artifacts', action='store_true',
                        help='After resume verification retain only the resumed best checkpoint')
    args = parser.parse_args()
    if args.samples < 2048 or args.samples > 64000:
        parser.error('--samples must be between 2048 and 64000')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    logger.remove()
    logger.add(str(output / 'engine.log'), level='INFO')
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    report = {'status': 'running', 'device': args.device, 'models': {},
        'limitations': ['Two train mixtures, one validation mixture, two full test mixtures.',
                        'Two smoke epochs do not establish convergence or model superiority.']}
    if device.type == 'cpu':
        report['limitations'].append('CUDA dispatch/timing substituted; GPU memory and CUDA kernels untested.')
    if args.skip_pesq:
        report['limitations'].append('PESQ explicitly disabled; full evaluation readiness NOT established.')
    if args.samples != 64000:
        report['limitations'].append(f'Train/validation crop reduced to {args.samples} samples.')
    metrics = []
    identities = []
    try:
        for name in MODELS:
            print(name, 'train / resume / evaluate', flush=True)
            config = yaml.safe_load((ROOT / 'models' / name / 'configs.yaml').read_text())['config']
            config['dataset']['max_len'] = args.samples
            config['dataloader']['num_workers'] = args.workers
            config['paired_initialization']['directory'] = str(output / 'initializations')
            config['engine'].update(max_epoch=2, checkpoint_epochs=[1, 2], diagnostic_examples=1,
                                    start_scheduling=0)
            config['scheduler']['WarmupConstantSchedule']['warmup_steps'] = 1
            config['check_computations'] = {'dummy_len': args.samples, 'metrics': []}
            config['evaluation'].update(bootstrap_samples=20, compute_pesq=not args.skip_pesq)
            packages = check_dependencies(config)
            original = run_one(name, config, output, device, args.workers, 'original')
            resumed = run_one(name, config, output, device, args.workers, 'resumed',
                              resume=original / 'checkpoints/epoch_0001.pth')
            expected = torch.load(original / 'checkpoints/latest.pth', map_location='cpu')
            actual = torch.load(resumed / 'checkpoints/latest.pth', map_location='cpu')
            assert actual['epoch'] == expected['epoch'] == 2
            if device.type == 'cpu':
                for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dicts', 'valid_loss'):
                    assert_tree_equal(expected[key], actual[key])
            identities.append(actual['paired_initialization_metadata']['initialization_sha256'])
            del expected, actual
            if args.compact_artifacts:
                for run in (original, resumed):
                    removed = []
                    for path in (run / 'checkpoints').glob('*.pth'):
                        if run == resumed and path.name == 'best.pth':
                            continue
                        if not path.resolve().is_relative_to(output):
                            raise ValueError('Smoke checkpoint path escaped output directory')
                        path.unlink()
                        removed.append(path.name)
                    (run / 'smoke_pruned_checkpoints.json').write_text(json.dumps({
                        'reason': 'Exact CPU resume checked before pruning; compact smoke artifacts only',
                        'removed': removed}, indent=2), encoding='utf-8')
            evaluated = run_one(name, config, output, device, args.workers, 'evaluation',
                mode='test_save', checkpoint=resumed / 'checkpoints/best.pth')
            csv_path = next(evaluated.glob('evaluation/*/metrics_per_utterance.csv'))
            summary = json.loads(csv_path.with_name('metrics_summary.json').read_text())
            assert summary['num_utterances'] == 2
            metrics.append(csv_path)
            archives = []
            for run in (original, resumed, evaluated):
                archive = backup(run, output / 'backups' / f'{name}_{run.parent.name if run.name.startswith("eval_") else run.name}.zip')
                archives.append({'path': str(archive), 'verified_files': verify(archive)})
            report['models'][name] = {'resume': 'exact' if device.type == 'cpu' else 'loaded (CUDA exactness not asserted)',
                'packages': packages, 'train': str(original), 'resume_run': str(resumed),
                'evaluation': str(evaluated), 'backups': archives,
                'compact_smoke_artifacts': args.compact_artifacts}
            print(name, 'passed', flush=True)
        assert identities[0] == identities[1], 'Backbone initialization differs'
        result = compare(*metrics, bootstrap_samples=100, confidence=.95, seed=0, expected_count=2)
        required = {'si_snri', 'bss_sdri', 'stoi', 'estoi'}
        if not args.skip_pesq:
            required.add('pesq_wb')
        available = {r['metric'] for r in result['results'] if r['n_pairs'] == 2}
        assert required <= available, f'Missing/nonfinite metric pairs: {required - available}'
        (output / 'paired_comparison.json').write_text(json.dumps(json_safe(result), indent=2, allow_nan=False), encoding='utf-8')
        _write_markdown(result, output / 'paired_comparison.md')
        report['paired_initialization_sha256'] = identities[0]
        report['status'] = 'partial_smoke_passed' if args.skip_pesq or device.type == 'cpu' else 'smoke_passed'
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(report['status'], output / 'report.json', flush=True)


if __name__ == '__main__':
    main()
