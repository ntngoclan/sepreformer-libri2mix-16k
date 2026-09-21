"""Explicit run directories and portable provenance for long experiments."""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from utils.run_contract import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def command_output(command):
    try:
        return subprocess.check_output(command, cwd=ROOT, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", timeout=60).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"


def prepare_run(args, config):
    """Never infer training resume from files left by an earlier experiment."""
    seed = getattr(args, 'seed', None)
    if seed is not None:
        config['experiment']['seed'] = seed
        config['dataloader']['seed'] = seed
    resume = getattr(args, 'resume', None)
    checkpoint = getattr(args, 'checkpoint', None)
    if resume and args.engine_mode != 'train':
        raise ValueError('--resume is only for training; use --checkpoint for evaluation.')
    if checkpoint and args.engine_mode == 'train':
        raise ValueError('--checkpoint is only for evaluation; use --resume for training.')
    selected = resume or checkpoint
    if selected and not Path(selected).is_file():
        raise FileNotFoundError(selected)
    seed = int(config['experiment']['seed'])
    run_id = getattr(args, 'run_id', None)
    if run_id and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise ValueError('run_id must be a single safe directory name.')
    output = Path(getattr(args, 'output_dir', None) or 'runs').resolve()
    if args.engine_mode != 'train':
        if checkpoint:
            # Every evaluation has a separate directory, even for the same epoch.
            run = output / args.model / f'seed_{seed:04d}' / (run_id or 'evaluations')
            run = run / datetime.now(timezone.utc).strftime('eval_%Y%m%dT%H%M%S_%fZ')
        elif run_id:
            parent = output / args.model / f'seed_{seed:04d}' / run_id
            args.checkpoint = str(parent / 'checkpoints/best.pth')
            if not Path(args.checkpoint).is_file():
                raise FileNotFoundError(args.checkpoint)
            run = parent / 'evaluations' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        else:
            raise ValueError('Evaluation requires --checkpoint or --run-id (selects best.pth).')
    else:
        run_id = run_id or datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
        run = output / args.model / f'seed_{seed:04d}' / run_id
    # Resume forks into a NEW output directory. Existing runs are immutable here.
    run.mkdir(parents=True, exist_ok=False)
    for name in ('checkpoints', 'tensorboard', 'logs'):
        (run / name).mkdir()
    (run / 'config.yaml').write_text(yaml.safe_dump({'config': config}, sort_keys=False), encoding='utf-8')
    args.run_dir = str(run)
    return run


def record_provenance(run, args, model, dataloaders=None):
    run = Path(run)
    status = command_output(['git', 'status', '--porcelain', '--untracked-files=normal'])
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'arguments': vars(args),
        'git_commit': command_output(['git', 'rev-parse', 'HEAD']),
        'git_dirty': bool(status), 'git_status': status,
        'python': sys.version, 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'gpu': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'driver': command_output(['nvidia-smi', '--query-gpu=name,driver_version,uuid', '--format=csv,noheader']),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'deterministic_warn_only': True,
        'determinism_limit': 'CUDA adaptive_avg_pool backward can remain nondeterministic; compare multiple seeds.',
        'paired_initialization': getattr(model, 'paired_initialization_metadata', None),
        'run_contract': model.run_contract,
    }
    if dataloaders and 'valid' in dataloaders:
        dataset = dataloaders['valid'].dataset
        count = model.run_contract['config']['engine'].get('diagnostic_examples', 4)
        metadata['diagnostic_validation_keys'] = [item[0] for item in dataset.examples[:count]]
    selected = getattr(args, 'resume', None) or getattr(args, 'checkpoint', None)
    if selected:
        metadata['input_checkpoint_sha256'] = file_sha256(selected)
    (run / 'provenance.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n', encoding='utf-8')
    (run / 'environment.txt').write_text(command_output([sys.executable, '-m', 'pip', 'freeze']) + '\n', encoding='utf-8')
    (run / 'source.patch').write_text(command_output(['git', 'diff', 'HEAD', '--no-ext-diff']), encoding='utf-8')
    # Capture every fingerprinted file, including untracked new implementation files.
    for relative in model.run_contract['source_files']:
        destination = run / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
