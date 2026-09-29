"""Fail before training when configured evaluation/profiling imports are broken."""
import importlib
from importlib.metadata import version


def check_dependencies(config):
    modules = {
        'torch': 'torch', 'torchaudio': 'torchaudio', 'numpy': 'numpy',
        'soundfile': 'soundfile', 'librosa': 'librosa', 'scipy': 'scipy',
        'loguru': 'loguru', 'PyYAML': 'yaml', 'tqdm': 'tqdm',
        'tensorboard': 'torch.utils.tensorboard', 'mir-eval': 'mir_eval',
    }
    evaluation = config.get('evaluation', {})
    if evaluation.get('compute_pesq', True):
        modules['pesq'] = 'pesq'
    if evaluation.get('compute_stoi', True) or evaluation.get('compute_estoi', True):
        modules['pystoi'] = 'pystoi'
    for name in config.get('check_computations', {}).get('metrics', []):
        modules[name] = name
    packages, failures = {}, {}
    for distribution, module in modules.items():
        try:
            importlib.import_module(module)
            packages[distribution] = version(distribution)
        except Exception as error:
            failures[distribution] = f'{type(error).__name__}: {error}'
    if failures:
        raise RuntimeError(f'Runtime dependencies unavailable: {failures}. '
                           'Install compatible PyTorch packages and requirements-runtime.txt.')
    # Importing a package alone does not exercise its required API/configuration.
    from utils.evaluation_metrics import SeparationMetricsEvaluator
    SeparationMetricsEvaluator(
        sampling_rate=config['dataset']['sampling_rate'],
        num_spks=config['model']['num_spks'],
        **{key: value for key, value in evaluation.items() if key != 'warmup_first_utterance'},
    )
    return packages
