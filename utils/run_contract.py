"""Checkpoint identity for a complete run, separate from shared epoch-0 weights."""

import copy
import hashlib
import json
from pathlib import Path


PIPELINE_REVISION = "length-aware-spectral-pit-v2"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_run_contract(model, config):
    root = Path(__file__).resolve().parents[1]
    module_path = root.joinpath(*type(model).__module__.split('.')).with_suffix('.py')
    source_files = [root / name for name in (
        'run.py', 'utils/util_engine.py', 'utils/util_system.py',
        'utils/paired_initialization.py', 'utils/run_contract.py',
        'utils/run_management.py', 'utils/parr_diagnostics.py',
        'utils/util_implement.py', 'utils/functions.py', 'utils/evaluation_metrics.py',
        'utils/decorators.py', 'utils/implements/optimizers.py')]
    source_files += [root / 'utils/implements/criterions.py',
                    root / 'utils/implements/schedulers.py', root / 'utils/runtime_state.py']
    if module_path.is_file():
        source_files += [module_path, module_path.parent / 'dataset.py', module_path.parent / 'engine.py',
                         module_path.parent / 'main.py']
        source_files += sorted((module_path.parent / 'modules').glob('*.py'))
    # A Windows CRLF checkout and Linux LF checkout must have the same identity.
    sources = {str(path.relative_to(root)).replace('\\', '/'):
               hashlib.sha256(path.read_text(encoding='utf-8-sig').encode('utf-8')).hexdigest()
               for path in source_files if path.is_file()}
    architecture = {'model_class': type(model).__module__ + '.' + type(model).__name__,
                    'model_config': config.get('model', {})}
    manifest = config.get('dataset', {}).get('manifest')
    dataset_identity = None
    if manifest:
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path
        dataset_identity = {'kind': 'manifest_sha256', 'sha256': file_sha256(manifest_path)}
    return {
        'dataset_identity': dataset_identity,
        'pipeline_revision': PIPELINE_REVISION,
        'full_training_config_sha256': _digest(config),
        'architecture_sha256': _digest(architecture),
        'source_sha256': _digest(sources),
        'source_files': sources,
        'config': copy.deepcopy(config),
    }


def validate_run_contract(checkpoint, expected, checkpoint_path, evaluation=False):
    actual = checkpoint.get('run_contract')
    if actual is None:
        raise RuntimeError(f"Checkpoint {checkpoint_path} lacks a run contract; its configuration/loss "
                           "cannot be verified. Start a new run or use it explicitly as model-only initialization.")
    fields = ['architecture_sha256', 'source_sha256', 'pipeline_revision']
    if not evaluation:
        fields.append('full_training_config_sha256')
        fields.append('dataset_identity')
    mismatches = [key for key in fields if actual.get(key) != expected.get(key)]
    if mismatches:
        raise RuntimeError(f"Checkpoint {checkpoint_path} run contract differs: {mismatches}. "
                           "Do not resume/evaluate with changed architecture or training semantics.")
