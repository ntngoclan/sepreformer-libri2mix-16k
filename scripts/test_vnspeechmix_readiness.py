"""Regression checks for VN archive workers, preparation and fail-fast dependencies."""
import copy
import importlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import yaml

from scripts.prepare_vnspeechmix_manifest import prepare
from utils.runtime_dependencies import check_dependencies

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('SepReformer_Base_VnSpeechMix_16K', 'SepReformer_LTRR_VnSpeechMix_16K')


class ReadinessTests(unittest.TestCase):
    def test_manifest_preserves_bytes_and_rejects_overwrite(self):
        payload = b'mixture_id,split\r\n' + b''.join(
            f'{split}_{i},{split}\r\n'.encode()
            for split, count in [('train', 18000), ('valid', 3000), ('test', 5000)]
            for i in range(count))
        with tempfile.TemporaryDirectory() as directory:
            archive, destination = Path(directory) / 'data.zip', Path(directory) / 'manifest.csv'
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('rendered/rendered_metadata.csv', payload)
            first = prepare(archive, destination)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(first, prepare(archive, destination))
            destination.write_bytes(b'existing user manifest')
            with self.assertRaises(FileExistsError):
                prepare(archive, destination)
            self.assertEqual(destination.read_bytes(), b'existing user manifest')

    def test_zip_reopened_after_process_change_and_pickle_has_no_handle(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'data.zip'
            audio = io.BytesIO()
            sf.write(audio, np.linspace(-.2, .2, 2048), 16000, format='WAV')
            with zipfile.ZipFile(archive, 'w') as stream:
                for folder in ('mix_clean', 's1', 's2'):
                    stream.writestr(f'rendered/train/{folder}/a.wav', audio.getvalue())
            for name in MODELS:
                with self.subTest(model=name):
                    config = yaml.safe_load((ROOT / 'models' / name / 'configs.yaml').read_text())['config']['dataset']
                    config.update(archive=str(archive), extracted_dir=str(Path(directory) / 'missing'))
                    module = importlib.import_module(f'models.{name}.dataset')
                    dataset = module.VnSpeechMixDataset(config, 'train')
                    first = dataset[0]['mix']
                    handle = dataset._zip
                    with patch.object(module.os, 'getpid', return_value=-1):
                        second = dataset[0]['mix']
                        self.assertIsNot(handle, dataset._zip)
                        self.assertIsNone(handle.fp)
                    np.testing.assert_array_equal(first, second)
                    self.assertIsNone(dataset.__getstate__()['_zip'])
                    dataset._zip.close()

    def test_missing_pesq_fails_before_training(self):
        config = yaml.safe_load((ROOT / 'models' / MODELS[0] / 'configs.yaml').read_text())['config']
        real_import = importlib.import_module

        def without_pesq(name):
            if name == 'pesq':
                raise ImportError('simulated missing pesq')
            return real_import(name)

        with patch('utils.runtime_dependencies.importlib.import_module', side_effect=without_pesq):
            with self.assertRaisesRegex(RuntimeError, 'pesq'):
                check_dependencies(config)
            partial = copy.deepcopy(config)
            partial['evaluation']['compute_pesq'] = False
            self.assertNotIn('pesq', check_dependencies(partial))

    def test_entrypoint_checks_dependencies_before_creating_run(self):
        from types import SimpleNamespace
        for name in MODELS:
            module = importlib.import_module(f'models.{name}.main')
            with patch.object(module, 'check_dependencies', side_effect=RuntimeError('missing dependency')):
                with patch.object(module, 'prepare_run') as create:
                    with self.assertRaisesRegex(RuntimeError, 'missing dependency'):
                        module.main(SimpleNamespace(config=None))
                    create.assert_not_called()


if __name__ == '__main__':
    unittest.main()
