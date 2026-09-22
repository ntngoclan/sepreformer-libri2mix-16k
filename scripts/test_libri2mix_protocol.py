"""Shared protocol, Libri2Mix data integrity, pairing and real-update resume tests."""
import copy
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf
import torch
import yaml

from scripts import test_ltrr as reference_tests
from scripts.prepare_libri2mix_manifest import build_manifest
from models.SepReformer_Base_Libri2Mix_16K.dataset import Libri2MixDataset

ROOT = Path(__file__).resolve().parents[1]
BASE = 'SepReformer_Base_Libri2Mix_16K'
LTRR = 'SepReformer_LTRR_Libri2Mix_16K'


def config(name):
    return yaml.safe_load((ROOT / 'models' / name / 'configs.yaml').read_text())['config']


class LibriLTRRTests(reference_tests.LTRRTests):
    """Run the same architecture/gradient/paired-init/resume checks for English."""
    @classmethod
    def setUpClass(cls):
        pairing = importlib.import_module(f'models.{LTRR}.modules.pairing')
        cls.patches = patch.multiple(reference_tests,
            BASELINE=BASE, CANDIDATE=LTRR,
            Baseline=importlib.import_module(f'models.{BASE}.model').Model,
            Model=importlib.import_module(f'models.{LTRR}.model').Model,
            LTRR=importlib.import_module(f'models.{LTRR}.modules.ltrr').LTRR,
            apply_paired_initialization=pairing.apply_paired_initialization,
            baseline_artifact_config=pairing.baseline_artifact_config,
            make_run_contract=pairing.make_run_contract,
            _load_training_resume=importlib.import_module(f'models.{LTRR}.engine')._load_training_resume)
        cls.patches.start()
        cls.addClassCleanup(cls.patches.stop)
        super().setUpClass()


class ProtocolTests(unittest.TestCase):
    def test_four_configs_share_training_protocol(self):
        normalized = []
        for name in (BASE, LTRR, 'SepReformer_Base_VnSpeechMix_16K', 'SepReformer_LTRR_VnSpeechMix_16K'):
            cfg = config(name)
            cfg.pop('paired_initialization')
            cfg['model'].pop('module_ltrr', None)
            cfg['dataset'] = {k: cfg['dataset'][k] for k in
                              ('sampling_rate', 'max_len', 'sample_length_multiple', 'mixture_dir', 'source_dirs')}
            normalized.append(cfg)
        for cfg in normalized[1:]:
            self.assertEqual(cfg, normalized[0])

    def test_negative_plateau_reduces_lr(self):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.))], lr=.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config(BASE)['scheduler']['ReduceLROnPlateau'])
        self.assertFalse(scheduler.is_better(-10., -10.))
        self.assertFalse(scheduler.is_better(-9.9995, -10.))
        for _ in range(4):
            scheduler.step(-10.)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .0008)

    def test_real_wav_directory_zip_and_manifest(self):
        cfg = copy.deepcopy(config(BASE))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / 'Libri2Mix/wav16k/min'
            cfg['dataset'].update(extracted_dir=str(data), archive=str(root / 'data.zip'),
                                  manifest=str(root / 'manifest.csv'),
                                  expected_sizes=dict(train=2, valid=2, test=2), max_len=1024)
            for split in ('train-100', 'dev', 'test'):
                for i in range(2):
                    a = np.linspace(-.1, .1, 2051, dtype=np.float32)
                    b = np.cos(np.arange(2051)).astype(np.float32) * .1
                    for sub, signal in [('mix_clean', a+b), ('s1', a), ('s2', b)]:
                        path = data / split / sub / f'{i}.wav'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(path, signal, 16000, subtype='FLOAT')
            path = build_manifest(cfg)
            directory_manifest = path.read_bytes()
            valid = Libri2MixDataset(cfg['dataset'], 'valid')
            np.testing.assert_array_equal(valid[0]['mix'], valid[0]['mix'])
            self.assertEqual(valid[0]['num_sample'], 1024)
            self.assertEqual(Libri2MixDataset(cfg['dataset'], 'test')[0]['num_sample'], 2048)
            with zipfile.ZipFile(cfg['dataset']['archive'], 'w') as archive:
                for file in data.rglob('*.wav'):
                    archive.write(file, file.relative_to(root).as_posix())
            zipped_config = copy.deepcopy(cfg)
            zipped_config['dataset']['extracted_dir'] = str(root / 'not_extracted')
            self.assertEqual(build_manifest(zipped_config).read_bytes(), directory_manifest)
            zipped = Libri2MixDataset(zipped_config['dataset'], 'valid')
            try:
                np.testing.assert_array_equal(valid[0]['mix'], zipped[0]['mix'])
            finally:
                zipped.__del__()
            # Exercise the actual CLI against the Libri2Mix manifest schema,
            # real WAVs, optimizer and checkpoint serialization (no GPU needed).
            pilot = root / 'pilot.yaml'
            pilot.write_text(yaml.safe_dump({'config': cfg}), encoding='utf-8')
            report = root / 'preflight.json'
            completed = subprocess.run([sys.executable, '-B',
                str(ROOT / 'scripts/preflight_vnspeechmix_baseline.py'),
                '--root', str(ROOT), '--model', BASE, '--config', str(pilot),
                '--device', 'cpu', '--samples', '1024', '--steps', '1',
                '--workers', '0', '--report', str(report)],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(json.loads(report.read_text())['status'], 'smoke_passed')
            sf.write(data / 'dev/s1/0.wav', np.zeros(2051), 8000)
            with self.assertRaisesRegex(ValueError, 'Invalid mono 16 kHz'):
                build_manifest(cfg)
            (data / 'train-100/s2/0.wav').unlink()
            with self.assertRaises(FileNotFoundError):
                Libri2MixDataset(cfg['dataset'], 'train')


if __name__ == '__main__':
    from loguru import logger
    logger.remove()
    torch.set_num_threads(2)
    unittest.main()
