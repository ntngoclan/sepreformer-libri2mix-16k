"""Validate Libri2Mix wav16k/min triplets and fingerprint their file contents."""
import argparse
import csv
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf
import yaml
from models.SepReformer_Base_Libri2Mix_16K.dataset import Libri2MixDataset


def build_manifest(config, replace=False):
    cfg = config['dataset']
    path = Path(cfg['manifest'])
    if not path.is_absolute():
        path = ROOT / path
    rows = []
    for split in ('train', 'valid', 'test'):
        ds = Libri2MixDataset(cfg, split)
        try:
            expected = cfg['expected_sizes'][split]
            if len(ds) != expected:
                raise ValueError(f'{split}: {len(ds)} mixtures, expected {expected}')
            # Reject unmatched sources as well as missing sources.
            keys = {item[0] for item in ds.examples}
            for source in ds.source_dirs:
                if ds.storage == 'directory':
                    actual = {p.name for p in (ds.extracted_root / ds.partition_dir / source).glob('*.wav')}
                else:
                    prefix = f'{ds.archive_root / ds.partition_dir / source}/'
                    actual = {name[len(prefix):] for name in ds._get_zip().namelist()
                              if name.startswith(prefix) and name.endswith('.wav')}
                if actual != keys:
                    raise ValueError(f'{split}/{source}: source keys do not match mixtures')
            for index, (key, mix, sources) in enumerate(ds.examples):
                hashes, lengths = [], []
                for reference in (mix, *sources):
                    if ds.storage == 'zip':
                        payload = ds._get_zip().read(reference)  # includes ZIP CRC verification
                        info = sf.info(io.BytesIO(payload))
                        digest = hashlib.sha256(payload).hexdigest()
                    else:
                        info = sf.info(reference)
                        h = hashlib.sha256()
                        with open(reference, 'rb') as stream:
                            for block in iter(lambda: stream.read(1024 * 1024), b''):
                                h.update(block)
                        digest = h.hexdigest()
                    if info.samplerate != cfg['sampling_rate'] or info.channels != 1 or info.frames <= 0:
                        raise ValueError(f'Invalid mono 16 kHz audio: {reference}: {info}')
                    hashes.append(digest)
                    lengths.append(info.frames)
                if len(set(lengths)) != 1:
                    raise ValueError(f'Mixture/source lengths differ: {split}/{key}: {lengths}')
                rows.append([split, key, lengths[0], cfg['sampling_rate'], *hashes])
                if (index + 1) % 1000 == 0:
                    print(f'{split}: {index + 1}/{len(ds)} triplets hashed', flush=True)
            print(f'{split}: {len(ds)} triplets OK ({ds.storage})', flush=True)
        finally:
            if ds._zip is not None:
                ds._zip.close()
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(['split', 'key', 'num_samples', 'sampling_rate',
                     'mixture_sha256', 's1_sha256', 's2_sha256'])
    writer.writerows(rows)
    content = buffer.getvalue().encode('utf-8')
    if path.exists() and path.read_bytes() != content and not replace:
        raise ValueError(f'{path} differs from this dataset. Preserve the old manifest; use --replace only for an intentional new dataset.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix='.tmp') as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Manifest: {path}\nSHA-256: {hashlib.sha256(content).hexdigest()}')
    return path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='models/SepReformer_Base_Libri2Mix_16K/configs.yaml')
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()
    build_manifest(yaml.safe_load((ROOT / args.config).read_text(encoding='utf-8'))['config'], args.replace)


if __name__ == '__main__':
    main()
