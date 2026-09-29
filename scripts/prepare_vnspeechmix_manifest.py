"""Copy the published manifest byte-for-byte from rendered.zip; leave audio in ZIP."""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path


def prepare(archive, destination):
    destination = Path(destination)
    with zipfile.ZipFile(archive) as source:
        member = 'rendered/rendered_metadata.csv'
        if source.namelist().count(member) != 1:
            raise ValueError('Expected exactly one published rendered manifest')
        payload = source.read(member)
    rows = list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))))
    counts = {}
    for split, expected in [('train', 18000), ('valid', 3000), ('test', 5000)]:
        keys = [r['mixture_id'] for r in rows if r['split'] == split]
        if len(keys) != expected or len(set(keys)) != expected:
            raise ValueError(f'Invalid published {split} count or duplicate keys')
        counts[split] = len(keys)
    if len(rows) != sum(counts.values()):
        raise ValueError('Unexpected split in manifest')
    if destination.exists():
        if destination.read_bytes() != payload:
            raise FileExistsError(f'Refusing to replace a different manifest: {destination}')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(payload)
    return {'manifest': str(destination), 'sha256': hashlib.sha256(payload).hexdigest(),
            'counts': counts, 'audio_storage': 'zip'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', default='data/VnSpeechMix/rendered.zip')
    parser.add_argument('--output', default='data/VnSpeechMix/rendered/rendered_metadata.csv')
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.output), indent=2))
