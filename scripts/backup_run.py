"""Create/verify a run archive on local or mounted persistent storage (stdlib only)."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(archive):
    archive = Path(archive)
    expected = archive.with_suffix(archive.suffix + '.sha256').read_text().split()[0]
    if digest_file(archive) != expected:
        raise ValueError('Archive SHA-256 mismatch')
    with zipfile.ZipFile(archive) as source:
        manifest = json.loads(source.read('SHA256.json'))
        if set(source.namelist()) != set(manifest) | {'SHA256.json'}:
            raise ValueError('Archive inventory differs from manifest')
        for name, expected in manifest.items():
            digest = hashlib.sha256()
            with source.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f'File SHA-256 mismatch: {name}')
    return len(manifest)


def backup(run, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    if not run.is_dir() or not (run / 'config.yaml').is_file():
        raise ValueError('Expected a run directory containing config.yaml')
    if destination.is_relative_to(run):
        raise ValueError('Backup destination must be outside the run directory')
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    files = sorted(p for p in run.rglob('*') if p.is_file() and not p.name.endswith('.tmp'))
    signatures = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
    manifest = {}
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_STORED) as archive:
        for path in files:
            name = path.relative_to(run).as_posix()
            digest = hashlib.sha256()
            with path.open('rb') as source, archive.open(name, 'w', force_zip64=True) as target:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
                    target.write(block)
            manifest[name] = digest.hexdigest()
        if any((p.stat().st_size, p.stat().st_mtime_ns) != sig for p, sig in signatures.items()):
            raise RuntimeError('Run changed during backup; retry at a quiet epoch boundary. Partial archive retained.')
        if set(files) != {p for p in run.rglob('*') if p.is_file() and not p.name.endswith('.tmp')}:
            raise RuntimeError('Run inventory changed during backup; retry at a quiet epoch boundary.')
        archive.writestr('SHA256.json', json.dumps(manifest, indent=2))
    temporary.rename(destination)
    destination.with_suffix(destination.suffix + '.sha256').write_text(
        digest_file(destination) + '  ' + destination.name + '\n', encoding='utf-8')
    verify(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    create.add_argument('run')
    create.add_argument('destination', help='New .zip path on persistent storage')
    check = sub.add_parser('verify')
    check.add_argument('archive')
    args = parser.parse_args()
    if args.command == 'create':
        print(backup(args.run, args.destination))
    else:
        print(f'Verified {verify(args.archive)} files')


if __name__ == '__main__':
    main()
