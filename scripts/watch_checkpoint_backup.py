"""Back up atomically saved VN-SpeechMix checkpoints with rclone (stdlib only).

Linux watcher; archives can be verified on Windows too. Recovery snapshots are
not full run archives: live logs, TensorBoard, audio and dataset ZIPs are omitted.
Retention removes only old run snapshots after verifying the retained copies.
No training files or configuration are changed. Support archives are retained.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile
import time
from datetime import datetime, timezone
import uuid
import zipfile


def log(message):
    print(datetime.now(timezone.utc).isoformat(timespec='seconds'), message, flush=True)


def signature(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory(root, files):
    result = {}
    for path in sorted(set(files)):
        path.resolve().relative_to(root.resolve())
        if path.is_symlink():
            raise ValueError(f'Symlink not supported: {path}')
        result[path.relative_to(root).as_posix()] = signature(path)
    return result


def make_archive(root, files, destination):
    """Read each file through one descriptor; atomic replacements remain safe.

    latest and best are independent records, possibly from different epochs.
    Detect in-place writes; never read a training checkpoint's .tmp file.
    """
    manifest = {}
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(set(files)):
            name = path.relative_to(root).as_posix()
            if path.name.endswith('.tmp'):
                raise ValueError(f'Temporary file refused: {path}')
            digest = hashlib.sha256()
            with path.open('rb') as source, archive.open(name, 'w', force_zip64=True) as target:
                before = os.fstat(source.fileno())
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    target.write(block)
                    digest.update(block)
                after = os.fstat(source.fileno())
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f'File changed in place during snapshot: {path}')
            manifest[name] = digest.hexdigest()
        archive.writestr('SHA256.json', json.dumps(manifest, sort_keys=True, indent=2))
    destination.with_suffix('.zip.sha256').write_text(
        sha256(destination) + '  ' + destination.name + '\n', encoding='utf-8')
    verify(destination)


def verify(archive_path):
    archive_path = Path(archive_path)
    expected = archive_path.with_suffix('.zip.sha256').read_text().split()[0]
    if sha256(archive_path) != expected:
        raise ValueError('Archive SHA-256 mismatch')
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read('SHA256.json'))
        if (len(archive.namelist()) != len(manifest) + 1
                or set(archive.namelist()) != set(manifest) | {'SHA256.json'}):
            raise ValueError('Archive inventory mismatch')
        for name, expected_hash in manifest.items():
            if name.startswith('/') or '..' in Path(name).parts or '\\' in name or ':' in name:
                raise ValueError(f'Unsafe archive path: {name}')
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected_hash:
                raise ValueError(f'File hash mismatch: {name}')
    return len(manifest)


def run_files(run):
    required = [run / name for name in ('config.yaml', 'provenance.json', 'environment.txt',
                                       'source.patch', 'checkpoints/latest.pth')]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    source = sorted((run / 'source').rglob('*.py'))
    if not source:
        raise ValueError(f'Missing source snapshot: {run}')
    best = run / 'checkpoints/best.pth'
    return required + source + ([best] if best.is_file() else [])


def support_files(root):
    # Explicit allowlist: never include .env, rclone.conf, screenshots or .git.
    files = [root / 'run.py', root / 'data/VnSpeechMix/rendered/rendered_metadata.csv']
    files += sorted(root.glob('requirements*.txt'))
    for folder in ('models', 'utils', 'scripts'):
        files += [p for p in (root / folder).rglob('*')
                  if p.is_file() and p.suffix in ('.py', '.sh', '.yaml', '.yml')]
    initializations = sorted((root / 'initializations/vnspeechmix_common_v2').glob('*.pth'))
    if not initializations:
        raise ValueError('Missing paired initialization .pth files')
    files += initializations
    files += sorted((root / 'initializations/vnspeechmix_common_v2').glob('*.json'))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return files


class Publisher:
    def __init__(self, root, work, remote, command, keep=0, prune_dry_run=False):
        if keep != 0 and keep < 2:
            raise ValueError('keep must be 0 (disabled) or at least 2')
        self.root, self.work = root, work
        self.remote, self.command = remote.rstrip('/'), command
        self.keep, self.prune_dry_run = keep, prune_dry_run
        self.work.mkdir(parents=True, exist_ok=True)
        self.state_path = self.work / 'state.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def invoke(self, *arguments, capture=False):
        result = subprocess.run(self.command + list(arguments) + ['--retries', '3', '--contimeout', '30s',
                                '--timeout', '5m'], check=True, timeout=3600,
                                stdout=subprocess.PIPE if capture else None,
                                text=True, encoding='utf-8')
        return result.stdout

    def listing(self, remote):
        items = json.loads(self.invoke('lsjson', remote, '--drive-skip-shortcuts', capture=True))
        names = [item['Name'] for item in items]
        if len(names) != len(set(names)):
            raise ValueError(f'Duplicate names on remote; refusing cleanup: {remote}')
        if any(item['Path'] != item['Name'] for item in items):
            raise ValueError(f'Unexpected nested listing; refusing cleanup: {remote}')
        return {item['Name']: item for item in items}

    def prune(self, key):
        # Never sweep the remote root, support archives, another run, or arbitrary paths.
        if not self.keep or key == 'support':
            return
        if not re.fullmatch(r'runs/SepReformer_(?:Base|LTRR)_VnSpeechMix_16K/'
                            r'seed_\d+/[A-Za-z0-9][A-Za-z0-9_.-]*', key):
            raise ValueError(f'Unsafe retention key: {key}')
        parent = self.remote + '/' + key
        record = self.state.get(parent, {})
        anchor_path = record.get('remote', '')
        anchor = anchor_path.removeprefix(parent + '/')
        pattern = r'\d{8}T\d{6}_\d{6}Z_[0-9a-f]{8}'
        if (not record.get('verified_utc') or anchor_path != parent + '/' + anchor
                or not re.fullmatch(pattern, anchor)):
            raise ValueError('No verified snapshot for this run; refusing cleanup')
        folders = self.listing(parent)
        if anchor not in folders or not folders[anchor]['IsDir']:
            raise ValueError('Latest verified snapshot is missing; refusing cleanup')
        names = sorted(name for name, item in folders.items()
                       if item['IsDir'] and re.fullmatch(pattern, name) and name <= anchor)
        if len(names) <= self.keep:
            return
        expected = {'snapshot.zip', 'snapshot.zip.sha256'}
        known, complete = {}, []
        for name in names:
            contents = self.listing(parent + '/' + name)
            if (set(contents) <= expected
                    and all(not item['IsDir'] for item in contents.values())):
                known[name] = contents
                if set(contents) == expected:
                    complete.append(name)
            else:
                log(f'PRUNE_SKIP unexpected contents: {parent}/{name}')
        if anchor not in complete or len(complete) < self.keep:
            raise ValueError('Not enough complete snapshots to retain; refusing cleanup')
        retained = complete[-self.keep:]
        # Partial leftovers from an interrupted deletion can be retried too.
        candidates = [name for name in known if name < retained[0]]
        if not candidates:
            return
        # Legacy backups have no remote verification marker. Download and verify
        # every retained archive, including the old state anchor, BEFORE deletion.
        for name in retained:
            remote = parent + '/' + name
            with tempfile.TemporaryDirectory(prefix='retain-', dir=self.work) as temp:
                self.invoke('copy', remote, temp, '--drive-skip-shortcuts', '--transfers', '1')
                archive = Path(temp) / 'snapshot.zip'
                verify(archive)
                with zipfile.ZipFile(archive) as snapshot:
                    manifest = json.loads(snapshot.read('SHA256.json'))
                if (key + '/checkpoints/latest.pth' not in manifest
                        or any(not path.startswith(key + '/') for path in manifest)):
                    raise ValueError(f'Archive belongs to a different run: {remote}')
            log(f'RETAIN_VERIFIED {remote}')
        for name in candidates:
            remote = parent + '/' + name
            if self.listing(remote) != known[name]:
                raise RuntimeError(f'Snapshot changed before deletion: {remote}')
            if self.prune_dry_run:
                log(f'PRUNE_WOULD_DELETE {remote}')
                continue
            # Exact filenames only; never purge directories or empty Drive trash.
            for filename in ('snapshot.zip', 'snapshot.zip.sha256'):
                if filename in known[name]:
                    self.invoke('deletefile', remote + '/' + filename, '--drive-use-trash=false')
            self.invoke('rmdir', remote, '--drive-use-trash=false')
            log(f'PRUNED {remote}; retained={self.keep}')

    def publish(self, key, files, stable=False):
        before = inventory(self.root, files)
        state_key = self.remote + '/' + key
        if self.state.get(state_key, {}).get('inventory') == before:
            self.prune(key)
            return False
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8]
        destination = state_key + '/' + stamp
        log(f'SNAPSHOT {key}')
        with tempfile.TemporaryDirectory(prefix='upload-', dir=self.work) as temp:
            staging = Path(temp)
            make_archive(self.root, files, staging / 'snapshot.zip')
            if stable and inventory(self.root, files) != before:
                raise RuntimeError('Support files changed during backup; retry next cycle')
            self.invoke('copy', str(staging), destination, '--immutable', '--transfers', '1')
            # Read the remote bytes back; only a successful check advances state.
            self.invoke('check', str(staging), destination, '--download', '--one-way', '--checkers', '1')
            self.state[state_key] = {'inventory': before, 'remote': destination,
                                     'verified_utc': datetime.now(timezone.utc).isoformat()}
            temporary = self.state_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.state, indent=2), encoding='utf-8')
            os.replace(temporary, self.state_path)
        log(f'VERIFIED {destination}')
        self.prune(key)
        return True


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/SepReformer'))
    parser.add_argument('--work', type=Path, default=Path('/workspace/checkpoint-backup-state'))
    parser.add_argument('--remote', default='vnbackup:SepReformer_Backups/recovery-v1')
    parser.add_argument('--rclone', default='/workspace/tools/rclone-v1.75.1-linux-amd64/rclone')
    parser.add_argument('--config', default='/workspace/backup-config/rclone.conf')
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--keep', type=int, default=3,
                        help='Retain N snapshots per run; minimum 2, or 0 to disable cleanup (default 3)')
    parser.add_argument('--prune-dry-run', action='store_true',
                        help='Back up normally, but only preview old snapshot deletions')
    parser.add_argument('--verify', type=Path, help='Verify a downloaded snapshot.zip and .sha256')
    args = parser.parse_args()
    if args.verify:
        print(f'Verified {verify(args.verify)} files')
        return
    if args.interval < 30:
        parser.error('--interval must be at least 30 seconds')
    if args.keep != 0 and args.keep < 2:
        parser.error('--keep must be 0 (disabled) or at least 2')
    if os.name != 'posix':
        parser.error('Run the watcher on the Linux Pod; --verify also works on Windows')
    import fcntl
    root, work = args.root.resolve(), args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'watcher.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another watcher is already running with this --work directory')
        if not Path(args.config).is_file():
            parser.error('rclone config not found: ' + args.config)
        publisher = Publisher(root, work, args.remote, [args.rclone, '--config', args.config],
                              keep=args.keep, prune_dry_run=args.prune_dry_run)
        while True:
            failures = 0
            try:
                publisher.publish('support', support_files(root), stable=True)
            except Exception as error:
                log(f'ERROR support: {error}')
                failures += 1
            runs = []
            for model in ('SepReformer_Base_VnSpeechMix_16K', 'SepReformer_LTRR_VnSpeechMix_16K'):
                runs += sorted((root / 'runs' / model).glob('seed_*/*/checkpoints/latest.pth'))
            if not runs:
                log('WAIT: no production checkpoint found')
                failures += 1
            for checkpoint in runs:
                run = checkpoint.parent.parent
                try:
                    publisher.publish(run.relative_to(root).as_posix(), run_files(run))
                except Exception as error:
                    log(f'ERROR {run.name}: {error}')
                    failures += 1
            log(f'CYCLE_DONE runs={len(runs)} errors={failures}; next check in {args.interval}s')
            if args.once:
                raise SystemExit(1 if failures else 0)
            time.sleep(args.interval)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('Backup watcher stopped; training is unaffected')
