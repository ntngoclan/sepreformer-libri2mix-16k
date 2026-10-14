"""Failure and consistency tests; no cloud credentials or GPU required."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from watch_checkpoint_backup import Publisher, make_archive, verify


class LocalPublisher(Publisher):
    fail_check = False
    calls = 0

    def invoke(self, *args, capture=False):
        self.calls += 1
        operation, source = args[:2]
        if operation == 'copy':
            shutil.copytree(source, args[2], dirs_exist_ok=True)
        elif operation == 'check':
            if self.fail_check:
                raise subprocess.CalledProcessError(1, ['rclone', 'check'])
            for path in Path(source).iterdir():
                assert path.read_bytes() == (Path(args[2]) / path.name).read_bytes()
        elif operation == 'lsjson':
            return json.dumps([{'Name': p.name, 'Path': p.name, 'IsDir': p.is_dir(),
                                'Size': p.stat().st_size} for p in Path(source).iterdir()])
        elif operation in ('deletefile', 'rmdir'):
            # Test mutations are confined to this fixture's remote directory.
            path = Path(source).resolve()
            path.relative_to(Path(self.remote).resolve())
            assert '--drive-use-trash=false' in args
            if operation == 'deletefile':
                path.unlink()
            else:
                path.rmdir()
        else:
            raise AssertionError(args)


class BackupTests(unittest.TestCase):
    key = 'runs/SepReformer_Base_VnSpeechMix_16K/seed_0000/vn_base_s0_resume1'

    def setUp(self):
        logger = patch('watch_checkpoint_backup.log')
        self.mock_log = logger.start()
        self.addCleanup(logger.stop)

    def seed_versions(self, root, count=5):
        checkpoint = root / self.key / 'checkpoints/latest.pth'
        checkpoint.parent.mkdir(parents=True)
        p = LocalPublisher(root, root / 'work', str(root / 'remote'), [])
        for epoch in range(count):
            checkpoint.write_bytes(b'epoch ' + b'x' * (epoch + 1))
            p.publish(self.key, [checkpoint])
        folders = sorted((root / 'remote' / self.key).iterdir())
        return p, checkpoint, folders

    def test_retention_cleans_legacy_without_new_checkpoint_and_preserves_other_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p, checkpoint, versions = self.seed_versions(root)
            for relative in ('support/old/keep.txt', 'runs/parent/keep.txt',
                             self.key + '/manual-copy/keep.txt'):
                path = root / 'remote' / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('untouched')
            p.keep = 3
            p.publish(self.key, [checkpoint])
            self.assertTrue(all(not path.exists() for path in versions[:2]))
            self.assertTrue(all(path.exists() for path in versions[2:]))
            self.assertEqual(len(list((root / 'remote').rglob('keep.txt'))), 3)
            checkpoint.write_bytes(b'next epoch')
            p.publish(self.key, [checkpoint])
            snapshots = list((root / 'remote' / self.key).glob('*/snapshot.zip'))
            self.assertEqual(len(snapshots), 3)

    def test_corrupt_retained_archive_prevents_all_deletion(self):
        with tempfile.TemporaryDirectory() as temp:
            p, checkpoint, versions = self.seed_versions(Path(temp))
            (versions[-2] / 'snapshot.zip').write_bytes(b'broken')
            p.keep = 3
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                p.publish(self.key, [checkpoint])
            self.assertTrue(all(path.exists() for path in versions))

    def test_dry_run_and_upload_failure_do_not_delete(self):
        with tempfile.TemporaryDirectory() as temp:
            p, checkpoint, versions = self.seed_versions(Path(temp))
            p.keep, p.prune_dry_run = 3, True
            p.publish(self.key, [checkpoint])
            self.assertTrue(all(path.exists() for path in versions))
            p.prune_dry_run, p.fail_check = False, True
            checkpoint.write_bytes(b'new failed epoch')
            with self.assertRaises(subprocess.CalledProcessError):
                p.publish(self.key, [checkpoint])
            self.assertTrue(all(path.exists() for path in versions))

    def test_missing_anchor_and_duplicate_listing_block_deletion(self):
        with tempfile.TemporaryDirectory() as temp:
            p, checkpoint, versions = self.seed_versions(Path(temp))
            p.keep = 3
            anchor = p.state[p.remote + '/' + self.key]['remote']
            p.state[p.remote + '/' + self.key]['remote'] = anchor + '-missing'
            with self.assertRaisesRegex(ValueError, 'No verified snapshot'):
                p.prune(self.key)
            p.state[p.remote + '/' + self.key]['remote'] = anchor
            entry = {'Name': versions[-1].name, 'Path': versions[-1].name, 'IsDir': True}
            with patch.object(p, 'invoke', return_value=json.dumps([entry, entry])):
                with self.assertRaisesRegex(ValueError, 'Duplicate names'):
                    p.prune(self.key)
            self.assertTrue(all(path.exists() for path in versions))

    def test_unexpected_files_are_untouched_and_partial_deletion_is_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            p, checkpoint, versions = self.seed_versions(Path(temp), count=6)
            (versions[0] / 'keep.txt').write_text('manual addition')
            # Simulate interruption between deletefile calls on another old snapshot.
            (versions[1] / 'snapshot.zip').unlink()
            p.keep = 3
            p.publish(self.key, [checkpoint])
            self.assertTrue((versions[0] / 'keep.txt').exists())
            self.assertFalse(versions[1].exists())
            self.assertFalse(versions[2].exists())
            self.assertTrue(all(path.exists() for path in versions[-3:]))

    def test_foreign_archive_blocks_cleanup_and_unsafe_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p, checkpoint, versions = self.seed_versions(root)
            p.keep = 3
            foreign = root / 'foreign.pth'
            foreign.write_bytes(b'wrong run')
            (versions[-1] / 'snapshot.zip').unlink()
            make_archive(root, [foreign], versions[-1] / 'snapshot.zip')
            with self.assertRaisesRegex(ValueError, 'different run'):
                p.prune(self.key)
            for key in ('runs/../../support', self.key + '/../parent', 'runs'):
                with self.assertRaisesRegex(ValueError, 'Unsafe retention key'):
                    p.prune(key)
            self.assertTrue(all(path.exists() for path in versions))

    def test_newer_unverified_snapshot_does_not_displace_retained_copies(self):
        with tempfile.TemporaryDirectory() as temp:
            p, checkpoint, versions = self.seed_versions(Path(temp))
            future = versions[0].parent / '20991010T120000_000000Z_abcdef01'
            future.mkdir()
            (future / 'snapshot.zip').write_bytes(b'failed upload')
            p.keep = 3
            p.prune(self.key)
            self.assertTrue(future.exists())
            self.assertTrue(all(path.exists() for path in versions[-3:]))
            self.assertFalse(versions[0].exists())

    def test_delete_failure_is_retried_with_retained_snapshots_intact(self):
        with tempfile.TemporaryDirectory() as temp:
            p, checkpoint, versions = self.seed_versions(Path(temp))
            p.keep = 3
            original = p.invoke
            def fail_on_sidecar(*args, **kwargs):
                if args[0] == 'deletefile' and args[1].endswith('.sha256'):
                    raise subprocess.CalledProcessError(1, ['rclone', 'deletefile'])
                return original(*args, **kwargs)
            with patch.object(p, 'invoke', side_effect=fail_on_sidecar):
                with self.assertRaises(subprocess.CalledProcessError):
                    p.prune(self.key)
            self.assertTrue(all(path.exists() for path in versions[-3:]))
            p.publish(self.key, [checkpoint])
            self.assertFalse(versions[0].exists())
            self.assertFalse(versions[1].exists())

    def test_disabled_retention_and_small_count_do_not_delete(self):
        with tempfile.TemporaryDirectory() as temp:
            p, checkpoint, versions = self.seed_versions(Path(temp), count=2)
            p.prune(self.key)
            p.keep = 3
            p.prune(self.key)
            self.assertTrue(all(path.exists() for path in versions))

    def test_failed_verification_retries_without_advancing_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / 'latest.pth'
            checkpoint.write_bytes(b'epoch 2')
            p = LocalPublisher(root, root / 'work', str(root / 'remote'), [])
            p.fail_check = True
            with self.assertRaises(subprocess.CalledProcessError):
                p.publish('run', [checkpoint])
            self.assertFalse(p.state)
            self.assertFalse(p.state_path.exists())
            p.fail_check = False
            self.assertTrue(p.publish('run', [checkpoint]))
            calls = p.calls
            self.assertFalse(p.publish('run', [checkpoint]))
            self.assertEqual(p.calls, calls)
            checkpoint.write_bytes(b'epoch 3 newer')
            self.assertTrue(p.publish('run', [checkpoint]))
            self.assertEqual(len(list((root / 'remote/run').iterdir())), 3)
            restarted = LocalPublisher(root, root / 'work', str(root / 'remote'), [])
            self.assertFalse(restarted.publish('run', [checkpoint]))

    @unittest.skipUnless(os.name == 'posix', 'Atomic replacement of open files needs POSIX')
    def test_replacement_during_read_keeps_complete_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint, replacement = root / 'latest.pth', root / 'latest.pth.tmp'
            old = b'old checkpoint' * 10000
            checkpoint.write_bytes(old)
            replacement.write_bytes(b'new checkpoint' * 20000)
            original_open = Path.open
            def replace_after_open(path, *args, **kwargs):
                handle = original_open(path, *args, **kwargs)
                if path == checkpoint and replacement.exists():
                    os.replace(replacement, checkpoint)
                return handle
            archive = root / 'snapshot.zip'
            with patch.object(Path, 'open', replace_after_open):
                make_archive(root, [checkpoint], archive)
            self.assertEqual(verify(archive), 1)
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.read('latest.pth'), old)

    def test_corrupt_download_fails_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / 'latest.pth'
            checkpoint.write_bytes(b'checkpoint bytes')
            archive = root / 'snapshot.zip'
            make_archive(root, [checkpoint], archive)
            self.assertEqual(verify(archive), 1)
            with archive.open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'Archive SHA-256 mismatch'):
                verify(archive)


if __name__ == '__main__':
    unittest.main()
