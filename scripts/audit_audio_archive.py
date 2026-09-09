"""Read every in-scope PCM WAV and verify ZIP CRCs without PyTorch dependencies.

This checks the archive, not CUDA or the SoundFile/DataLoader runtime.
Run from repository root: python scripts/audit_audio_archive.py
"""

import argparse
import io
import json
import time
import wave
import zipfile
from pathlib import Path


def audit(path):
    started = time.perf_counter()
    report = {"archive": str(Path(path).resolve()), "partitions": {}}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("Duplicate ZIP member names are ambiguous.")
        members = set(names)
        for partition, expected in (("train-100", 13900), ("dev", 3000), ("test", 3000)):
            root = f"Libri2Mix/wav16k/min/{partition}"
            prefix = f"{root}/mix_clean/"
            mixtures = sorted(name for name in members if name.startswith(prefix) and name.endswith(".wav"))
            if len(mixtures) != expected:
                raise ValueError(f"{partition}: expected {expected}, got {len(mixtures)} mixtures")
            total_frames = 0
            widths = set()
            for index, mixture in enumerate(mixtures):
                key = mixture[len(prefix):]
                lengths = []
                for member in (mixture, f"{root}/s1/{key}", f"{root}/s2/{key}"):
                    raw = archive.read(member)  # Full read verifies ZIP CRC, not just WAV header.
                    with wave.open(io.BytesIO(raw), "rb") as audio:
                        if audio.getnchannels() != 1 or audio.getframerate() != 16000:
                            raise ValueError(f"Expected mono 16 kHz: {member}")
                        if audio.getcomptype() != "NONE" or audio.getnframes() <= 0:
                            raise ValueError(f"Expected nonempty PCM audio: {member}")
                        frames, width = audio.getnframes(), audio.getsampwidth()
                        if len(audio.readframes(frames)) != frames * width:
                            raise ValueError(f"Truncated PCM payload: {member}")
                        widths.add(width)
                        lengths.append(frames)
                if len(set(lengths)) != 1:
                    raise ValueError(f"Unequal raw lengths: {partition}/{key}: {lengths}")
                total_frames += lengths[0]
                if (index + 1) % 1000 == 0:
                    print(f"{partition}: {index + 1}/{expected} mixtures verified", flush=True)
            report["partitions"][partition] = {
                "mixtures": len(mixtures), "wav_streams_read": 3 * len(mixtures),
                "mixture_hours": total_frames / 16000 / 3600,
                "pcm_sample_width_bytes": sorted(widths),
                "zip_crc_and_raw_length_checks": "passed",
            }
    report["elapsed_seconds"] = time.perf_counter() - started
    report["status"] = "passed"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/Libri2Mix.zip")
    args = parser.parse_args()
    print(json.dumps(audit(args.archive), indent=2))
