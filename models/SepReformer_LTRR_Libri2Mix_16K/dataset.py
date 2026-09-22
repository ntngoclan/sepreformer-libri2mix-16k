import io
import random
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from utils.decorators import logger_wraps


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(path):
    path = Path(path)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _seed_worker(_worker_id):
    """Give NumPy/Python crop RNG a deterministic DataLoader worker seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _collate(examples):
    if not isinstance(examples, list) or not examples:
        raise ValueError("Expected a non-empty list of examples")

    examples = sorted(examples, key=lambda item: item["num_sample"], reverse=True)
    num_spks = len(examples[0]["src"])
    mixture = torch.nn.utils.rnn.pad_sequence(
        [torch.as_tensor(item["mix"], dtype=torch.float32) for item in examples],
        batch_first=True,
    )
    sources = [
        torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(item["src"][spk], dtype=torch.float32) for item in examples],
            batch_first=True,
        )
        for spk in range(num_spks)
    ]
    input_sizes = torch.as_tensor(
        [item["num_sample"] for item in examples], dtype=torch.float32
    )
    keys = [item["key"] for item in examples]
    return input_sizes, mixture, sources, keys


@logger_wraps()
def get_dataloaders(args, dataset_config, loader_config):
    # Single-file inference does not need to scan the 14 GB dataset archive.
    if args.engine_mode == "infer_sample":
        return {}

    partitions = ["test"] if "test" in args.engine_mode else ["train", "valid", "test"]
    dataloaders = {}
    loader_seed = int(loader_config.get("seed", 0))
    for partition in partitions:
        dataset = Libri2MixDataset(dataset_config, partition)
        generator = torch.Generator()
        generator.manual_seed(loader_seed)
        dataloaders[partition] = DataLoader(
            dataset,
            batch_size=loader_config["batch_size"] if partition == "train" else 1,
            shuffle=partition == "train",
            pin_memory=loader_config["pin_memory"],
            num_workers=loader_config["num_workers"],
            drop_last=loader_config["drop_last"] if partition == "train" else False,
            collate_fn=_collate,
            worker_init_fn=_seed_worker,
            generator=generator,
        )
    return dataloaders


class Libri2MixDataset(Dataset):
    """Read Libri2Mix from an extracted directory or directly from its ZIP file."""

    def __init__(self, config, partition):
        self.partition = partition
        self.sampling_rate = int(config["sampling_rate"])
        self.max_len = int(config["max_len"])
        self.length_multiple = int(config.get("sample_length_multiple", 1))
        self.partition_dir = config["partitions"][partition]
        self.mixture_dir = config.get("mixture_dir", "mix_clean")
        self.source_dirs = tuple(config.get("source_dirs", ["s1", "s2"]))
        self.extracted_root = _project_path(config["extracted_dir"])
        self.archive_path = _project_path(config["archive"])
        self.archive_root = PurePosixPath(config.get("archive_root", "Libri2Mix/wav16k/min"))
        self._zip = None

        extracted_partition = self.extracted_root / self.partition_dir
        required_dirs = (self.mixture_dir,) + self.source_dirs
        if all((extracted_partition / name).is_dir() for name in required_dirs):
            self.storage = "directory"
            self.examples = self._index_directory(extracted_partition)
        elif self.archive_path.is_file():
            self.storage = "zip"
            self.examples = self._index_archive()
        else:
            raise FileNotFoundError(
                "Libri2Mix 16 kHz was not found. Expected either "
                f"'{self.extracted_root}' or '{self.archive_path}'."
            )

        if not self.examples:
            raise RuntimeError(f"No WAV examples found for Libri2Mix partition '{partition}'")
        logger.info(
            f"Libri2Mix {partition}: {len(self.examples)} mixtures from {self.storage} storage"
        )

    def _index_directory(self, partition_root):
        mixture_root = partition_root / self.mixture_dir
        examples = []
        for mixture in sorted(mixture_root.glob("*.wav")):
            sources = tuple(partition_root / name / mixture.name for name in self.source_dirs)
            missing = [str(path) for path in sources if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Missing source WAV(s) for '{mixture.name}': {', '.join(missing)}"
                )
            examples.append((mixture.name, str(mixture), tuple(map(str, sources))))
        return examples

    def _index_archive(self):
        partition_root = self.archive_root / self.partition_dir
        mixture_prefix = f"{partition_root / self.mixture_dir}/"
        source_prefixes = tuple(f"{partition_root / name}/" for name in self.source_dirs)

        with zipfile.ZipFile(self.archive_path) as archive:
            members = {
                info.filename
                for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".wav")
            }

        mixtures = sorted(name for name in members if name.startswith(mixture_prefix))
        examples = []
        for mixture in mixtures:
            key = PurePosixPath(mixture).name
            sources = tuple(prefix + key for prefix in source_prefixes)
            missing = [source for source in sources if source not in members]
            if missing:
                raise FileNotFoundError(
                    f"Archive is missing source WAV(s) for '{key}': {', '.join(missing)}"
                )
            examples.append((key, mixture, sources))
        return examples

    def __len__(self):
        return len(self.examples)

    def __getstate__(self):
        # ZipFile handles cannot be sent to DataLoader worker processes.
        state = self.__dict__.copy()
        state["_zip"] = None
        return state

    def _get_zip(self):
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.archive_path)
        return self._zip

    def _load_audio(self, reference):
        if self.storage == "zip":
            with self._get_zip().open(reference) as stream:
                audio, sampling_rate = sf.read(
                    io.BytesIO(stream.read()), dtype="float32", always_2d=False
                )
        else:
            audio, sampling_rate = sf.read(reference, dtype="float32", always_2d=False)

        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"'{reference}' has sampling rate {sampling_rate}, expected {self.sampling_rate} Hz"
            )
        if audio.ndim != 1:
            raise ValueError(f"'{reference}' is not mono (shape={audio.shape})")
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"'{reference}' contains empty/non-finite audio")
        return np.asarray(audio, dtype=np.float32)

    def __getitem__(self, index):
        key, mixture_ref, source_refs = self.examples[index]
        mixture = self._load_audio(mixture_ref)
        sources = [self._load_audio(reference) for reference in source_refs]

        lengths = [len(mixture)] + [len(source) for source in sources]
        if len(set(lengths)) != 1:
            raise ValueError(f"'{key}' mixture/source lengths differ: {lengths}")
        aligned_len = lengths[0]
        mixture = mixture[:aligned_len]
        sources = [source[:aligned_len] for source in sources]

        if self.partition == "train" and aligned_len > self.max_len:
            start = random.randint(0, aligned_len - self.max_len)
            stop = start + self.max_len
            mixture = mixture[start:stop]
            sources = [source[start:stop] for source in sources]
        elif self.partition == "valid" and aligned_len > self.max_len:
            # A deterministic validation crop makes epoch-to-epoch metrics comparable.
            start = (aligned_len - self.max_len) // 2
            stop = start + self.max_len
            mixture = mixture[start:stop]
            sources = [source[start:stop] for source in sources]

        remainder = len(mixture) % self.length_multiple
        if remainder:
            mixture = mixture[:-remainder]
            sources = [source[:-remainder] for source in sources]
        if not len(mixture):
            raise ValueError(f"'{key}' is shorter than one model stride")

        return {
            "num_sample": len(mixture),
            "mix": mixture,
            "src": sources,
            "key": key,
        }

    def __del__(self):
        archive = getattr(self, "_zip", None)
        if archive is not None:
            archive.close()
