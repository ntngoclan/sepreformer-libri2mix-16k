#!/usr/bin/env python3
"""Prepare Libri2Mix .scp files and patch configs.yaml paths for packaged models."""
from __future__ import annotations

import argparse
from pathlib import Path

MODELS = [
    "SepReformer_Base_Libri2Mix_16K",
    "SepReformer_LTRR_Libri2Mix_16K",
]


def expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def write_scp(data_root: Path, scp_dir: Path) -> None:
    split_map = {"tr": "train-100", "cv": "dev", "tt": "test"}
    sub_map = [("mix", "mix_clean"), ("s1", "s1"), ("s2", "s2")]
    scp_dir.mkdir(parents=True, exist_ok=True)
    for prefix, split_name in split_map.items():
        split_dir = data_root / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        for suffix, subdir in sub_map:
            wav_dir = split_dir / subdir
            if not wav_dir.is_dir():
                raise FileNotFoundError(f"Missing wav directory: {wav_dir}")
            utt_ids = sorted(p.stem for p in wav_dir.glob("*.wav"))
            out = scp_dir / f"{prefix}_{suffix}.scp"
            with out.open("w", encoding="utf-8") as f:
                for uid in utt_ids:
                    f.write(f"{uid} {wav_dir / (uid + '.wav')}\n")
            print(f"Wrote {out}: {len(utt_ids)} utterances")


def patch_config(config_path: Path, scp_dir: Path, max_epoch: int | None, num_workers: int | None) -> None:
    text = config_path.read_text(encoding="utf-8", errors="ignore")
    # Replace old absolute paths that appeared on the training server.
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("scp_dir:"):
            indent = line[: len(line) - len(line.lstrip())]
            line = f'{indent}scp_dir: "{scp_dir}"'
        elif max_epoch is not None and stripped.startswith("max_epoch:"):
            indent = line[: len(line) - len(line.lstrip())]
            line = f"{indent}max_epoch: {max_epoch}"
        elif num_workers is not None and stripped.startswith("num_workers:"):
            indent = line[: len(line) - len(line.lstrip())]
            line = f"{indent}num_workers: {num_workers}"
        lines.append(line)
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Patched config: {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="~/SepReformer/data/Libri2Mix/wav16k/min")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--max-epoch", type=int, default=None, help="Optionally override engine.max_epoch")
    parser.add_argument("--num-workers", type=int, default=None, help="Optionally override dataloader.num_workers")
    args = parser.parse_args()

    data_root = expand(args.data_root)
    project_root = expand(args.project_root)
    models_root = project_root / "models"
    if not models_root.is_dir():
        raise FileNotFoundError(f"Missing models root: {models_root}. Run 01_prepare_data.sh first.")

    for model in MODELS:
        model_dir = models_root / model
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Missing model dir: {model_dir}")
        scp_dir = model_dir / "data" / "scp_libri2mix_16k"
        write_scp(data_root, scp_dir)
        patch_config(model_dir / "configs.yaml", scp_dir, args.max_epoch, args.num_workers)

    print("Done. Configs and SCP files are ready.")


if __name__ == "__main__":
    main()
