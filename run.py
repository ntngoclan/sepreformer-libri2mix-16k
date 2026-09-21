import argparse
import importlib
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Parse args
parser = argparse.ArgumentParser(
    description="Command to start PIT training, configured by .yaml files")
parser.add_argument(
    "--model",
    type=str,
    default="IQformer_v6_0_0",
    dest="model",
    help="Insert model name")
parser.add_argument(
    "--engine-mode",
    choices=["train", "test", "test_save", "infer_sample"],
    default="train",
    help="This option is used to chooose the mode")
parser.add_argument(
    "--sample-file",
    type=str,
    default=None,
    help="directoy for sample audio")
parser.add_argument(
    "--out-wav-dir",
    type=str,
    default=None,
    help="Optional directory for separated WAV files in test_save mode")
parser.add_argument('--run-id', help='New run name; existing training directories are rejected')
parser.add_argument('--seed', type=int, help='Override experiment and DataLoader seed together')
parser.add_argument('--output-dir', default='runs')
parser.add_argument('--config', help='Optional YAML config override')
parser.add_argument('--resume', help='Explicit full checkpoint; continue in a new run directory')
parser.add_argument('--checkpoint', help='Explicit checkpoint for evaluation/inference')

# Call target model
if __name__ == '__main__':
    args = parser.parse_args()
    main_module = importlib.import_module(f"models.{args.model}.main")
    main_module.main(args)
