import os

import torch
from loguru import logger

from utils import util_implement, util_system
from utils.decorators import logger_wraps
from utils.paired_initialization import apply_paired_initialization

from .dataset import get_dataloaders
from .engine import Engine
from .model import Model


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(MODEL_DIR, "log"), exist_ok=True)
logger.add(os.path.join(MODEL_DIR, "log/system_log.log"), level="DEBUG", mode="a")


@logger_wraps()
def main(args):
    config = util_system.parse_yaml(os.path.join(MODEL_DIR, "configs.yaml"))["config"]
    util_system.set_random_seed(**config.get("experiment", {"seed": 0, "deterministic": True}))
    dataloaders = get_dataloaders(args, config["dataset"], config["dataloader"])
    model = Model(**config["model"])
    if args.engine_mode == "train":
        apply_paired_initialization(
            model,
            model_name=os.path.basename(MODEL_DIR),
            config=config,
            workspace_root=os.path.dirname(os.path.dirname(MODEL_DIR)),
        )
        # The helper baseline may need to be constructed when PARR runs first.
        # Reset all training RNGs so file creation/existence cannot affect training.
        util_system.set_random_seed(**config["experiment"])

    gpuid = tuple(map(int, config["engine"]["gpuid"].split(",")))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SepReformer training and evaluation.")
    invalid_gpu_ids = [gpu for gpu in gpuid if gpu < 0 or gpu >= torch.cuda.device_count()]
    if invalid_gpu_ids:
        raise RuntimeError(
            f"Configured GPU IDs {invalid_gpu_ids} are unavailable; "
            f"visible CUDA device count is {torch.cuda.device_count()}."
        )
    device = torch.device(f"cuda:{gpuid[0]}")
    torch.cuda.set_device(device)
    # Build optimizers only after parameters reside on their training device.
    model = model.to(device)
    criterions, optimizers, schedulers = [], [], []
    if args.engine_mode == "train":
        criterions = util_implement.CriterionFactory(config["criterion"], device).get_criterions()
        optimizers = util_implement.OptimizerFactory(
            config["optimizer"], model.parameters()
        ).get_optimizers()
        schedulers = util_implement.SchedulerFactory(
            config["scheduler"], optimizers
        ).get_schedulers()

    engine = Engine(
        args,
        config,
        model,
        dataloaders,
        criterions,
        optimizers,
        schedulers,
        gpuid,
        device,
        work_dir=MODEL_DIR,
    )
    if args.engine_mode == "infer_sample":
        engine._inference_sample(args.sample_file)
    else:
        engine.run()
