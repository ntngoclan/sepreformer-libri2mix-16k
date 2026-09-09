import os
import random
import yaml
import torch
import inspect
import numpy as np

from loguru import logger
from utils.decorators import *


def set_random_seed(seed, deterministic=True):
    """Seed Python, NumPy and PyTorch consistently for an experiment run."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    logger.info(f"Experiment seed={seed}, deterministic={bool(deterministic)}")


@logger_wraps()
def parse_yaml(path):
    """
    Parse and return the contents of a YAML file.

    Args:
        path (str): Path to the YAML file to be parsed.

    Returns:
        dict: A dictionary containing the parsed contents of the YAML file.

    Raises:
        FileNotFoundError: If the provided path does not point to an existing file.
    """
    try:
        with open(path, 'r') as yaml_file:
            config_dict = yaml.full_load(yaml_file)
        return config_dict
    except FileNotFoundError:
        raise
