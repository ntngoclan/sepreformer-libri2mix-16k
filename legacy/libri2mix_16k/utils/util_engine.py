import os
import torch
from loguru import logger
from torchinfo import summary as summary_
from ptflops import get_model_complexity_info
from thop import profile
import numpy as np
import torch



def load_checkpoint_file(checkpoint_file, model, optimizer=None, location="cpu"):
    """
    Load a specific checkpoint file.

    Args:
        checkpoint_file (str): Path to checkpoint file.
        model (torch.nn.Module): Model to load state_dict into.
        optimizer (torch.optim.Optimizer or None): Optimizer to load state_dict into.
        location: Device for torch.load.

    Returns:
        int: checkpoint epoch + 1
    """
    checkpoint_file = os.path.expanduser(checkpoint_file)

    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    logger.info(f"Loaded checkpoint from explicit path: {checkpoint_file} .....")

    checkpoint_dict = torch.load(checkpoint_file, map_location=location)

    model.load_state_dict(checkpoint_dict["model_state_dict"], strict=False)

    if optimizer is not None and "optimizer_state_dict" in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])

    return checkpoint_dict.get("epoch", -1) + 1


def load_last_checkpoint_n_get_epoch(checkpoint_dir, model, optimizer, location):
    """
    Load latest checkpoint by epoch number from checkpoint_dir.
    Expected filename format: epoch.0059.pth
    """
    checkpoint_files = []

    for f in os.listdir(checkpoint_dir):
        if not f.endswith((".pth", ".pt", ".pkl")):
            continue

        try:
            epoch_num = int(f.split(".")[1])
        except Exception:
            continue

        checkpoint_files.append((epoch_num, f))

    if not checkpoint_files:
        return 1

    latest_epoch, latest_file = max(checkpoint_files, key=lambda x: x[0])
    latest_checkpoint_file = os.path.join(checkpoint_dir, latest_file)

    logger.info(f"Loaded latest checkpoint from {latest_checkpoint_file} .....")

    checkpoint_dict = torch.load(latest_checkpoint_file, map_location=location)

    model.load_state_dict(checkpoint_dict["model_state_dict"], strict=False)

    if optimizer is not None and "optimizer_state_dict" in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])

    return checkpoint_dict.get("epoch", latest_epoch) + 1
    
def save_checkpoint_per_nth(nth, epoch, model, optimizer, train_loss, valid_loss, checkpoint_path, wandb_run):
    """
    Save the state of the model and optimizer every nth epoch to a checkpoint file.
    Additionally, log and save the checkpoint file using wandb.

    Args:
        nth (int): Interval for which checkpoints should be saved.
        epoch (int): The current training epoch.
        model (nn.Module): The model whose state needs to be saved.
        optimizer (Optimizer): The optimizer whose state needs to be saved.
        checkpoint_path (str): Directory path where the checkpoint will be saved.
        wandb_run (wandb.wandb_run.Run): The current wandb run to log and save the checkpoint.

    Returns:
        None
    """
    if epoch % nth == 0:
        # Save the state of the model and optimizer to a checkpoint file
        torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss
                    },
                    os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))
        
        # Log and save the checkpoint file using wandb
        wandb_run.save(os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))

def save_checkpoint_per_best(best, valid_loss, train_loss, epoch, model, optimizer, checkpoint_path):
    """
    Save the state of the model and optimizer every nth epoch to a checkpoint file.
    Additionally, log and save the checkpoint file using wandb.

    Args:
        nth (int): Interval for which checkpoints should be saved.
        epoch (int): The current training epoch.
        model (nn.Module): The model whose state needs to be saved.
        optimizer (Optimizer): The optimizer whose state needs to be saved.
        checkpoint_path (str): Directory path where the checkpoint will be saved.
        wandb_run (wandb.wandb_run.Run): The current wandb run to log and save the checkpoint.

    Returns:
        None
    """
    if valid_loss < best:
        # Save the state of the model and optimizer to a checkpoint file
        torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss
                    },
                    os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))
        
        # # Log and save the checkpoint file using wandb
        # wandb_run.save(os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))
        best = valid_loss
    return best

def step_scheduler(scheduler, **kwargs):
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(kwargs.get('val_loss'))
    elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
        scheduler.step()
    elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        scheduler.step()
    # Add another schedulers
    else:
        raise ValueError(f"Unknown scheduler type: {type(scheduler)}")

def print_parameters_count(model):
    total_parameters = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_parameters += param_count
        logger.info(f"{name}: {param_count}")
    logger.info(f"Total parameters: {(total_parameters / 1e6):.2f}M")

def model_params_mac_summary(model, input, dummy_input, metrics):
    
    # ptflpos
    if 'ptflops' in metrics:
        MACs_ptflops, params_ptflops = get_model_complexity_info(model, (input.shape[1],), print_per_layer_stat=False, verbose=False) # (num_samples,)
        MACs_ptflops, params_ptflops = MACs_ptflops.replace(" MMac", ""), params_ptflops.replace(" M", "")
        logger.info(f"ptflops: MACs: {MACs_ptflops}, Params: {params_ptflops}")

    # thop
    if 'thop' in metrics:
        MACs_thop, params_thop = profile(model, inputs=(input, ), verbose=False)
        MACs_thop, params_thop = MACs_thop/1e9, params_thop/1e6
        logger.info(f"thop: MACs: {MACs_thop} GMac, Params: {params_thop}")
    
    # torchinfo
    if 'torchinfo' in metrics:
        model_profile = summary_(model, input_size=input.size(), verbose=0)
        MACs_torchinfo, params_torchinfo = model_profile.total_mult_adds/1e6, model_profile.total_params/1e6
        logger.info(f"torchinfo: MACs: {MACs_torchinfo} GMac, Params: {params_torchinfo}")
