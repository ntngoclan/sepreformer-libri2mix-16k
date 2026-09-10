import os
import copy
import torch
from utils.runtime_state import capture_runtime_state
from loguru import logger
from torchinfo import summary as summary_
from ptflops import get_model_complexity_info
from thop import profile


class InferenceView(torch.nn.Module):
    """Profiler view of the final waveform path, with auxiliary heads disabled."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform):
        return self.model(waveform, return_aux=False)[0]


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    auxiliary = sum(p.numel() for name, p in model.named_parameters()
                    if name.startswith(("out_layer_bn.", "decoder_bn.")))
    return {"training_total": total, "auxiliary_heads": auxiliary,
            "inference_active": total - auxiliary}



def load_last_checkpoint_n_get_epoch(checkpoint_dir, model, optimizer, location):
    """
    Load the latest checkpoint (model state and optimizer state) from a given directory.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files.
        model (torch.nn.Module): The model into which the checkpoint's model state should be loaded.
        optimizer (torch.optim.Optimizer): The optimizer into which the checkpoint's optimizer state should be loaded.
        location (str, optional): Device location for loading the checkpoint. Defaults to 'cpu'.

    Returns:
        int: The epoch number associated with the loaded checkpoint. 
             If no checkpoint is found, returns 0 as the starting epoch.

    Notes:
        - The checkpoint file is expected to have keys: 'model_state_dict', 'optimizer_state_dict', and 'epoch'.
        - If there are multiple checkpoint files in the directory, the one with the highest epoch number is loaded.
    """
    # List all .pkl files in the directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir)]

    # If there are no checkpoint files, return 0 as the starting epoch
    if not checkpoint_files: return 1
    else:
        # Extract the epoch numbers from the file names and find the latest (max)
        epochs = [int(f.split('.')[1]) for f in checkpoint_files]
        latest_checkpoint_file = os.path.join(checkpoint_dir, checkpoint_files[epochs.index(max(epochs))])

        # Load the checkpoint into the model & optimizer
        logger.info(f"Loaded Pretrained model from {latest_checkpoint_file} .....")
        checkpoint_dict = torch.load(latest_checkpoint_file, map_location=location)
        model.load_state_dict(checkpoint_dict['model_state_dict'], strict=False) # Depend on weight file's key!!
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        
        # Retrun latent epoch
        return checkpoint_dict['epoch'] + 1
    
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

def _checkpoint_payload(
    valid_loss,
    train_loss,
    epoch,
    model,
    optimizer,
    schedulers=None,
    best_valid_loss=None,
    dataloaders=None,
):
    checkpoint = {
        'epoch': int(epoch),
        'runtime_state': capture_runtime_state(dataloaders),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': float(train_loss),
        'valid_loss': float(valid_loss),
        'best_valid_loss': float(
            valid_loss if best_valid_loss is None else best_valid_loss
        ),
    }
    if schedulers is not None:
        checkpoint['scheduler_state_dicts'] = [
            scheduler.state_dict() for scheduler in schedulers
        ]
    paired_metadata = getattr(model, 'paired_initialization_metadata', None)
    if paired_metadata is not None:
        checkpoint['paired_initialization_metadata'] = copy.deepcopy(paired_metadata)
    if hasattr(model, 'run_contract'):
        checkpoint['run_contract'] = copy.deepcopy(model.run_contract)
    return checkpoint


def _atomic_torch_save(checkpoint, destination):
    """Write a checkpoint atomically so interruption cannot corrupt the target."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = destination + '.tmp'
    torch.save(checkpoint, temporary)
    os.replace(temporary, destination)


def save_latest_checkpoint(
    valid_loss,
    train_loss,
    epoch,
    model,
    optimizer,
    checkpoint_path,
    schedulers=None,
    best_valid_loss=None,
    dataloaders=None,
):
    checkpoint = _checkpoint_payload(
        valid_loss,
        train_loss,
        epoch,
        model,
        optimizer,
        schedulers,
        best_valid_loss,
        dataloaders,
    )
    _atomic_torch_save(checkpoint, os.path.join(checkpoint_path, 'latest.pth'))


def save_checkpoint_per_best(
    best,
    valid_loss,
    train_loss,
    epoch,
    model,
    optimizer,
    checkpoint_path,
    schedulers=None,
    dataloaders=None,
):
    """
    Save the best model, optimizer, and optional scheduler states.

    Args:
        epoch (int): The current training epoch.
        model (nn.Module): The model whose state needs to be saved.
        optimizer (Optimizer): The optimizer whose state needs to be saved.
        checkpoint_path (str): Directory path where the checkpoint will be saved.
        schedulers (optional): Schedulers whose state should be restored on resume.

    Returns:
        None
    """
    if valid_loss < best:
        checkpoint = _checkpoint_payload(
            valid_loss,
            train_loss,
            epoch,
            model,
            optimizer,
            schedulers,
            valid_loss,
            dataloaders,
        )
        _atomic_torch_save(checkpoint, os.path.join(checkpoint_path, 'best.pth'))
        
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
    # Profilers install hooks/buffers; never let a failing profiler contaminate
    # the live model or its checkpoint. Each profiler receives a disposable copy.
    results = {
        'input_num_samples': int(input.shape[-1]),
        'total_parameters': int(sum(parameter.numel() for parameter in model.parameters())),
    }
    # ptflpos
    if 'ptflops' in metrics:
        try:
            MACs_ptflops, params_ptflops = get_model_complexity_info(
                copy.deepcopy(model),
                (input.shape[1],),
                print_per_layer_stat=False,
                verbose=False,
                as_strings=False,
            )
            MACs_ptflops, params_ptflops = MACs_ptflops / 1e9, params_ptflops / 1e6
            logger.info(f"ptflops: MACs: {MACs_ptflops} GMac, Params: {params_ptflops} M")
            results['ptflops'] = {
                'macs_giga': float(MACs_ptflops),
                'parameters_million': float(params_ptflops),
            }
        except Exception as error:
            logger.warning(f"ptflops profiling skipped after error: {error}")
            results['ptflops'] = {'error': str(error)}

    # thop
    if 'thop' in metrics:
        try:
            MACs_thop, params_thop = profile(copy.deepcopy(model), inputs=(input, ), verbose=False)
            MACs_thop, params_thop = MACs_thop/1e9, params_thop/1e6
            logger.info(f"thop: MACs: {MACs_thop} GMac, Params: {params_thop}")
            results['thop'] = {
                'macs_giga': float(MACs_thop),
                'parameters_million': float(params_thop),
            }
        except Exception as error:
            logger.warning(f"THOP profiling skipped after error: {error}")
            results['thop'] = {'error': str(error)}
    
    # torchinfo
    if 'torchinfo' in metrics:
        try:
            model_profile = summary_(copy.deepcopy(model), input_size=input.size(), verbose=0)
            MACs_torchinfo = model_profile.total_mult_adds / 1e9
            params_torchinfo = model_profile.total_params / 1e6
            logger.info(f"torchinfo: MACs: {MACs_torchinfo} GMac, Params: {params_torchinfo}")
            results['torchinfo'] = {
                'mult_adds_giga': float(MACs_torchinfo),
                'parameters_million': float(params_torchinfo),
            }
        except Exception as error:
            logger.warning(f"torchinfo profiling skipped after error: {error}")
            results['torchinfo'] = {'error': str(error)}
    return results
