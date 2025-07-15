from pathlib import Path
import yaml
import torch
from typing import Optional, Union, Literal
import os
import re
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


BASE_DIR = Path(__file__).resolve().parent


def load_yml(yml_file: Path):
    """Load a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    yml_file : Path
        Path to the YAML file to be loaded.

    Returns
    -------
    dict
        Dictionary containing the key-value pairs parsed from the YAML file.

    Examples
    --------
    >>> from pathlib import Path
    >>> data = load_yml(Path("config.yml"))
    >>> print(data["learning_rate"])
    """
    with open(yml_file, encoding='ascii', errors='ignore') as f:
        kwdict = yaml.safe_load(f)
    return kwdict


def select_device(device: Optional[Union[str, torch.device]]=""):
    """Select and configure the single device (CPU or CUDA GPU) for PyTorch training or inference.

    Parameters
    ----------
    device : Optional[Union[str, torch.device]]
        Device to use. Examples: 'cpu', 'cuda', '0', '0,1' -> convert 'cuda', torch.device('cuda:0').
        If empty, defaults to GPU if available, else CPU.

    Returns
    -------
    torch.device: 
        Selected PyTorch device object, e.g., torch.device('cuda:0') or torch.device('cpu').

    Examples
    --------
    >>> device = select_device('cpu')  # force CPU
    >>> device = select_device()  # auto-select best device (GPU preferred)
    """

    
    if isinstance(device, torch.device):
        return device
    
    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "") # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
    cpu = device == "cpu"
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # force torch.cuda.is_avaliable() = False
    elif device: # GPU device requested 
        if device == "cuda":
            device = "0"
        if "," in device:
            device = ",".join([x for x in device.split(",") if x]) # remove sequential commas, i.e. "0,,1" -> "0,1"
        # visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        # os.environ["CUDA_VISIBLE_DEVICES"] = device # set environment variable - must be before assert is_available()
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.split(","))):
            print(f"Select Device CUDA {device}")
            install = (
                "See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no "
                "CUDA devices are seen by torch.\n"
                if torch.cuda.device_count() == 0
                else ""
            )
            raise ValueError(
                f"Invalid CUDA 'device={device}' requested."
                f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                f" i.e. 'device=0' for single GPU\n"
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}"
                # f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                f"{install}"
            )
    
    if not cpu and torch.cuda.is_available(): # prefer GPU if available
        devices = device.split(",") if device else "0" # i.e. "0,1" -> ["0", "1"]
        # n = len(devices) # device count
        arg = f"cuda:{devices[0]}" if len(devices) == 1 else "cuda:0"
    else:
        arg = "cpu"
        torch.set_num_threads(min(8, max(1, os.cpu_count() - 1)))
    return torch.device(arg)


def increment_path(path: Path, exist_ok: bool=False, 
                   sep: str="", mkdir: bool=True):
    """Increment file or directory path if it exists to avoid overwriting.

    Appends a number (e.g., path2, path3, ...) until a unique path is found.

    Parameters
    ----------
    path : Path
        The initial path to check and increment if needed.

    exist_ok : bool
        If True, returns the original path even if it exists.
        If False, increments the path until it does not exist.

    sep : str, optional (default="")
        Separator string to insert between the base path and the increment number.
        For example, `sep='_'` results in paths like `path_2`, `path_3`, etc.

    mkdir : bool, optional (default=False)
        If True, creates the directory at the final path.
        Parent directories are created as needed.

    Returns
    -------
    Path
        A unique file or directory path, incremented if necessary.

    Examples
    --------
    >>> increment_path(Path("runs/exp"))
    PosixPath('runs/exp2')  # if 'runs/exp' exists
    """
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}" # increment path
            if not os.path.exists(p):
                break
        path = Path(p)
    
    if mkdir:
        path.mkdir(parents=True, exist_ok=True) # make directory
    
    return path


def check_amp(model: torch.nn.Module):
    """Check if Automatic Mixed Precision (AMP) training is safe to use on the given model's GPU.

    Some GPUs (e.g., GTX 16xx series and certain Quadros/Teslas) are known to cause instability 
    when using AMP (e.g., NaN losses or zero mAP). This function detects such GPUs and disables AMP accordingly.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model whose parameters will be checked to determine the device type (CPU or GPU).

    Returns
    -------
    bool
        True if AMP is safe to use on the detected device.
        False if AMP should be disabled (e.g., on problematic GPUs or CPU).

    Examples
    --------
    >>> model = MyModel().to('cuda:0')
    >>> amp_ok = check_amp(model)
    >>> print(f"AMP usable: {amp_ok}")
    """
    print("Check AMP")
    device = next(model.parameters()).device
    if device.type == "cpu": 
        return False
    else:
        # GPUs that have issues with AMP
        pattern = re.compile(
            r"(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)", re.IGNORECASE
        )

        gpu = torch.cuda.get_device_name(device)
        if bool(pattern.search(gpu)):
            print(
                f"checks failed ❌. AMP training on {gpu} GPU may cause "
                f"NaN losses or zero-mAP results, so AMP will be disabled during training."
            )
            return False
    return True


class EarlyStopping:
    """Implements early stopping mechanism to terminate training when validation fitness fails to improve.

    Early stopping monitors the "fitness" metric and stops training if no improvement is seen 
    over a specified number of consecutive epochs (`patience`). This helps avoid overfitting 
    and saves computation time.

    Parameters
    ----------
    patience : int
        Number of consecutive epochs to wait for improvement before stopping.
        Set `patience=0` to disable early stopping.

    Attributes
    ----------
    best_fitness : float
        The best fitness score observed during training.
    best_epoch : int
        The epoch at which the best fitness score was observed.
    possible_stop : bool
        Indicates if early stopping is likely to be triggered in the next epoch.
    
    Examples
    --------
    >>> stopper = EarlyStopping(patience=20)
    >>> for epoch in range(100):
    ...     fitness = validate_model()
    ...     if stopper(epoch, fitness):
    ...         break  # stop training early
    """
    def __init__(self, patience: int=50):
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.patience = patience or float("inf") # epochs to wait after fitness stops improving before stopping
        self.possible_stop = False # possible stop may occur next epoch

    def __call__(self, epoch: int, fitness: float):
        print("Early Stopping")
        if fitness is None:
            return False # check if fitness=None (happens when val=False)
        
        if fitness > self.best_fitness  or self.best_fitness == 0: # allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1) # possible stop may occur next epoch
        stop = delta >= self.patience # stop training if patience exceeded 
        if stop:
            print(
                f"Training stopped early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `patience=300` or use `patience=0` to disable EarlyStopping."
            )
        return stop