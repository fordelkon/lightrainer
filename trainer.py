"""Basic trainer class"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import subprocess
import math
import gc
from typing import Optional, Literal

try:
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import calib
    from pytorch_quantization.tensor_quant import QuantDescriptor
    quant_enable = True
    quant_desc_weight=QuantDescriptor(num_bits=8, axis=(0))
    quant_desc_input=QuantDescriptor(num_bits=8, calib_method="histogram")
except ImportError:
    quant_enable = False

from .utils import *
from .modules import *


class BaseTrainer:
    """
    Base trainer class for deep learning model training, validation, and export.

    This class provides a complete training pipeline with support for:
    - Model training and evaluation infrastructure
    - Quantization-aware training (QAT) and post-training quantization (PTQ)
    - Multiple optimizers and learning rate schedulers
    - Automatic mixed precision (AMP) training
    - Exporting models to ONNX and TensorRT formats
    - Early stopping and checkpointing

    Args:
        model (nn.Module): The PyTorch model to be trained. It must return a total loss value during the forward pass.
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        cfg (Path): Path to the configuration file containing all training hyperparameters.

    Attributes:
        model (nn.Module): The model being trained.
        train_dataset (Dataset): The dataset used for training.
        val_dataset (Dataset): The dataset used for validation.
        device (torch.device): Training device (CPU or GPU).
        batch_size (int): Batch size for training and validation.
        epochs (int): Number of training epochs.
        lr0 (float): Initial learning rate.
        amp (bool): Whether to use automatic mixed precision training.
        quant_enable (bool): Whether pytorch quantization is installed.
        best_fitness (float): Best validation metric observed so far.
    """
    def __init__(self, model: nn.Module, 
                 train_dataset: Dataset, val_dataset: Dataset, 
                 cfg: Path):
        # The model should output total loss as we expected
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.kwdict = load_yml(cfg)

        self.device = select_device(self.kwdict["device"]) # select device, e.g. 0 or 1 or 2 for single GPU
        self.batch_size = self.kwdict["batch"] # batch size
        self.nbs = self.kwdict["nbs"]  # nominal batch size
        self.num_workers = self.kwdict["num_workers"] # how many subprocesses are used to load data in parallel
        self.epochs = self.kwdict["epochs"] # number of epochs to train
        self.patience = self.kwdict["patience"] # number of epochs with no improvement after which training will be stopped

        self.warmup_epochs = self.kwdict["warmup_epochs"] # number of warmup epochs
        self.warmup_bias_lr = self.kwdict["warmup_bias_lr"] # initial learning rate for bias
        self.warmup_momentum = self.kwdict["warmup_momentum"] # initial momentum for optimizer
        self.weight_decay = self.kwdict["weight_decay"] # weight decay for optimizer
        self.momentum = self.kwdict["momentum"] # momentum for optimizer
        self.optimizer_choice = self.kwdict["optimizer"]
        self.scheduler_choice = self.kwdict["scheduler"]

        self.scheduler_params = {
            "lrf": self.kwdict["lrf"],  # final learning rate factor
            "gamma": self.kwdict["gamma"],  # must be used in `Exponential`
            "power": self.kwdict["power"]  # must be used in `Polynomial`
        }

        self.lr0 = self.kwdict["lr0"] # initial learning rate
        self.amp = self.kwdict["amp"] # whether to use automatic mixed precision training

        self.quant_enable = quant_enable
        if self.quant_enable:
            self.ptq = self.kwdict["ptq"] # whether to use nvidia pytorch_quantization to post training quantize
            self.qat = self.kwdict["qat"] # whether to use nviida pytorch_quantization to quantize aware train
            self.quant_desc_weight=quant_desc_weight
            self.quant_desc_input=quant_desc_input

        else:
            self.ptq = False
            self.qat = False

        self.train_loader = self.get_loader(
            self.train_dataset, mode="train"
        )
        self.val_loader = self.get_loader(
            self.val_dataset, mode="val"
        )

        self.start_epoch = 0
        self.best_fitness = None

        self.best: Optional[Path] = None # should be defined in method `train`
        self.last: Optional[Path] = None # should be defined in method `train`
        self.onnx: Optional[Path] = None # should be defined in method `train`
        self.save_dir: Optional[Path] = None # should be defined in method `train`
        self.fitness: Optional[float] = None # should be defined in method `train`

        print(f"If you want to reset the trainer config, please move to the {cfg}")

    def replace_to_quant_modules(self, module=None):
        """
        Replace standard PyTorch modules with their quantized counterparts.

        This method recursively traverses all submodules of the given model and replaces
        standard Conv2d and Linear layers with their quantized versions while preserving
        the original weight parameters. Supported replacements include:
        - nn.Conv2d -> quant_nn.QuantConv2d  
        - nn.Linear -> quant_nn.QuantLinear
        - Conv2dWithConstraint -> QuantConv2dWithConstraint
        - LinearWithConstraint -> QuantLinearWithConstraint

        Args:
            module (nn.Module, optional): The module to process. Defaults to `self.model`.

        Raises:
            AssertionError: If the `pytorch_quantization` module is not installed.

        Note:
            This method modifies the model structure in-place and should typically be called
            before starting quantization-aware training (QAT).
        """
        assert self.quant_enable, f"pytorch_quantization module has not found, you can't use the `replace_to_quant_modules` method!!!"
        if module is None:
            module = self.model
        
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d) and type(child) == nn.Conv2d:
                quant_conv = quant_nn.QuantConv2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    child.stride,
                    child.padding,
                    child.dilation,
                    child.groups,
                    bias=child.bias is not None,
                    quant_desc_weight=self.quant_desc_weight,
                    quant_desc_input=self.quant_desc_input
                )
                quant_conv.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    quant_conv.bias.data.copy_(child.bias.data)
                setattr(module, name, quant_conv)
            
            elif isinstance(child, Conv2dWithConstraint):
                quant_conv_constraint = QuantConv2dWithConstraint(
                    child.in_channels, 
                    child.out_channels, 
                    child.kernel_size, 
                    max_norm=child.max_norm, 
                    stride=child.stride, 
                    padding=child.padding, 
                    dilation=child.dilation, 
                    groups=child.groups, 
                    bias=child.bias is not None, 
                    quant_desc_weight=self.quant_desc_weight, 
                    quant_desc_input=self.quant_desc_input
                )
                quant_conv_constraint.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    quant_conv_constraint.bias.data.copy_(child.bias.data)
                setattr(module, name, quant_conv_constraint)

            elif isinstance(child, nn.Linear) and type(child) == nn.Linear:
                quant_linear = quant_nn.QuantLinear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    quant_desc_weight=self.quant_desc_weight,
                    quant_desc_input=self.quant_desc_input
                )
                quant_linear.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    quant_linear.bias.data.copy_(child.bias.data)
                setattr(module, name, quant_linear)
            
            elif isinstance(child, LinearWithConstraint):
                quant_linear_constraint = QuantLinearWithConstraint(
                    child.in_features,
                    child.out_features,
                    max_norm=child.max_norm, 
                    bias=child.bias is not None,
                    quant_desc_weight=self.quant_desc_weight,
                    quant_desc_input=self.quant_desc_input
                )
                quant_linear_constraint.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    quant_linear_constraint.bias.data.copy_(child.bias.data)
                setattr(module, name, quant_linear_constraint)

            else:
                self.replace_to_quant_modules(child)

    @torch.no_grad()
    def collect_stats(self, num_batches: Optional[int]=None):
        """
        Collect quantization statistics for calibration.

        This method is used during the calibration phase of post-training quantization (PTQ).
        It performs forward passes on the training data to gather activation statistics,
        which are used to determine quantization parameters such as scale factors and zero-points.

        Args:
            num_batches (int, optional): Number of batches to use for calibration.
                If None, the entire training dataset will be used.

        Raises:
            AssertionError: If the `pytorch_quantization` module is not installed.

        Note:
            - This method temporarily disables quantization and enables calibration mode.
            - Typically called during the PTQ workflow before invoking `compute_amax`.
            - Model parameters are not updated during calibration (decorated with @torch.no_grad()).
        """
        assert self.quant_enable, f"pytorch_quantization module has not found, you can't use the `collect_stats` method!!!"
        for _, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()
    
        for i, (X, _) in enumerate(self.train_loader):
            X = X.to(self.device)
            self.model(X)
            if isinstance(num_batches, int):
                if i > num_batches:
                    break
        
        # Disable calibrators
        for _, module in self.model.named_modules():
            if isinstance(module, quant_nn.torch.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    @torch.no_grad()
    def compute_amax(self, **kwargs):
        """
        Compute and load quantized activation maxima (amax).

        This method calculates the quantization parameters—specifically, the activation maxima—
        based on statistics gathered by `collect_stats`. These values are critical for determining
        the quantization scale factors. Different calibration methods (e.g., MaxCalibrator,
        HistogramCalibrator) may require specific arguments.

        Args:
            **kwargs: Additional keyword arguments passed to `load_calib_amax`,
                depending on the type of calibrator being used, such as `method="entropy"`, `method="mse"`, `method="percentile", percentile=90)`

        Raises:
            AssertionError: If the `pytorch_quantization` module is not installed.

        Note:
            - This method must be called after `collect_stats`.
            - It moves the model to the target device, which is important for quantization.
            - Different calibrators may require different parameters, such as `percentile`.
        """
        assert self.quant_enable, f"pytorch_quantization module has not found, you can't use the `compute_amax` method!!!"
        # Load calib result
        for _, module in self.model.named_modules():
            if isinstance(module, quant_nn.torch.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)
        self.model.to(self.device) # important!!!!!!!!

    def train(self):
        """
        Main training routine to be implemented by subclasses.

        This method must be overridden in derived classes and should include the full
        training loop logic, such as forward pass, loss computation, backward pass, 
        and parameter updates. A typical implementation should cover:
        - Core training loop logic
        - Periodic validation and metric evaluation
        - Model checkpointing
        - Early stopping logic
        - Learning rate scheduling

        Raises:
            NotImplementedError: This method must be implemented in a subclass.

        Note:
            When implementing this method, `setup_train()` should be called first to
            initialize the optimizer, scheduler, and other training components.
        """
        raise NotImplementedError("Not Implemented.")

    def get_loss(self, batch):
        """
        Compute the loss value for a given batch of data.

        This method must be implemented in a subclass and is responsible for defining
        the loss computation logic. Typically, this includes performing a forward pass 
        through the model and applying one or more loss functions.

        Args:
            batch: A single batch of data, usually containing input features and labels.

        Returns:
            The computed loss value. Can be a single tensor or a dictionary containing
            multiple loss components.

        Raises:
            NotImplementedError: This method must be implemented in a subclass.

        Note:
            The implementation of this method depends on the task type 
            (e.g., classification, regression, detection) and the loss function(s) used.
        """
        raise NotImplementedError("Not Implemented.")

    @torch.inference_mode()
    def validate(self):
        """
        Performs model validation.

        This method must be implemented in a subclass and is used to evaluate the model's
        performance on the validation set. It should be decorated with @torch.inference_mode()
        to improve inference efficiency and reduce memory usage.

        Returns:
            Validation metrics — typically a scalar or a dictionary of multiple metrics.

        Raises:
            NotImplementedError: This method must be implemented in a subclass.

        Note:
            - This method should set the model to evaluation mode using `model.eval()`.
            - Model parameters must not be updated during validation.
            - The returned metrics are used for model selection and early stopping.
        """
        raise NotImplementedError("Not Implemented.")

    def setup_train(self):
        """
        Sets up the components required for training.

        This method initializes various components needed during training, including:
        - Gradient scaler for automatic mixed precision (AMP)
        - Optimizer (selected based on configuration)
        - Learning rate scheduler
        - Early stopping mechanism
        - Gradient accumulation parameters

        It also automatically adjusts training settings based on configuration and
        model characteristics — for example, reducing the initial learning rate during 
        quantization-aware training (QAT).

        Note:
            - This method is typically called at the beginning of the `train()` method.
            - AMP compatibility is automatically checked based on the model and device.
            - If QAT is enabled, the learning rate is reduced to 10% of the original.
        """
        if self.amp:
            self.amp = check_amp(self.model)
        self.scaler = torch.amp.GradScaler(self.device, enabled=self.amp)

        self.accumulate = max(round(self.nbs / self.batch_size), 1) # accumulate loss before optimizing
        scaled_weight_decay = self.weight_decay * self.batch_size * self.accumulate / self.nbs # scale weight_decay
        if not self.qat:
            self.optimizer = self.build_optimizer(
                choice=self.optimizer_choice, 
                lr=self.lr0, 
                momentum=self.momentum, 
                decay=scaled_weight_decay, 
            )
        else:
            self.optimizer = self.build_optimizer(
                choice=self.optimizer_choice, 
                lr=self.lr0 * 0.1, 
                momentum=self.momentum, 
                decay=scaled_weight_decay, 
            ) # qat should decrease the initial lr

        # Scheduler
        self.setup_scheduler(self.scheduler_choice, scheduler_params=self.scheduler_params)
        self.stopper, self.stop = EarlyStopping(patience=self.patience), False
        self.scheduler.last_epoch = self.start_epoch - 1 # should start from -1, and auto increase with epoch

    def get_loader(self, dataset: Dataset,
                   mode: Literal["train", "val"]="train"):
        """
        Creates a data loader.

        Constructs a DataLoader instance based on the specified mode (training or validation).
        In training mode, data shuffling is enabled; in validation mode, data is loaded sequentially.

        Args:
            dataset (Dataset): The dataset to be loaded.
            mode (Literal["train", "val"]): The data loading mode.
                - "train": Enables shuffling for training.
                - "val": Disables shuffling for validation.

        Returns:
            DataLoader: A configured DataLoader instance.

        Note:
            - Batch size and number of worker processes are retrieved from the configuration.
            - Data is shuffled during training to improve generalization.
            - Data order is preserved during validation to ensure reproducibility.
        """
        shuffle = mode == "train"

        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
                            shuffle=shuffle)
        
        return loader

    def build_optimizer(self, choice: Literal["Adam", "Adamax", "AdamW", "NAdam", 
                                              "RAdam", "RMSProp", "SGD"]="Adam", 
                                              lr: float=0.001, momentum: float=0.9, decay: float=1e-5):
        """
        Builds and configures the optimizer.

        This method creates an optimizer instance based on the specified type,
        and groups model parameters to apply different weight decay strategies:
        - Group 0: Regular weights (full weight decay applied)
        - Group 1: BatchNorm and logit_scale parameters (no weight decay)
        - Group 2: Embedding layer weights (smaller weight decay)
        - Group 3: Bias parameters (no weight decay)

        Args:
            choice (str): Optimizer type. Supported options:
                "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD"
            lr (float): Initial learning rate, default 0.001
            momentum (float): Momentum parameter, default 0.9
            decay (float): Weight decay parameter, default 1e-5

        Returns:
            torch.optim.Optimizer: Configured optimizer instance

        Raises:
            NotImplementedError: If an unsupported optimizer type is selected

        Note:
            - Different parameter groups apply different weight decay strategies to improve training.
            - Embedding layers use a smaller weight decay (decay * 1e-2).
            - Bias parameters and BatchNorm parameters do not use weight decay.
        """
        g = [], [], [], [] # optimizer parameter groups
        n = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k) # normalization layers, i.e. BatchNorm2d
        e = tuple(v for k, v in torch.nn.__dict__.items() if "Embedding" in k) # embedding layers, i.e. Embedding()
        for module_name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname: # bias (no decay)
                    g[3].append(param)
                elif isinstance(module, e): # weight (less decay)
                    g[2].append(param)
                elif isinstance(module, n) or "logit_scale" in fullname: # weight (no decay)
                    # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
                    g[1].append(param)
                else: # weight (with decay)
                    g[0].append(param)
        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD"}

        if choice in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(torch.optim, choice, torch.optim.Adam)(g[3], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif choice == "RMSProp":
            optimizer = torch.optim.RMSprop(g[3], lr=lr, momentum=momentum)
        elif choice == "SGD":
            optimizer = torch.optim.SGD(g[3], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{choice}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )
        
        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        optimizer.add_param_group({"params": g[2], "weight_decay": decay * 1e-2}) # add g2 (Embedding weight)

        print(
            f"'optimizer:' {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[2])} weight(decay={decay*1e-2}), {len(g[0])} weight(decay={decay}), {len(g[3])} bias(decay=0.0)"
        )
        return optimizer
    
    def optimizer_step(self):
        """
        Performs the optimizer update step.

        This method executes the full optimizer update process, including:
        1. Unscaling gradients (used in automatic mixed precision training)
        2. Gradient clipping to prevent exploding gradients (max norm set to 10.0)
        3. Performing the optimizer step to update parameters
        4. Updating the gradient scaler state
        5. Zeroing gradients to prepare for the next iteration

        Note:
            - This method integrates gradient handling for automatic mixed precision training.
            - Gradient clipping helps stabilize the training process.
            - Should be called after each backward pass.
        """
        self.scaler.unscale_(self.optimizer) # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0) # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def setup_scheduler(self, choice: Literal["Cosine", "Linear", "Exponential", "Polynomial"], 
                        scheduler_params):
        """
        Sets up the learning rate scheduler.

        Creates a learning rate scheduler function based on the specified scheduler type. Supports four scheduling strategies:
        - Cosine: Cosine annealing scheduler, learning rate changes following a cosine function.
        - Linear: Linear decay scheduler, learning rate decreases linearly.
        - Exponential: Exponential decay scheduler, learning rate decays exponentially.
        - Polynomial: Polynomial decay scheduler, learning rate decays polynomially.

        Args:
            choice (str): Scheduler type, options include:
                "Cosine", "Linear", "Exponential", "Polynomial"
            scheduler_params (dict): Dictionary of scheduler parameters including:
                - "lrf": Final learning rate factor
                - "gamma": Decay factor for exponential scheduler
                - "power": Power for polynomial scheduler

        Raises:
            ValueError: If an unsupported scheduler type is selected.

        Note:
            - All schedulers support the lrf parameter to control the final learning rate.
            - The scheduler automatically updates the learning rate at the end of each epoch.
            - The starting epoch is set to start_epoch - 1 to ensure correct scheduling.
        """
        if choice == "Cosine":
            self.lr_func = lambda x: max((1 - math.cos(x * math.pi / self.epochs)) / 2, 0) * (scheduler_params["lrf"] - 1) + 1 # Cosine
        elif choice == "Linear":
            self.lr_func = lambda x: max(1 - x / self.epochs, 0) * (1 - scheduler_params["lrf"]) + scheduler_params["lrf"] # Linear
        elif choice == "Exponential":
            self.lr_func = lambda x: scheduler_params["gamma"] ** max(x, 0) * (1 - scheduler_params["lrf"]) + scheduler_params["lrf"] # Exponential
        elif choice == "Polynomial":
            self.lr_func = lambda x: max(1 - x / self.epochs, 0) ** scheduler_params["power"] * (1 - scheduler_params["lrf"]) + scheduler_params["lrf"] # Polynomial
        else:
            raise ValueError("The scheduler type is not supported!!!")
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_func)

    def get_memory(self, fraction=False):
        """
        Get the current GPU memory usage.

        This method queries the GPU memory usage status and can return either the absolute memory usage 
        or the usage as a fraction of total GPU memory. For CPU training, it returns 0.

        Args:
            fraction (bool): Whether to return memory usage as a fraction.
                - True: Returns usage as a fraction (between 0 and 1).
                - False: Returns absolute memory usage in GB.

        Returns:
            float: Memory usage.
                - If fraction=True, returns a value between 0 and 1 representing the fraction used.
                - If fraction=False, returns the absolute memory usage in gigabytes.
                - Returns 0 if device is CPU.

        Note:
            - Only applicable for CUDA devices; CPU devices always return 0.
            - Uses torch.cuda.memory_reserved() to get reserved memory.
            - Useful for monitoring memory usage during training.
        """
        memory, total = 0, 0
        if self.device.type != "cpu":
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)

    def clear_memory(self):
        """
        Clear memory caches.

        This method performs memory cleanup to free unused memory:
        1. Calls the Python garbage collector to clean up CPU memory.
        2. If using a GPU, clears the CUDA cache.

        Note:
            - Calling this periodically during training can prevent memory overflow.
            - Especially useful when handling large models or large batch sizes.
            - In CPU mode, only garbage collection is performed.
        """
        gc.collect()
        if self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def save_model(self):
        """
        Save the current model checkpoint to disk.
        
        This method serializes the training state including:
        - Current epoch
        - Best fitness score achieved so far
        - Model weights (supports both DataParallel and normal models)
        - Optimizer state
        - Timestamp of saving

        The checkpoint is saved to 'last.pt', and if the current model has the best fitness,
        it is also saved to 'best.pt'.
        """
        import io
        from copy import deepcopy
        from datetime import datetime

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.start_epoch,
                "best_fitness": self.best_fitness,
                "model": deepcopy(self.model.module) if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else deepcopy(self.model), # support parallelization 
                "optimizer": deepcopy(self.optimizer.state_dict()),
                "date": datetime.now().isoformat(),
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            print(f"Epoch {self.start_epoch} has trained a better fit model, we saved it.")
            self.best.write_bytes(serialized_ckpt)  # save best.pt

    def export_to_onnx(self):
        """
        Export the model to ONNX format.

        - Uses a single batch from the training loader as dummy input.
        - Supports dynamic batch size and some spatial dimensions.
        - Exports ONNX file to the path specified by self.onnx.
        - Prepares parameters for subsequent TensorRT export.
        """
        self.dummy_input = None
        for X, _ in self.train_loader:
            self.dummy_input = X.to(self.device)
            break
        torch.onnx.export(
            self.model.eval(), self.dummy_input, self.onnx, 
            opset_version=13, do_constant_folding=True, 
            input_names=["input"], output_names=["output"], 
            dynamic_axes={"input": {0: "batch", 2: "dim2", 3: "dim3"}, 
                          "output": {0: "batch"}}, # dynamic batch size
            verbose=False
        )
        self.tensorrt = self.save_dir / "best.engine"
        self.trt_quant = self.kwdict["trt_quant"] # whether to use TensorRT int8 quantization Deployment
    
    def export_to_tensorrt(self):
        """
        Convert the ONNX model to a TensorRT engine using the `trtexec` tool.
        This method requires that `export_to_onnx` has been run first to generate the ONNX model.
        It sets optimization, minimum, and maximum input shapes based on dummy input and batch size.
        Supports INT8 quantization if enabled via self.trt_quant.
        Prints the command output or error messages.
        """
        print("To run the `export_to_tensorrt` method, you shold first run the `export_to_onnx` method.")
        cmd = ["trtexec", f"--onnx={self.onnx}",
               f"--saveEngine={self.tensorrt}",  
               f"--optShapes=input:{self.batch_size}x{self.dummy_input.shape[1]}x{self.dummy_input.shape[2]}x{self.dummy_input.shape[3]}", 
               f"--minShapes=input:{1}x{self.dummy_input.shape[1]}x{self.dummy_input.shape[2]}x{self.dummy_input.shape[3]}", 
               f"--maxShapes=input:{self.batch_size * 2}x{self.dummy_input.shape[1]}x{self.dummy_input.shape[2]}x{self.dummy_input.shape[3]}", 
               f"--device={self.device.index}"]
        cmd.append("--int8") if self.trt_quant else cmd
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            print(f"The output of command {' '.join(cmd)} is \n {result.stdout}.")
        except FileNotFoundError:
            print("Failed to run `trtexec`: Command not found.\n"
                "Please ensure that TensorRT is correctly installed and `trtexec` is in your system PATH.")
        except subprocess.CalledProcessError as e:
            print(f"`trtexec` failed with error code {e.returncode}:\n{e.stderr}")