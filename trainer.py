import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sklearn
from pathlib import Path
from typing import Literal
import subprocess
from tqdm import tqdm
import math
import warnings
import gc
from typing import Optional
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
import lpips
import kornia
import wandb

from .utils import *
from .modules import *


BASE_DIR = Path(__file__).resolve().parent


class BaseTrainer:
    def __init__(self, model: nn.Module, 
                 train_dataset: Dataset, val_dataset: Dataset, 
                 cfg: Path):
        # The model should output total loss as we expected
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.kwdict = load_yml(cfg)

        # self.task = self.kwdict["task"]
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

        self.lr0 = self.kwdict["lr0"]
        self.amp = self.kwdict["amp"]

        self.ptq = self.kwdict["ptq"]
        self.qat = self.kwdict["qat"]

        self.train_loader = self.get_loader(
            self.train_dataset, mode="train"
        )
        self.val_loader = self.get_loader(
            self.val_dataset, mode="val"
        )

        self.start_epoch = 0
        self.best_fitness = None

        print(f"If you want to reset the trainer config, please move to the {cfg}")

    def replace_to_quant_modules(self, module=None):
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
        # self.model.to(self.device)
        # Enable calibrators
        for _, module in self.model.named_modules():
            if isinstance(module, quant_nn.torch.TensorQuantizer):
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
        raise NotImplementedError("Not Implemented.")

    def get_loss(self, batch):
        raise NotImplementedError("Not Implemented.")

    @torch.inference_mode()
    def validate(self):
        raise NotImplementedError("Not Implemented.")

    def setup_train(self):
        if self.amp:
            self.amp = check_amp(self.model)
        self.scaler = torch.amp.GradScaler(self.device, enabled=self.amp)

        # Optimizer
        # print("-" * 30, nbs)
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
            )

        # Scheduler
        self.setup_scheduler(self.scheduler_choice, scheduler_params=self.scheduler_params)
        self.stopper, self.stop = EarlyStopping(patience=self.patience), False
        self.scheduler.last_epoch = self.start_epoch - 1

    def get_loader(self, dataset: Dataset,
                   mode: Literal["train", "val"]="train"):
        shuffle = mode == "train"

        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
                            shuffle=shuffle)
        
        return loader

    def build_optimizer(self, choice: Literal["Adam", "Adamax", "AdamW", "NAdam", 
                                              "RAdam", "RMSProp", "SGD"]="Adam", 
                                              lr: float=0.001, momentum: float=0.9, decay: float=1e-5):
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
        self.scaler.unscale_(self.optimizer) # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0) # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def setup_scheduler(self, choice: Literal["Cosine", "Linear", "Exponential", "Polynomial"], 
                        scheduler_params):
        """ 
            scheduler_params = {
                                    "lrf": 0.1, 
                                    "gamma": 0.9, # must be used in `Exponential` 
                                    "power": 2.0 # must be used in `Polynomial`
                                }
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
        memory, total = 0, 0
        if self.device.type != "cpu":
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)

    def clear_memory(self):
        gc.collect()
        if self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def save_model(self):
        import io
        from copy import deepcopy
        from datetime import datetime

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.start_epoch,
                "best_fitness": self.best_fitness,
                "model": deepcopy(self.model.module) if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else deepcopy(self.model), 
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
        self.trt_quant = self.kwdict["trt_quant"]
    
    def export_to_tensorrt(self):
        print("To run the `export_to_torch.Tensorrt` method, you shold first run the `export_to_onnx` method.")
        cmd = ["trtexec", f"--onnx={self.onnx}",
               f"--saveEngine={self.tensorrt}",  
               f"--optShapes=input:{self.batch_size}x{self.dummy_input.shape[1]}x{self.dummy_input.shape[2]}x{self.dummy_input.shape[3]}", 
               f"--minShapes=input:{1}x{self.dummy_input.shape[1]}x{self.dummy_input.shape[2]}x{self.dummy_input.shape[3]}", 
               f"--maxShapes=input:{self.batch_size * 2}x{self.dummy_input.shape[1]}x{self.dummy_input.shape[2]}x{self.dummy_input.shape[3]}", 
               f"--device={self.device.index}"]
        cmd.append("--int8") if self.trt_quant else cmd
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
        print(f"The output of command {' '.join(cmd)} is \n {result.stdout}.")


class ClassifyTrainer(BaseTrainer):
    def __init__(self, model, train_dataset, val_dataset, quant_desc_weight = None, quant_desc_input = None, cfg = BASE_DIR / "trainer_cls_cfg.yml"):
        # The model outputs logits
        super().__init__(model, train_dataset, val_dataset, cfg)
        if self.ptq:
            assert quant_desc_weight is not None and quant_desc_input is not None, f"You are start to quantize the model, `quant_desc_weight` and `quant_desc_input` shouldn't be none"
            self.quant_desc_weight = quant_desc_weight
            self.quant_desc_input = quant_desc_input
            self.num_batches = self.kwdict["num_batches"]
            self.method = self.kwdict["method"]
            self.replace_to_quant_modules() 
            self.model.to(self.device)
            self.collect_stats(self.num_batches)
            if self.method == "percentile":
                self.compute_amax(method=self.method, percentile=90.0)
            else:
                self.compute_amax(method=self.method)
            val_loss, val_acc = self.validate()
            self.save_dir = increment_path(Path(self.kwdict["save_dir"]), exist_ok=self.kwdict["exist_ok"])  # increment save_dir
            self.onnx = self.save_dir / "best.onnx"
            print(f"After PTQ, the val loss is {val_loss}, the val acc is {val_acc}")
        elif self.qat:
            assert quant_desc_weight is not None and quant_desc_input is not None, f"You are start to quantize the model, `quant_desc_weight` and `quant_desc_input` shouldn't be none"
            self.quant_desc_weight = quant_desc_weight
            self.quant_desc_input = quant_desc_input
            self.replace_to_quant_modules()
            self.model.to(self.device) 
            self.train()

    def train(self):
        self.setup_train()
        self.model = self.model.to(self.device)

        nb = len(self.train_loader) # number of batches
        nw = max(round(self.warmup_epochs * nb), 100) if self.warmup_epochs > 0 else -1 # warmup iterations
        last_opt_step = -1
        self.optimizer.zero_grad()
        for epoch in range(1, self.epochs + 1):
            print("\n")
            print("-" * 30)
            print("\n")
            self.train_loss = 0.
            self.start_epoch = epoch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # supress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()
            
            self.model.train() # No freeze layers
            pbar = enumerate(self.train_loader)
            
            pbar = tqdm(enumerate(self.train_loader), total=nb)
            
            for i, batch in pbar:
                # Warmup 
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw] # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lr_func(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.warmup_momentum, self.momentum])

                # Forward
                X, _ = batch[0], batch[1]
                _, loss = self.get_loss(batch)
                self.train_loss += loss.item()
                
                # Backward
                self.scaler.scale(loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                pbar.set_description(
                    f"{epoch}/{self.epochs}",
                    f"{self.get_memory():.3g}G",  # (GB) GPU memory util
                )

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            print(f"All types `lr` of epoch {epoch}: {self.lr}")

            self.train_loss /= len(self.train_loader)
            print(f"Epoch {epoch}: Train loss {self.train_loss}")

            final_epoch = epoch + 1 >= self.epochs

            self.val_loss, self.fitness = self.validate()
            print(f"Epoch {epoch}: Val loss {self.val_loss}, Val Acc {self.fitness}")
            print(f"Epoch {epoch} Val Acc: {self.fitness}, before best Acc: {self.stopper.best_fitness}")
            self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
            self.best_fitness = self.stopper.best_fitness

            if self.stop:
                break

            if not final_epoch and not self.qat:
                self.save_dir = increment_path(Path(self.kwdict["save_dir"]), exist_ok=self.kwdict["exist_ok"])  # increment save_dir
                self.last, self.best = self.save_dir / "last.pt", self.save_dir / "best.pt"
                
                self.save_model()

            if self.get_memory(fraction=True) > 0.5:
                self.clear_memory() # clear if memory utilization > 50%

            print(f"\n{self.start_epoch} epochs completed!")
            print("\n")
            print("-" * 30)
            print("\n")
        self.clear_memory()

        # After train we can export onnx
        self.onnx = self.save_dir / "best.onnx"

    @torch.inference_mode()
    def validate(self):
        self.model.eval()
        pbar = enumerate(self.val_loader)
        pbar = tqdm(pbar, total=len(self.val_loader), bar_format="{l_bar}{bar:10}{r_bar}")
        y_true, y_pred = [], []

        val_loss = 0.0
        for _, batch in pbar:
            X, y = batch
            logits, loss = self.get_loss(batch)
            val_loss += loss.item()
            pred = logits.argmax(dim=1)
            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
        
        val_loss /= len(self.val_loader)
        y_true = np.array(y_true).reshape(-1)
        y_pred = np.array(y_pred).reshape(-1)

        return val_loss, sklearn.metrics.accuracy_score(y_true, y_pred)
    
    def get_loss(self, batch):
        X, y = batch[0], batch[1]
        X, y = X.to(self.device), y.to(self.device)
        logits = self.model(X)
        loss = nn.functional.cross_entropy(logits, y) 
        return logits, loss
    

# PSNR and SSRM and LLPS
class LowLevelVisionTrainer(BaseTrainer):
    def __init__(self, model, train_dataset, val_dataset, quant_desc_weight = None, quant_desc_input = None, cfg = BASE_DIR / "trainer_lowvision_cfg.yml"):
        super().__init__(model, train_dataset, val_dataset, cfg)

        if self.ptq:
            assert quant_desc_weight is not None and quant_desc_input is not None, f"You are start to quantize the model, `quant_desc_weight` and `quant_desc_input` shouldn't be none"
            self.quant_desc_weight = quant_desc_weight
            self.quant_desc_input = quant_desc_input
            self.num_batches = self.kwdict["num_batches"]
            self.method = self.kwdict["method"]
            self.replace_to_quant_modules() 
            self.model.to(self.device)
            self.collect_stats(self.num_batches)
            if self.method == "percentile":
                self.compute_amax(method=self.method, percentile=90.0)
            else:
                self.compute_amax(method=self.method)
            val_loss, perplexity, psnr, ssim = self.validate()
            self.save_dir = increment_path(Path(self.kwdict["save_dir"]), exist_ok=self.kwdict["exist_ok"])  # increment save_dir
            self.onnx = self.save_dir / "best.onnx"
            print(f"After PTQ, the val loss is {val_loss}, the perplexity is {perplexity}, the Peak Singal-to-Ratio is {psnr}, the Structure Similarity Index is {ssim}")
        elif self.qat:
            assert quant_desc_weight is not None and quant_desc_input is not None, f"You are start to quantize the model, `quant_desc_weight` and `quant_desc_input` shouldn't be none"
            self.quant_desc_weight = quant_desc_weight
            self.quant_desc_input = quant_desc_input
            self.replace_to_quant_modules()
            self.model.to(self.device) 
            self.train()
        # else:
        #     self.model.to(self.device)

    def train(self, use_wandb=False):
        self.setup_train()
        if use_wandb:
            wandb.init(
                project="lowlevelvision-task", 
                config={
                    "batch_size": self.batch_size,
                    "initial_learning_rate": self.lr0, 
                    "final_learning_rate": self.lr0 * self.scheduler_params["lrf"], 
                    "optimizer": self.optimizer_choice, 
                    "scheduler": self.scheduler_choice, 
                    "weight_decay": self.weight_decay, 
                    "warmup_epochs": self.warmup_epochs, 
                    "warmup_momentum": self.warmup_momentum, 
                    "warmup_bias_lr": self.warmup_bias_lr, 
                    "recon_weight": self.kwdict["recon_weight"], 
                    "lpips_weight": self.kwdict["lpips_weight"], 
                    "vq_weight": self.kwdict["vq_weight"],
                }
            )
        self.lpips_fn = lpips.LPIPS().to(self.device)
        self.model = self.model.to(self.device)

        nb = len(self.train_loader) # number of batches
        nw = max(round(self.warmup_epochs * nb), 100) if self.warmup_epochs > 0 else -1 # warmup iterations
        last_opt_step = -1
        self.optimizer.zero_grad()
        for epoch in range(1, self.epochs + 1):
            print("\n")
            print("-" * 30)
            print("\n")
            self.train_loss = 0.
            self.train_loss_recon = 0.
            self.train_loss_lpips = 0.
            self.train_loss_vq = 0.
            self.train_perplexity = 0.
            self.start_epoch = epoch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # supress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()
            
            self.model.train() # No freeze layers
            pbar = enumerate(self.train_loader)
            
            pbar = tqdm(enumerate(self.train_loader), total=nb)
            
            for i, batch in pbar:
                # Warmup 
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw] # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lr_func(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.warmup_momentum, self.momentum])

                # Forward
                loss_vq, loss_recon, loss_lpips, perplexity, psnr, ssim = self.get_loss(batch)
                loss = self.kwdict["vq_weight"] * loss_vq + self.kwdict["recon_weight"] * loss_recon + self.kwdict["lpips_weight"] * loss_lpips
                self.train_loss += loss.item()
                self.train_loss_recon += loss_recon.item()
                self.train_loss_lpips += loss_lpips.item()
                self.train_loss_vq += loss_vq.item()
                self.train_perplexity += perplexity.item()
                
                # Backward
                self.scaler.scale(loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                pbar.set_description(
                    f"{epoch}/{self.epochs}",
                    f"{self.get_memory():.3g}G",  # (GB) GPU memory util
                )

                if use_wandb:
                    if i % 50 == 0:
                        wandb.log(
                            {
                                "step": (epoch - 1) * nb + i, 
                                "train/loss": loss.item(), 
                                "train/recon_loss": loss_recon.item(), 
                                "train/perceptual_loss": loss_lpips.item(), 
                                "train/vq_loss": loss_vq.item(), 
                                "train/perplexity": perplexity.item(), 
                                "train/psnr": psnr.item(), 
                                "train/ssim": ssim.item(), 
                            }
                        )

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            print(f"All types `lr` of epoch {epoch}: {self.lr}")

            self.train_loss /= nb
            self.train_loss_recon /= nb
            self.train_loss_lpips /= nb
            self.train_loss_vq /= nb
            self.train_perplexity /= nb

            print(f"Epoch {epoch}: Train loss {self.train_loss}")
            if use_wandb:
                wandb.log(
                    {
                        "epoch": epoch, 
                        "train/avg_loss": self.train_loss, 
                        "train/avg_recon_loss": self.train_loss_recon, 
                        "train/avg_perceptual_loss": self.train_loss_lpips, 
                        "train/avg_perplexity": self.train_perplexity, 
                    }
                )

            final_epoch = epoch + 1 >= self.epochs

            self.val_loss, perplexity, self.fitness, ssim= self.validate(use_wandb=use_wandb)

            print(f"Epoch {epoch}: Val loss {self.val_loss}, Val Peak Singal-to-Ratio is {self.fitness}, Val Perplexity is {perplexity}, Val Structure Similarity Index is {ssim}.")
            print(f"Epoch {epoch} Val Peak Singal-to-Ratio: {self.fitness}, before best Peak Singal-to-Ratio: {self.stopper.best_fitness}")
            self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
            self.best_fitness = self.stopper.best_fitness

            if self.stop:
                break

            if not final_epoch and not self.qat:
                self.save_dir = increment_path(Path(self.kwdict["save_dir"]), exist_ok=self.kwdict["exist_ok"])  # increment save_dir
                self.last, self.best = self.save_dir / "last.pt", self.save_dir / "best.pt"
                
                self.save_model()

            if self.get_memory(fraction=True) > 0.5:
                self.clear_memory() # clear if memory utilization > 50%

            print(f"\n{self.start_epoch} epochs completed!")
            print("\n")
            print("-" * 30)
            print("\n")

        self.clear_memory()

        if use_wandb:
            wandb.finish()

        # After train we can export onnx
        self.onnx = self.save_dir / "best.onnx"

    # Independent Function
    def train_multi_gpus(self, gpu_id: int, use_wandb: bool=False):
        from torch.utils.data.distributed import DistributedSampler
        from torch.nn.parallel import DistributedDataParallel as DDP
        import torch.distributed as dist

        def get_multi_gpus_loss(batch):
            X = batch[0]
            X = X.to(gpu_id)
            X_recon, loss_vq, perplexity = self.model(X)
            # Reconstruction Loss
            loss_recon = nn.functional.mse_loss(X_recon, X)
            # LPIS Loss
            X_norm, X_recon_norm = X * 2. - 1., X_recon * 2. - 1.
            loss_lpips = lpips_fn(X_norm, X_recon_norm).mean()

            psnr = kornia.metrics.psnr(X, X_recon, max_val=1.)
            
            ssim = kornia.metrics.ssim(X, X_recon, window_size=11).mean()
        
            return (loss_vq, loss_recon, loss_lpips, 
                    perplexity, psnr, ssim)
        
        @torch.inference_mode()
        def validate_multi_gpus(use_wandb: bool):
            self.model.eval()
            pbar = enumerate(self.val_loader)
            if dist.get_rank() == 0:
                pbar = tqdm(pbar, total=len(self.val_loader), bar_format="{l_bar}{bar:10}{r_bar}")

            val_loss = 0.
            val_loss_recon = 0.
            val_loss_lpips = 0.
            val_loss_vq = 0.
            val_perplexity = 0.
            val_ssim = 0.
            val_psnr = 0.
            nb_val = len(self.val_loader)

            for _, batch in pbar:
                loss_vq, loss_recon, loss_lpips, perplexity, psnr, ssim = get_multi_gpus_loss(batch)
                loss = self.kwdict["vq_weight"] * loss_vq + self.kwdict["recon_weight"] * loss_recon + self.kwdict["lpips_weight"] * loss_lpips
                val_loss += loss.item()
                val_loss_recon += loss_recon.item()
                val_loss_lpips += loss_lpips.item()
                val_loss_vq += loss_vq.item()
                val_perplexity += perplexity.item()
                val_ssim += ssim
                val_psnr += psnr
    
            val_loss /= nb_val
            val_loss_recon /= nb_val
            val_loss_lpips /= nb_val
            val_loss_vq /= nb_val
            val_perplexity /= nb_val
            val_ssim /= nb_val
            val_psnr /= nb_val

            if use_wandb:
                if dist.get_rank() == 0:
                    wandb.log(
                        {
                            "epoch": self.start_epoch,
                            "val/avg_loss": val_loss, 
                            "val/avg_recon_loss": val_loss_recon, 
                            "val/avg_perceptual_loss": val_loss_lpips, 
                            "val/avg_vq_loss": val_loss_vq, 
                            "val/avg_psnr": val_psnr, 
                            "val/avg_ssim": val_ssim, 
                            "val/avg_perplexity": val_perplexity,
                        }
                    )
            return val_loss, val_perplexity, val_psnr, val_ssim

        self.model = self.model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[gpu_id])

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True,
                                       sampler=DistributedSampler(self.train_dataset, shuffle=True))
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, 
                                    sampler=DistributedSampler(self.val_dataset, shuffle=False))
        
        lpips_fn = lpips.LPIPS().to(gpu_id)
        
        self.setup_train()

        if dist.get_rank() == 0:
            if use_wandb:
                wandb.init(
                    project="lowlevelvision-task", 
                    config={
                        "batch_size": self.batch_size,
                        "initial_learning_rate": self.lr0, 
                        "final_learning_rate": self.lr0 * self.scheduler_params["lrf"], 
                        "optimizer": self.optimizer_choice, 
                        "scheduler": self.scheduler_choice, 
                        "weight_decay": self.weight_decay, 
                        "warmup_epochs": self.warmup_epochs, 
                        "warmup_momentum": self.warmup_momentum, 
                        "warmup_bias_lr": self.warmup_bias_lr, 
                        "recon_weight": self.kwdict["recon_weight"], 
                        "lpips_weight": self.kwdict["lpips_weight"], 
                        "vq_weight": self.kwdict["vq_weight"],
                    }
                )

        nb = len(self.train_loader) # number of batches
        nw = max(round(self.warmup_epochs * nb), 100) if self.warmup_epochs > 0 else -1 # warmup iterations
        last_opt_step = -1
        self.optimizer.zero_grad()
        for epoch in range(1, self.epochs + 1):
            if dist.get_rank() == 0:
                print("\n")
                print("-" * 30)
                print("\n")
            self.train_loss = 0.
            self.train_loss_recon = 0.
            self.train_loss_lpips = 0.
            self.train_loss_vq = 0.
            self.train_perplexity = 0.
            self.start_epoch = epoch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # supress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()
            
            self.model.train() # No freeze layers
            pbar = enumerate(self.train_loader)

            if dist.get_rank() == 0:
                pbar = tqdm(enumerate(self.train_loader), total=nb)
            
            for i, batch in pbar:
                # Warmup 
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw] # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lr_func(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.warmup_momentum, self.momentum])

                # Forward
                loss_vq, loss_recon, loss_lpips, perplexity, psnr, ssim = get_multi_gpus_loss(batch)
                loss = self.kwdict["vq_weight"] * loss_vq + self.kwdict["recon_weight"] * loss_recon + self.kwdict["lpips_weight"] * loss_lpips
                self.train_loss += loss.item()
                self.train_loss_recon += loss_recon.item()
                self.train_loss_lpips += loss_lpips.item()
                self.train_loss_vq += loss_vq.item()
                self.train_perplexity += perplexity.item()
                
                # Backward
                self.scaler.scale(loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                if dist.get_rank() == 0:
                    pbar.set_description(
                        f"{epoch}/{self.epochs}",
                        f"{self.get_memory():.3g}G",  # (GB) GPU memory util
                    )
                    if use_wandb:
                        if i % 50 == 0:
                            wandb.log(
                                {
                                    "step": (epoch - 1) * nb + i, 
                                    "train/loss": loss.item(), 
                                    "train/recon_loss": loss_recon.item(), 
                                    "train/perceptual_loss": loss_lpips.item(), 
                                    "train/vq_loss": loss_vq.item(), 
                                    "train/perplexity": perplexity.item(), 
                                    "train/psnr": psnr.item(), 
                                    "train/ssim": ssim.item(), 
                                }
                            )

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            
            if dist.get_rank() == 0:
                print(f"All types `lr` of epoch {epoch}: {self.lr}")

            self.train_loss /= nb
            self.train_loss_recon /= nb
            self.train_loss_lpips /= nb
            self.train_loss_vq /= nb
            self.train_perplexity /= nb

            if dist.get_rank() == 0:
                print(f"Epoch {epoch}: Train loss {self.train_loss}")
                if use_wandb:
                    wandb.log(
                        {
                            "epoch": epoch, 
                            "train/avg_loss": self.train_loss, 
                            "train/avg_recon_loss": self.train_loss_recon, 
                            "train/avg_perceptual_loss": self.train_loss_lpips, 
                            "train/avg_perplexity": self.train_perplexity, 
                        }
                    )

            final_epoch = epoch + 1 >= self.epochs

            self.val_loss, perplexity, self.fitness, ssim= validate_multi_gpus(use_wandb=use_wandb)

            if dist.get_rank() == 0:
                print(f"Epoch {epoch}: Val loss {self.val_loss}, Val Peak Singal-to-Ratio is {self.fitness}, Val Perplexity is {perplexity}, Val Structure Similarity Index is {ssim}.")
                print(f"Epoch {epoch} Val Peak Singal-to-Ratio: {self.fitness}, before best Peak Singal-to-Ratio: {self.stopper.best_fitness}")
            self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
            self.best_fitness = self.stopper.best_fitness

            broadcast_list = [self.stop if dist.get_rank() == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0) # broadcast stop to all ranks
            self.stop = broadcast_list[0]

            if self.stop:
                break

            if dist.get_rank() == 0:
                if not final_epoch and not self.qat:
                    self.save_dir = increment_path(Path(self.kwdict["save_dir"]), exist_ok=self.kwdict["exist_ok"])  # increment save_dir
                    self.last, self.best = self.save_dir / "last.pt", self.save_dir / "best.pt"
                    self.onnx = self.save_dir / "best.onnx"
                    self.save_model()

            if self.get_memory(fraction=True) > 0.5:
                self.clear_memory() # clear if memory utilization > 50%

            if dist.get_rank() == 0:
                print(f"\n{self.start_epoch} epochs completed!")
                print("\n")
                print("-" * 30)
                print("\n")
        self.clear_memory()

        if dist.get_rank() == 0:
            if use_wandb:
                wandb.finish()

    
    @torch.inference_mode()
    def validate(self, use_wandb: bool):
        self.model.eval()
        pbar = enumerate(self.val_loader)
        pbar = tqdm(pbar, total=len(self.val_loader), bar_format="{l_bar}{bar:10}{r_bar}")

        val_loss = 0.
        val_loss_recon = 0.
        val_loss_lpips = 0.
        val_loss_vq = 0.
        val_perplexity = 0.
        val_ssim = 0.
        val_psnr = 0.
        nb_val = len(self.val_loader)
        for _, batch in pbar:
            loss_vq, loss_recon, loss_lpips, perplexity, psnr, ssim = self.get_loss(batch)
            loss = self.kwdict["vq_weight"] * loss_vq + self.kwdict["recon_weight"] * loss_recon + self.kwdict["lpips_weight"] * loss_lpips
            val_loss += loss.item()
            val_loss_recon += loss_recon.item()
            val_loss_lpips += loss_lpips.item()
            val_loss_vq += loss_vq.item()
            val_perplexity += perplexity.item()
            val_ssim += ssim
            val_psnr += psnr
        val_loss /= nb_val
        val_loss_recon /= nb_val
        val_loss_lpips /= nb_val
        val_loss_vq /= nb_val
        val_perplexity /= nb_val
        val_ssim /= nb_val
        val_psnr /= nb_val

        if use_wandb:
            wandb.log(
                {
                    "epoch": self.start_epoch,
                    "val/avg_loss": val_loss, 
                    "val/avg_recon_loss": val_loss_recon, 
                    "val/avg_perceptual_loss": val_loss_lpips, 
                    "val/avg_vq_loss": val_loss_vq, 
                    "val/avg_psnr": val_psnr, 
                    "val/avg_ssim": val_ssim, 
                    "val/avg_perplexity": val_perplexity,
                }
            )

        return val_loss, val_perplexity, val_psnr, val_ssim
    
    def get_loss(self, batch):
        X = batch[0]
        X = X.to(self.device)
        X_recon, loss_vq, perplexity = self.model(X)
        # Reconstruction Loss
        loss_recon = nn.functional.mse_loss(X_recon, X)
        # LPIS Loss
        X_norm, X_recon_norm = X * 2. - 1., X_recon * 2. - 1.
        loss_lpips = self.lpips_fn(X_norm, X_recon_norm).mean()

        psnr = kornia.metrics.psnr(X, X_recon, max_val=1.)
        
        ssim = kornia.metrics.ssim(X, X_recon, window_size=11).mean()
    
        return (loss_vq, loss_recon, loss_lpips, 
                perplexity, psnr, ssim)