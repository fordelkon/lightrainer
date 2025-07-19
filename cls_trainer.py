import torch
from torch import nn
import numpy as np
import sklearn
from pathlib import Path
from tqdm import tqdm
import warnings
try: 
    import wandb
    use_wandb = True
except ImportError:
    print("wandb module not installed!!! If you want to install it, see https://pypi.org/project/wandb/")
    use_wandb = False

from .utils import *
from .modules import *
from .trainer import BaseTrainer


class CLSTrainer(BaseTrainer):
    def __init__(self, model, train_dataset, val_dataset, cfg = BASE_DIR / "trainer_cls_cfg.yml"):
        # The model outputs logits
        super().__init__(model, train_dataset, val_dataset, cfg)
        self.use_wandb = use_wandb
        if self.ptq:
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
            self.replace_to_quant_modules()
            self.model.to(self.device) 
            self.train()

    def train(self, use_wandb: bool=False):
        self.use_wandb = self.use_wandb & use_wandb
        self.setup_train()
        if self.use_wandb:
            wandb.init(
                project="cls-task", 
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
                }
            )
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
                self.scheduler.step() # start schuduler <-> `last_epoch=0``
            
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
            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch, 
                        "train/avg_loss": self.train_loss, 
                    }
                )

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
        if self.use_wandb:
            wandb.finish()

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
        if self.use_wandb:
            wandb.log(
                {
                    "epoch": self.start_epoch,
                    "val/avg_loss": val_loss, 
                }
            )
        y_true = np.array(y_true).reshape(-1)
        y_pred = np.array(y_pred).reshape(-1)

        return val_loss, sklearn.metrics.accuracy_score(y_true, y_pred)
    
    def get_loss(self, batch):
        X, y = batch[0], batch[1]
        X, y = X.to(self.device), y.to(self.device)
        logits = self.model(X)
        loss = nn.functional.cross_entropy(logits, y) 
        return logits, loss