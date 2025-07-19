import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import lpips
import kornia
try:
    import wandb
    use_wandb = True
except ImportError:
    print("wandb module not installed!!! If you want to install it, see https://pypi.org/project/wandb/")
    use_wandb = False

from .utils import *
from .modules import *
from .trainer import BaseTrainer


# PSNR and SSRM and LLPS
class ImGenTrainer(BaseTrainer):
    def __init__(self, model, train_dataset, val_dataset, cfg = BASE_DIR / "trainer_imgen_cfg.yml"):
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
            val_loss, perplexity, psnr, ssim = self.validate()
            self.save_dir = increment_path(Path(self.kwdict["save_dir"]), exist_ok=self.kwdict["exist_ok"])  # increment save_dir
            self.onnx = self.save_dir / "best.onnx"
            print(f"After PTQ, the val loss is {val_loss}, the perplexity is {perplexity}, the Peak Singal-to-Ratio is {psnr}, the Structure Similarity Index is {ssim}")
        elif self.qat:
            self.replace_to_quant_modules()
            self.model.to(self.device) 
            self.train()

    def train(self, use_wandb: bool=False):
        self.use_wandb = use_wandb & self.use_wandb
        self.setup_train()
        if self.use_wandb:
            wandb.init(
                project="imgen-task", 
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

                if self.use_wandb:
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
            if self.use_wandb:
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

            self.val_loss, perplexity, self.fitness, ssim= self.validate()

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

        if self.use_wandb:
            wandb.finish()

        # After train we can export onnx
        self.onnx = self.save_dir / "best.onnx"

    @torch.inference_mode()
    def validate(self):
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

        if self.use_wandb:
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

    # Independent Function
    def train_multi_gpus(self, gpu_id: int, use_wandb: bool=False):
        from torch.utils.data.distributed import DistributedSampler
        from torch.nn.parallel import DistributedDataParallel as DDP
        import torch.distributed as dist

        self.use_wandb = use_wandb & self.use_wandb
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
        def validate_multi_gpus():
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

            if self.use_wandb:
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
            if self.use_wandb:
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
                    if self.use_wandb:
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
                if self.use_wandb:
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

            self.val_loss, perplexity, self.fitness, ssim= validate_multi_gpus()

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
            if self.use_wandb:
                wandb.finish()