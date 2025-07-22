# lightrainer
Simple Trainer for PyTorch Deep Learning

## 📘 文档

<details open>
<summary>安装</summary>

1. 精简版安装
> 将仓库克隆并在一个[**Python ≥ 3.12.0**](https://www.python.org/)的环境中安装依赖项。请确保你已安装[**PyTorch ≥ 2.7.0+cu128**](https://pytorch.org/get-started/locally/) 。

```bash
git clone https://github.com/fordelkon/lightrainer.git

cd lightrainer

pip install -r requirements.txt
```

2. 完备版安装（在精简版的基础上）
-  安装[`pytorch_quantization`库](https://github.com/NVIDIA/TensorRT/blob/main/tools/pytorch-quantization/README.md)

- 安装[`wandb`库](https://pypi.org/project/wandb/)

-  安装[`TensorRT`命令行](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html)
</details>

<details open>
<summary>代码说明</summary>

- **基类训练器（BaseTraner）**
    - 基类训练器提供训练所需要的基本步骤: `train`，`validate`，`get_loss`，`get_loader`，`save_model` 等方法需要我们在不同任务中自定义需求，其余方法比如一些是模型量化所需要的方法`replace_to_quant_modules`、`collect_stats`和`compute_max`，还有一些是无需改动的配置方法以及优化方法如`setup_train`、`build_optimizer`、`setup_scheduler`、`optimizer_step`、`get_memory`、`clear_memory`，最后一些是训练后的以不同的文件格式导出如`export_to_onnx`和`export_to_tensorrt`。

- **训练参数配置（YaML文件）**
    - 训练参数可以通过设计不同的`.yml`文件来进行配置，最基本的可设定参数如下
    ```yaml
    save_dir: runs_eeg  # (str) 用于保存实验结果的目录
    device: 0           # (int) 使用的设备编号，例如 0、1 或 2，表示使用哪块 GPU
    batch: 32           # (int) 批量大小
    nbs: 64             # (int) 标准批量大小（用于归一化学习率）
    num_workers: 4      # (int) 用于并行加载数据的子进程数量
    patience: 100       # (int) 在没有提升的情况下，最多等待多少个 epoch 后停止训练
    epochs: 200         # (int) 总训练轮数
    exist_ok: True      # (bool) 是否允许覆盖已存在的实验目录
    amp: False          # (bool) 是否使用自动混合精度训练（AMP）
    ptq: False          # (bool) 是否使用 NVIDIA 的 pytorch_quantization 进行训练后量化（PTQ）
    num_batches: None   # (None 或 int) 在 PTQ 中，用于收集统计信息的批次数量
    method: entropy     # (str) 在 PTQ 中用于决定最大绝对值的方法，可选值=[entropy, max, percentile, mse]
    qat: False          # (bool) 是否使用 NVIDIA 的 pytorch_quantization 进行量化感知训练（QAT）
    trt_quant: False    # (bool) 是否使用 TensorRT 的 int8 量化

    optimizer: Adam     # (str) 使用的优化器，可选值=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp]
    scheduler: Cosine   # (str) 学习率调度器类型，可选值=[Cosine, Linear, Exponential, Polynomial]
    lr0: 0.01           # (float) 初始学习率（如 SGD=1E-2，Adam=1E-3）
    lrf: 0.01           # (float) 最终学习率为初始学习率乘以该值
    gamma: 0.9          # (float) < 1.0 时用于指数衰减的因子（每轮学习率乘以 gamma）
    power: 2.0          # (float) 用于多项式调度的曲线形状控制因子
    momentum: 0.937     # (float) SGD 的动量/Adam 的 beta1 参数
    weight_decay: 0.0005 # (float) 优化器的权重衰减参数，典型值为 5e-4

    warmup_epochs: 3.0       # (float) 预热训练的 epoch 数（支持小数）
    warmup_momentum: 0.8     # (float) 预热阶段的初始动量
    warmup_bias_lr: 0.1      # (float) 预热阶段偏置项的初始学习率
    ```
    但是我们仍然可以继续在上述基本配置参数中继续添加一些特定的参数以适应特定任务。比如说在`VAVQE`的图像生成任务中，我们可以在上述参数条件下添加以下额外参数
    ```yaml
    recon_weight: 10.0
    lpips_weight: 1.0
    vq_weight: 0.5
    ```
    表示对于不同损失所赋予的权重，在自定义任务的训练器中调用`self.kwdict[key]`来进行使用。

- **分类训练器（CLSTrainer）**
    - 自定义分类训练器中的训练函数
    ```python 
    def train(self, use_wandb: bool=False):
        ......

        for epoch in range(1, self.epochs + 1):
            .......
            
            for i, batch in pbar:
                ......

                # Forward
                _, loss = self.get_loss(batch)
                self.train_loss += loss.item()
                
                # Backward
                self.scaler.scale(loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                ......


            ......

            self.val_loss, self.fitness = self.validate()

            ......

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
    ```

- **图像生成训练器（ImGenTrainer）**
    - 自定义图像生成训练器的训练函数（针对`VQVAE`）
    ```python
    def train(self, use_wandb: bool=False):
        ......

        for epoch in range(1, self.epochs + 1):
            ......
            
            for i, batch in pbar:
                ......

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

                ......

            ......

            self.train_loss /= nb
            self.train_loss_recon /= nb
            self.train_loss_lpips /= nb
            self.train_loss_vq /= nb
            self.train_perplexity /= nb

            ......

            self.val_loss, perplexity, self.fitness, ssim= self.validate()

            ......

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
    ```
</details>

<details open>
<summary>总结</summary>

对比两个特定任务的`train`方法可以看出，该方法的基本框架是没有什么变化的，主要变化的是`get_loss`和`validate`方法的返回结果，由于不同模型的损失以及选择保存模型的评判标准不同，所以这两个方法具有很强的灵活性。当然有些时候需要对`dataset`进行进一步的处理比如`padding`等方法，需要再定义`collate_fn`方法再`get_loader`里使用。有些情况我们只想保存模型的一部分，也可以根据自己**model**的结构自定义`save_model`方法来保存自己想保存的部分，也就是说后续可以根据上述两种范例继续定义更多针对特定任务的训练器。
</details>

<details open>
<summary>使用教程</summary>

- 项目目录结构
<pre lang="markdown">
```
📁 lightrainer
.
├── cls_trainer.py              # 分类训练器
├── imgen_trainer.py            # 图像生成训练器
├── modules.py                  # 模型结构或模块定义
├── README.md                   # 项目说明文档
├── requirements.txt            # 依赖包列表
├── trainer_cls_cfg.yml         # 分类任务配置文件
├── trainer_imgen_cfg.yml       # 图像生成任务配置文件
├── trainer.py                  # 基类训练器
└── utils.py                    # 工具函数
```
</pre>

- 多模态图模型节点分类任务（以`MMGCN`为例，[colab notebook链接](https://colab.research.google.com/drive/1cWtSiUQm0J7kUJQyveMcdxTXJF8L5jT0?usp=sharing)）
    - 将`lightrainer`项目克隆下来之后新建立要训练的`MMGCN`模型文件与该项目同级（文件排布如下所示）
<pre lang="markdown">
📦 myprojects
├── 📁 lightrainer 
├── IEMOCAP_features.pkl
└── mmgcn.ipynb # MMGCN分类
</pre>

- 图像生成任务（以`VQVAE`为例）
    - 将`lightrainer`项目克隆下来之后新建立要训练的``模型文件与该项目同级（文件排布如下所示）
<pre lang="markdown">
📦 myprojects
├── 📁 lightrainer 
└── vqvae.ipynb # 多模态低秩融合分类  
</pre>
</details>