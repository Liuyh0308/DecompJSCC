# DecompJSCC

[English](README.md) | **简体中文**

DecompJSCC 的 PyTorch 实现，用于图像联合信源信道编码。仓库包含主模型、训练与评测代码、机制分析工具，以及 SwinJSCC、ADJSCC、SNR-EQ-JSCC、SA-RA-JSCC 和 BPG+LDPC 对比方法。

建议按 README 依次完成：**安装环境 → 准备数据集（与预训练模型权重） → 预览训练参数 → 小规模试跑 → 训练 → 评测**。以下命令均在项目根目录执行，实验结果统一放入 `outputs/`。

## 1. 安装环境

本项目仓库依赖 Python 3.11、PyTorch 2.9.1 和 CUDA 12.8 环境，具体版本见 [requirements.txt](requirements.txt)。

```bash
python -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt

# 查看 PyTorch 版本及 GPU 是否可用
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

BPG+LDPC 的复现需要 Sionna；只运行神经网络方法时无需安装：

```bash
pip install -r external_baselines/bpg_ldpc/requirements-sionna.txt
```

## 2. 准备数据和权重

`datasets/` 与 `checkpoints/` 分别用于存放数据集和模型权重。已有资源可以直接放到本地，后文示例会显式指定使用的路径。

从头训练主模型时可以不加载已有权重；分阶段训练和部分对比方法需要相应的初始化权重。评测则需要训练好的权重及其配套配置文件，两者应一起保留，以便按训练时的结构加载模型。

默认数据根目录为 `datasets/swinjscc_hr/`，目录布局如下：

```text
datasets/swinjscc_hr/
├── DIV2K/
│   ├── DIV2K_train_HR/
│   └── DIV2K_valid_HR/
├── CLIC2020/
│   ├── mobile_train/
│   └── mobile_valid/
├── CLIC2021/
│   ├── train/
│   ├── valid/
│   └── test/
├── CLIC2022/val/
└── Kodak/
```

除了上述数据集，也可以使用自己的图片数据集：

```text
datasets/custom/train/    训练图片
datasets/custom/val/      验证图片
```

在训练命令中加入 `--hr-train-dirs datasets/custom/train --hr-val-dirs datasets/custom/val` 即可使用自己的图片，多个目录用逗号分隔。训练集用于更新模型，验证集用于比较每轮训练效果和选择最佳权重，两者应分开。

检查资源目录是否存在：

```bash
python analyze.py resources validate
```

该命令只检查目录，不代表数据、权重或论文资源已经完整。

## 3. 入口和常用参数

| 入口 | 用途 | 执行方式 |
| --- | --- | --- |
| `train.py source` | DecompJSCC、SwinJSCC 及结构变体训练 | 默认训练；`--dry-run` 只打印配置 |
| `train.py fixed` | 固定 CBR 的两种 SNR 自适应对照 | 默认预览；`--run` 训练 |
| `train.py adjscc / snr_eq_jscc / sa_ra_jscc` | 外部神经基线训练 | 默认训练；`--dry-run` 预览 |
| `eval.py source / grid / bpg / reconstruction` | 评测和重建 | `--run` 才执行计算；预览仍可能读取输入文件 |
| `analyze.py <子命令>` | 机制、复杂度和结果分析 | 通常直接执行，参数见子命令帮助 |

`--C`、`--rate`、`--rates` 使用整数预算 k。当前协议下 CBR = k / 1536，不能把 `1/16` 直接传给这些参数：

| k | 32 | 64 | 96 | 128 | 192 |
| --- | --- | --- | --- | --- | --- |
| CBR | 1/48 | 1/24 | 1/16 | 1/12 | 1/8 |

SNR 单位为 dB，多值写成 `1,4,7,10,13`。`source` 训练用 `--multiple-snr`，外部基线和评测用 `--snr-db-values`。

## 4. 训练

### DecompJSCC：从预览到正式训练

`train.py source` 是 DecompJSCC 和 SwinJSCC 共用的训练入口，`--recipe-stage` 决定使用的模型结构。下面的示例让同一个 DecompJSCC 模型在多种码率预算和 SNR 条件下训练，默认使用 AWGN 信道和 Kodak 验证集。

先保留 `--dry-run` 查看配置，确认数据路径、模型参数和输出目录；这一步不会启动训练，也不需要数据或 GPU。

```bash
python train.py source \
  --recipe-stage no_modnet_sall_cross_universal \
  --data-dir datasets/swinjscc_hr --hr-eval-split kodak \
  --model-size base --channel-type awgn \
  --C 32,64,96,128,192 --multiple-snr 1,4,7,10,13 \
  --rcsb-rate-gain-mode active_rms \
  --attnres-query-mode snr_depth_logit_gate \
  --attnres-condition-gate-mode snr \
  --epochs 100 --lr 1e-4 --batch-size 8 --val-batch-size 1 \
  --num-workers 0 --seed 42 --dist-mode single --gpu-ids 0 \
  --checkpoint-path outputs/decompjscc/checkpoints \
  --exp-log-dir outputs/decompjscc/log --log-dir outputs/decompjscc/runs \
  --dry-run
```

准备好数据后去掉 `--dry-run` 开始训练。首次运行可改为 `--epochs 1 --batch-size 1`，并加上 `--max-train-batches 2 --max-val-batches 2`，用少量批次检查训练、验证和保存流程。

`--epochs` 控制训练轮数，`--lr` 控制学习率，`--batch-size` 控制每次送入模型的图片数。显存不足时可减小 batch；例如 batch 4 配合 `--gradient-accumulation-steps 2`，会累计两次梯度再更新参数。数据路径和预算范围可按前文说明调整。

上面的命令是自定义训练示例。论文的分阶段训练配置记录在 [configs/models/](configs/models/) 中，训练时需将对应设置转成命令行参数，`train.py` 没有通用 `--config` 参数。

训练时每轮都会验证并保存最近状态，验证 PSNR 提高时更新最佳权重。输出结构如下：

```text
outputs/decompjscc/
├── checkpoints/last.pth          最近一轮的模型、优化器和训练状态
├── checkpoints/best.pth          验证 PSNR 最优权重
├── checkpoints/best_psnr.pth     与 best.pth 同步保存
├── log/config.json              本次训练的实际配置
├── log/training_history.jsonl   每轮训练与验证指标
└── runs/                        TensorBoard 日志
```

通常用 `best.pth` 评测、用 `last.pth` 续训；`config.json` 记录模型与训练设置，后文评测会读取它。逐轮指标可查看 JSONL 文件，也可以通过 TensorBoard 观察训练曲线：

```bash
tensorboard --logdir outputs/decompjscc/runs
```

### 初始化、续训与模型切换

`source` 的 `--resume-checkpoint` 单独使用时，只把兼容的模型参数作为初始化，适合开始新的训练阶段。继续同一次训练时，还需加 `--resume-training-state --resume-strict`，恢复优化器、训练进度和随机状态；目前只支持单卡，并应保持原模型与数据参数一致。

例如，已经训练 100 轮后设置 `--epochs 200`，表示继续到总计 200 轮。续训到新输出目录时，原来的最佳权重不会自动复制过去，应保留父任务的 `best.pth`。

训练 SwinJSCC 时，可将 `--recipe-stage` 改为 `fixed`、`snr_adaptive` 或 `universal`，分别对应固定条件、SNR 自适应和联合自适应模型，并相应设置预算与 SNR 范围。多卡训练使用 `torchrun` 配合 `--dist-mode ddp`；`source` 的 batch 是每卡批量。

### 固定 CBR 的专用训练入口

`fixed` 用于 `decompjscc_wo_rcsb` 和 `swinjscc_sa` 的固定训练配方。它要求默认布局下的 **2663 张训练图片、24 张 Kodak 图片**，以及 `checkpoints/pretraining/swinjscc_fixed/model_best.pth` 中指定 SHA-256 的官方初始化权重；自定义数据或训练参数应使用 `source`。

```bash
# 预览 CBR=1/16 的配置；改为 --method swinjscc_sa 可切换方法
python train.py fixed --method decompjscc_wo_rcsb --rate 96 \
  --epochs 300 --output-root outputs/fixed --run-name c96
```

默认只打印计划，加 `--run` 才训练。该入口使用固定配方，不接受通用的 `--lr`、`--batch-size` 参数；数据或训练设置需要自定义时，使用 `source`。输出权重文件名为 `model_last.pth` 和 `model_best.pth`，完整目录会在计划中打印。

### 外部神经基线

外部神经基线使用独立的网络与训练入口。ADJSCC 可以从头训练，SNR-EQ-JSCC 和 SA-RA-JSCC 默认需要 SwinJSCC fixed 初始化。下面以固定 CBR=1/16 的 ADJSCC 为例，准备好资源后去掉 `--dry-run` 即可训练：

```bash
python train.py adjscc --rate 96 --epochs 300 \
  --data-dir datasets/swinjscc_hr --snr-db-values 1,4,7,10,13 \
  --learning-rate 1e-4 --global-batch-size 16 --micro-batch-size 8 \
  --output-root outputs/adjscc --dry-run
```

这些入口使用 `--learning-rate`、`--global-batch-size` 和 `--micro-batch-size`，数据目录参数与 `source` 相同。上例在单卡下每次处理 8 张图片，累积两次后更新参数，有效 batch 为 16。

切换到 `snr_eq_jscc` 或 `sa_ra_jscc` 时，用 `--fixed-init-checkpoint` 指定初始化权重。ADJSCC、SNR-EQ 每个 `--rate` 单独训练，SA-RA 则用 `--rates` 让一个模型覆盖多个预算。输出位于指定根目录的 `checkpoints/`、`exp_log/` 和 `runs/` 下，预览会打印完整路径。

## 5. 评测

### 评测自己训练的 DecompJSCC 或 SwinJSCC

评测使用训练生成的权重和配套 `config.json`，配置文件用于恢复模型结构。下面沿用训练示例的输出，在 Kodak 上评测 3 档预算、5 个 SNR，每个条件重复 3 次噪声采样，得到 15 组条件的平均结果：

```bash
python eval.py source \
  --checkpoint-path outputs/decompjscc/checkpoints/best.pth \
  --config-path outputs/decompjscc/log/config.json \
  --data-dir datasets/swinjscc_hr --hr-eval-split kodak \
  --rates 32,96,192 --snr-db-values 1,4,7,10,13 \
  --batch-size 1 --noise-realizations 3 --seed 20260811 \
  --strict-checkpoint --gpu-id 0 \
  --output-json outputs/eval/decompjscc_kodak.json --run
```

修改 `--rates` 和 `--snr-db-values` 可更换评测条件，固定预算模型只评测对应的 rate。使用自己的图片时增加 `--hr-val-dirs datasets/custom/val`；使用 CLIC2021 时改为 `--hr-eval-split clic2021`。想先检查流程，可缩小为一个 rate、一个 SNR，并加 `--max-val-batches 1`。

显式传入权重、配置和数据路径，可避免依赖旧资源索引。`--strict-checkpoint` 用于检查权重与模型结构是否匹配，建议与训练时保存的配置一起使用。

输出 JSON 包含 `results`（条件均值）、`realization_results`（每次噪声采样）和 `per_image_results`（逐图记录）。此入口计算 **PSNR、SSIM 和 CBR**；其中 `ssim` 不是 MS-SSIM。

### 其他评测功能

`eval.py grid` 用于统一比较不同方法，并额外计算 RGB 五尺度 MS-SSIM；`eval.py reconstruction` 用于导出指定条件下的重建结果。这些功能仍依赖资源索引或论文选择文件，当前精简仓库未包含相应的 `configs/paper_resources.json` 和论文数据包。自己的 DecompJSCC、SwinJSCC 模型可先按上例使用 `source` 评测。

BPG+LDPC 使用 `eval.py bpg`，除 Sionna 外，还需要可运行的 `bpgenc`、`bpgdec` 和 AMC 校准文件。默认论文校准文件未随当前仓库保留，需要用 `--mcs-calibration` 指定已有文件；相关参数见 `python eval.py bpg --help`。

## 6. 分析和结果汇总

### 参数量与推理时延

训练完成后，可以从模型开销和重建质量两方面分析结果。下面使用训练好的 DecompJSCC，在合成 256×256 输入上统计参数量、模型状态大小和推理时延，因此无需准备图片数据：

```bash
python analyze.py profile-role --role decompjscc \
  --checkpoint outputs/decompjscc/checkpoints/best.pth \
  --model-size base --rates 32,96,192 --image-size 256 \
  --device cuda:0 --warmup 20 --iterations 80 \
  --output-json outputs/analysis/profile.json \
  --output-markdown outputs/analysis/profile.md
```

结果同时写入 JSON 和 Markdown：前者便于后续处理，后者可以直接查看。只统计参数量和状态大小可加 `--skip-latency --device cpu`；比较推理时延时，应保持设备、输入尺寸和 batch 一致。

机制分析使用 `analyze.py routing`，用于比较正常 SNR 条件和关闭 SNR 条件后的路由表现。它需要训练权重与验证图片，具体条件可通过 `python analyze.py routing --help` 配置。

### 将评测 JSON 汇总为 CSV

`analyze.py summary` 读取已有评测结果，整理条件均值和逐图指标，不会重新运行模型。对于前面的完整 Kodak `source` 评测，可以这样导出 CSV：

```bash
python analyze.py summary \
  --input decompjscc:kodak:outputs/eval/decompjscc_kodak.json \
  --expected-rates 32,96,192 --expected-snr-values 1,4,7,10,13 \
  --expected-realizations 3 --expected-images kodak=24 \
  --output-dir outputs/analysis/summary
```

输出包含 `summary.json`、`cell_summary.csv`、`realization_results.csv` 和 `per_image_results.csv`。如果用了自己的图片、不同条件或少量 batch，需同步修改预期图像数和条件。可重复传入 `--input 方法:数据集:结果路径` 添加其他方法；配对比较要求图像集合、条件和噪声种子一致。

## 7. 代码导航和检查

```text
train.py / eval.py / analyze.py   三个命令行入口
models/                          DecompJSCC 模型及组件
SwinJSCC_base/                   共用 Swin 骨干、信道和损失
experiments/                     训练、评测、机制与统计实现
external_baselines/              对比方法及 BPG 源码
configs/models/                  模型和分阶段训练配置记录
tests/                           测试
```

日常使用从三个根目录入口开始；需要修改算法时查看 `models/`，调整训练或评测流程时查看 `experiments/`。不同子命令的参数通过各自的帮助查看：

```bash
python train.py source --help
python train.py fixed --help
python train.py adjscc --help
python eval.py source --help
python eval.py grid --help
python analyze.py routing --help
python analyze.py profile-role --help
```

可先运行不依赖资源包的训练、分析和统计测试：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -B -m unittest \
  tests.test_training tests.test_analysis tests.test_statistics
```

