# DecompJSCC

DecompJSCC 的 PyTorch 实现，用于图像联合信源信道编码，包含训练、评测、机制分析，以及 SwinJSCC、ADJSCC、SNR-EQ-JSCC、SA-RA-JSCC 和 BPG+LDPC 对比方法。

## 安装

使用 Python 3.11，依赖版本见 [requirements.txt](requirements.txt)。以下命令在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
```

BPG+LDPC 额外安装：

```bash
pip install -r external_baselines/bpg_ldpc/requirements-sionna.txt
```

## 项目结构

```text
train.py              训练入口
eval.py               评测与重建入口
analyze.py            机制、复杂度和结果分析入口
models/               DecompJSCC 模型
SwinJSCC_base/         共用 Swin 骨干与信道
experiments/          训练、评测和分析实现
external_baselines/   对比方法
configs/models/       模型与训练配置记录
datasets/             数据集存放位置
checkpoints/          模型权重存放位置
tests/                测试
```

## 使用

数据和权重需自行准备，分别放入 `datasets/` 和 `checkpoints/`。默认数据根目录为 `datasets/swinjscc_hr/`，训练使用 DIV2K、CLIC，验证使用 Kodak；可用 `--hr-train-dirs` 和 `--hr-val-dirs` 指定图像目录。

预览 DecompJSCC 训练配置：

```bash
python train.py source \
  --recipe-stage no_modnet_sall_cross_universal \
  --C 32,64,96,128,192 --multiple-snr 1,4,7,10,13 \
  --rcsb-rate-gain-mode active_rms \
  --attnres-query-mode snr_depth_logit_gate \
  --attnres-condition-gate-mode snr \
  --checkpoint-path outputs/train/checkpoints \
  --exp-log-dir outputs/train/log --log-dir outputs/train/runs \
  --dry-run
```

准备好数据后，去掉 `--dry-run` 开始训练。初始化权重用 `--resume-checkpoint` 指定；同任务续训还需 `--resume-training-state --resume-strict`，仅支持单卡模式。

训练入口支持 `source`、`fixed`、`adjscc`、`snr_eq_jscc`、`sa_ra_jscc`。完整参数通过帮助查看：

```bash
python train.py source --help
python train.py fixed --help
python eval.py --help
python analyze.py --help
```

`train.py fixed` 和 `eval.py grid/source/bpg/reconstruction` 需加 `--run` 才执行。`--C`、`--rate`、`--rates` 表示潜变量宽度 k，当前模型的 CBR = k / 1536。

部分评测及资源下载仍依赖 `configs/paper_resources.json` 和论文数据文件；当前精简仓库未包含这些资源，使用相关功能前需补齐。
