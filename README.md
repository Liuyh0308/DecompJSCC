# DecompJSCC

**English** | [简体中文](README.zh-CN.md)

A PyTorch implementation of DecompJSCC for joint source-channel coding of images. The repository includes the main model, training and evaluation code, mechanism analysis tools, and the SwinJSCC, ADJSCC, SNR-EQ-JSCC, SA-RA-JSCC, and BPG+LDPC baselines.

Follow this workflow: **install the environment → prepare datasets and any required pretrained weights → preview the training configuration → run a small trial → train → evaluate**. Run all commands from the repository root and save experiment results under `outputs/`.

## 1. Environment setup

This project uses Python 3.11, PyTorch 2.9.1, and CUDA 12.8. See [requirements.txt](requirements.txt) for the exact dependency versions.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt

# Check the PyTorch version and GPU availability
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

The BPG+LDPC implementation also requires Sionna. This dependency is optional if you only use neural methods:

```bash
pip install -r external_baselines/bpg_ldpc/requirements-sionna.txt
```

## 2. Datasets and checkpoints

Store datasets in `datasets/` and model weights in `checkpoints/`. You can place existing resources in these directories; the examples below specify their paths explicitly.

The main model can be trained from scratch without loading existing weights. Staged training and some baselines require the appropriate initialization weights. Evaluation needs trained weights and their matching configuration file. Keep both together so the model can be loaded with the same architecture used during training.

The default data root is `datasets/swinjscc_hr/`, with the following layout:

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

You can also use your own images:

```text
datasets/custom/train/    Training images
datasets/custom/val/      Validation images
```

Add `--hr-train-dirs datasets/custom/train --hr-val-dirs datasets/custom/val` to the training command. Separate multiple directories with commas. Training images update the model; validation images measure progress and select the best checkpoint. Keep the two sets separate.

Check that the resource directories exist:

```bash
python analyze.py resources validate
```

This command checks directories only. It does not verify that datasets, weights, or paper resources are complete.

## 3. Entry points and common parameters

| Entry point | Purpose | Execution |
| --- | --- | --- |
| `train.py source` | Train DecompJSCC, SwinJSCC, and architecture variants | Trains by default; `--dry-run` prints the configuration only |
| `train.py fixed` | Train the two SNR-adaptive controls at a fixed CBR | Previews by default; add `--run` to train |
| `train.py adjscc / snr_eq_jscc / sa_ra_jscc` | Train external neural baselines | Trains by default; add `--dry-run` to preview |
| `eval.py source / grid / bpg / reconstruction` | Evaluate models and reconstruct images | Add `--run` to execute; previews may still read input files |
| `analyze.py <command>` | Analyze mechanisms, complexity, and results | Usually executes directly; see the subcommand's help |

`--C`, `--rate`, and `--rates` take an integer budget k. Under the current protocol, CBR = k / 1536. Pass the integer value, rather than a fraction such as `1/16`:

| k | 32 | 64 | 96 | 128 | 192 |
| --- | --- | --- | --- | --- | --- |
| CBR | 1/48 | 1/24 | 1/16 | 1/12 | 1/8 |

SNR is measured in dB. Specify multiple values as `1,4,7,10,13`. The `source` training entry point uses `--multiple-snr`; external baselines and evaluation use `--snr-db-values`.

## 4. Training

### DecompJSCC: preview and train

`train.py source` is the shared training entry point for DecompJSCC and SwinJSCC. `--recipe-stage` selects the model architecture. The example below trains one DecompJSCC model across multiple rate budgets and SNR conditions, using an AWGN channel and Kodak for validation.

Keep `--dry-run` initially to inspect the data paths, model settings, and output directories. This preview does not start training or require data or a GPU.

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

Once the data is ready, remove `--dry-run` to train. For an initial trial, use `--epochs 1 --batch-size 1` and add `--max-train-batches 2 --max-val-batches 2` to check training, validation, and checkpoint saving with a few batches.

`--epochs` controls the training length, `--lr` sets the learning rate, and `--batch-size` sets the number of images processed at once. Reduce the batch size if GPU memory is limited. For example, batch size 4 with `--gradient-accumulation-steps 2` accumulates gradients twice before updating parameters. Adjust the data paths and budget range as described above.

This is a custom training example. The paper's staged training configurations are recorded in [configs/models/](configs/models/). Apply the relevant settings through command-line arguments; `train.py` has no general `--config` argument. 

Each epoch validates the model and saves its latest state. The best checkpoint is updated when validation PSNR improves. Outputs are organized as follows:

```text
outputs/decompjscc/
├── checkpoints/last.pth          Latest model, optimizer, and training state
├── checkpoints/best.pth          Weights with the best validation PSNR
├── checkpoints/best_psnr.pth     Saved together with best.pth
├── log/config.json              Resolved configuration for this run
├── log/training_history.jsonl   Training and validation metrics for each epoch
└── runs/                        TensorBoard logs
```

Use `best.pth` for evaluation and `last.pth` to resume training. `config.json` records the model and training settings and is used by the evaluation example below. Inspect epoch metrics in the JSONL file or view training curves with TensorBoard:

```bash
tensorboard --logdir outputs/decompjscc/runs
```

### Initialization, resuming, and model selection

Using `--resume-checkpoint` alone with `source` initializes compatible model parameters, which is useful when starting a new training stage. To resume the same run, also add `--resume-training-state --resume-strict` to restore the optimizer, training progress, and random state. This currently supports single-GPU mode only; keep the original model and data settings.

For example, after completing 100 epochs, `--epochs 200` continues training to a total of 200 epochs. Resuming into a new output directory does not copy the previous best checkpoint, so retain the parent run's `best.pth`.

To train SwinJSCC, set `--recipe-stage` to `fixed`, `snr_adaptive`, or `universal` for fixed conditions, SNR adaptation, or joint SNR/rate adaptation, respectively. Set the budget and SNR range accordingly. For multiple GPUs, use `torchrun` with `--dist-mode ddp`; the `source` batch size is specified per GPU.

### Dedicated fixed-CBR training

`fixed` provides the prescribed training recipes for `decompjscc_wo_rcsb` and `swinjscc_sa`. It requires **2,663 training images and 24 Kodak images** in the default layout, plus the official initialization checkpoint at `checkpoints/pretraining/swinjscc_fixed/model_best.pth` with the SHA-256 expected by the code. Use `source` for custom datasets or training settings.

```bash
# Preview CBR=1/16; use --method swinjscc_sa to switch methods
python train.py fixed --method decompjscc_wo_rcsb --rate 96 \
  --epochs 300 --output-root outputs/fixed --run-name c96
```

This entry point prints a plan by default; add `--run` to train. Its recipe is fixed and does not accept general `--lr` or `--batch-size` arguments. Output checkpoints are named `model_last.pth` and `model_best.pth`; the preview prints their full directories.

### External neural baselines

The external neural baselines use separate networks and training entry points. ADJSCC can train from scratch, while SNR-EQ-JSCC and SA-RA-JSCC require SwinJSCC fixed initialization by default. The following example uses ADJSCC at CBR=1/16. Remove `--dry-run` once the resources are ready:

```bash
python train.py adjscc --rate 96 --epochs 300 \
  --data-dir datasets/swinjscc_hr --snr-db-values 1,4,7,10,13 \
  --learning-rate 1e-4 --global-batch-size 16 --micro-batch-size 8 \
  --output-root outputs/adjscc --dry-run
```

These entry points use `--learning-rate`, `--global-batch-size`, and `--micro-batch-size`. Their data directory arguments match those of `source`. On a single GPU, the example processes 8 images at a time and accumulates two batches before updating parameters, giving an effective batch size of 16.

When switching to `snr_eq_jscc` or `sa_ra_jscc`, specify initialization weights with `--fixed-init-checkpoint`. ADJSCC and SNR-EQ train a separate model for each `--rate`; SA-RA uses `--rates` to cover multiple budgets with one model. Outputs go under `checkpoints/`, `exp_log/`, and `runs/` within the selected output root. The preview prints the full paths.

## 5. Evaluation

### Evaluate your own DecompJSCC or SwinJSCC model

Evaluation uses the trained checkpoint and its matching `config.json` to restore the model architecture. The following command uses the training example's outputs and evaluates 3 budgets at 5 SNR values on Kodak. Each condition uses 3 noise realizations, producing average results for 15 conditions:

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


Change `--rates` and `--snr-db-values` to select evaluation conditions. Evaluate a fixed-budget model only at its corresponding rate. Add `--hr-val-dirs datasets/custom/val` for your own images, or use `--hr-eval-split clic2021` for CLIC2021. For a quick check, select one rate and one SNR and add `--max-val-batches 1`.

Explicit checkpoint, configuration, and data paths avoid dependence on the old resource index. `--strict-checkpoint` checks that the weights match the model architecture; use it with the configuration saved during training.

The output JSON contains `results` for condition averages, `realization_results` for each noise realization, and `per_image_results` for individual images. This entry point computes **PSNR, SSIM, and CBR**. The `ssim` field is not MS-SSIM.

### Other evaluation features

`eval.py grid` compares methods under a common protocol and also computes RGB five-scale MS-SSIM. `eval.py reconstruction` exports reconstructed images for selected conditions. These features still depend on a resource index or paper selection files. The current minimal repository does not include the corresponding `configs/paper_resources.json` or paper data bundle. For your own DecompJSCC or SwinJSCC model, use the `source` example above.

BPG+LDPC uses `eval.py bpg`. In addition to Sionna, it needs working `bpgenc` and `bpgdec` executables and an AMC calibration file. The default paper calibration file is not included in the current repository, so specify an existing file with `--mcs-calibration`. See `python eval.py bpg --help` for the available options.

## 6. Analysis and result summaries

### Parameter counts and inference latency

After training, analyze both model cost and reconstruction quality. The following command measures parameter count, model state size, and inference latency for the trained DecompJSCC model. It uses synthetic 256×256 inputs, so no image dataset is needed:

```bash
python analyze.py profile-role --role decompjscc \
  --checkpoint outputs/decompjscc/checkpoints/best.pth \
  --model-size base --rates 32,96,192 --image-size 256 \
  --device cuda:0 --warmup 20 --iterations 80 \
  --output-json outputs/analysis/profile.json \
  --output-markdown outputs/analysis/profile.md
```

Results are written as JSON for further processing and Markdown for reading. To measure only parameter count and state size, add `--skip-latency --device cpu`. Keep the device, input size, and batch size consistent when comparing inference latency.

Use `analyze.py routing` to compare routing behavior with normal SNR conditioning and with SNR conditioning disabled. This analysis requires trained weights and validation images. See `python analyze.py routing --help` to configure the conditions.

### Export evaluation JSON to CSV

`analyze.py summary` reads existing evaluation results and organizes condition averages and per-image metrics without rerunning the model. For the complete Kodak `source` evaluation above, export CSV files with:

```bash
python analyze.py summary \
  --input decompjscc:kodak:outputs/eval/decompjscc_kodak.json \
  --expected-rates 32,96,192 --expected-snr-values 1,4,7,10,13 \
  --expected-realizations 3 --expected-images kodak=24 \
  --output-dir outputs/analysis/summary
```

Outputs include `summary.json`, `cell_summary.csv`, `realization_results.csv`, and `per_image_results.csv`. If you used custom images, different conditions, or only a few batches, adjust the expected image count and conditions accordingly. Repeat `--input METHOD:SPLIT:PATH` to add other methods. Paired comparisons require matching image sets, conditions, and noise seeds.

## 7. Code layout and checks

```text
train.py / eval.py / analyze.py   Training, evaluation, and analysis entry points
models/                          DecompJSCC model and components
SwinJSCC_base/                   Shared Swin backbone, channels, and losses
experiments/                     Training, evaluation, mechanism analysis, and statistics
external_baselines/              Baseline methods and BPG source code
configs/models/                  Model and staged training configurations
tests/                           Tests
```

Start with the three entry points at the repository root for everyday use. Look in `models/` for model changes and `experiments/` for training or evaluation changes. Each subcommand provides its own help:

```bash
python train.py source --help
python train.py fixed --help
python train.py adjscc --help
python eval.py source --help
python eval.py grid --help
python analyze.py routing --help
python analyze.py profile-role --help
```

Start with the training, analysis, and statistics tests that do not require the resource bundle:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -B -m unittest \
  tests.test_training tests.test_analysis tests.test_statistics
```

