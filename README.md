# AI-Based Restoration of Degraded Images for Semiconductor Inspection

This repository contains the final **FastNAF-SR** solution for restoring degraded grayscale semiconductor inspection images with fixed **2× restoration / super-resolution**.

## Final validation results

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| Bicubic | 22.9572 dB | 0.535671 | 0.435001 |
| **FastNAF-SR (epoch 29)** | **25.3076 dB** | **0.633729** | **0.375866** |

Improvement over bicubic:

- PSNR: **+2.3504 dB**
- SSIM: **+0.098058**
- LPIPS: **−0.059135**

## Model configuration

```text
Input channels : 1
Width          : 64
NAF blocks     : 12
Scale factor   : 2
Upsampling     : PixelShuffle
Input          : grayscale H × W
Output         : grayscale 2H × 2W
```

The learned branch sees the original degraded values, including values outside `[0,1]`. A bicubic 2× branch is added residually. Final outputs are clipped to `[0,1]` and saved as `float32` NumPy arrays.

## Development-machine inference throughput

| Batch | ms/batch | ms/image | images/s |
|---:|---:|---:|---:|
| **32** | **446.402** | **13.9501** | **71.68** |
| 64 | 897.508 | 14.0236 | 71.31 |
| 128 | 1815.823 | 14.1861 | 70.40 |

Batch size **32** is therefore the default in `evaluation.py`.

These are measurements from the Colab GPU used during development, **not H100 benchmark results**.

## Repository structure

```text
.
├── run.py
├── model.py
├── best_model.pt
├── requirements.txt
├── README.md
├── verify_submission.py
└── restored_test_outputs/
```

## Installation

```bash
pip install -r requirements.txt
```

## Run evaluation

Positional form:

```bash
python run.py /path/to/test_inputs /path/to/output_dir
```

Named form:

```bash
python run.py \
    --input_dir /path/to/test_inputs \
    --output_dir /path/to/output_dir
```

The evaluator expects `best_model.pt` beside `evaluation.py` unless `--model_path` is supplied.

Optional arguments:

```bash
--batch_size 32
--precision fp16
--model_path /path/to/best_model.pt
```

On CPU, evaluation automatically uses FP32.

## Input / output format

The final challenge data used for development had:

```text
input shape  : 128 × 128
input dtype  : float32
output shape : 256 × 256
output dtype : float32
```

The evaluator preserves input `.npy` filenames and saves restored outputs as `float32` arrays clipped to `[0,1]`.

The final development test processed:

```text
400 / 400 files
output shape : 256 × 256
dtype        : float32
value range  : [0,1]
sanity check : PASSED
```

## Final checkpoint

The selected checkpoint is **epoch 29**:

```text
PSNR  : 25.3076 dB
SSIM  : 0.633729
LPIPS : 0.375866
```

The epoch-29 checkpoint completed successfully and was selected before FP16 overflow protection stopped the subsequent epoch.
