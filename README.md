# UAVidToolKit

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/github/license/dronefreak/UAVidToolKit)](LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/dronefreak/UAVidToolKit)](https://github.com/dronefreak/UAVidToolKit/issues)
[![GitHub Stars](https://img.shields.io/github/stars/dronefreak/UAVidToolKit?style=social)](https://github.com/dronefreak/UAVidToolKit)

A collection of utilities for working with the [UAVid](https://uavid.nl/) semantic segmentation dataset. The toolkit covers the full workflow from label preprocessing through to quantitative evaluation and visualisation.

**Capabilities:**
- Convert RGB colour label images to single-channel train-ID images (and back)
- Blend camera images with colour-coded label overlays for visual inspection
- Compute per-class IoU, mean IoU, and pixel accuracy against ground truth
- Generate image–label path-pair text files for training and inference pipelines

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Dataset Layout](#dataset-layout)
4. [Label Colour Map](#label-colour-map)
5. [Usage](#usage)
   - [RGB Labels → Train-ID Labels](#1-rgb-labels--train-id-labels)
   - [Train-ID Labels → RGB Labels](#2-train-id-labels--rgb-labels)
   - [Blend Images with Labels](#3-blend-images-with-labels)
   - [Performance Evaluation](#4-performance-evaluation)
   - [Generate Path-Pair Files](#5-generate-path-pair-files)
6. [Contributing](#contributing)

---

## Requirements

| Package | Purpose |
|---|---|
| `numpy` | Array operations |
| `Pillow` | Image I/O |
| `opencv-python` | Image blending |
| `tqdm` | Progress bars |
| `scikit-learn` | Fallback confusion matrix (pure Python path) |
| `matplotlib` | Evaluation visualisation |
| `Cython` + `setuptools` | C++ evaluation extension |

Install all dependencies at once:

```bash
pip install -r requirements.txt
```

---

## Installation

Clone the toolkit into your UAVid dataset folder and compile the optional C++ extension for fast evaluation:

```bash
cd <UAVid dataset folder>
git clone https://github.com/dronefreak/UAVidToolKit.git
cd UAVidToolKit
pip install -r requirements.txt
python setup.py build_ext --inplace
cd ..
```

> **Note:** The Cython extension (`addToConfusionMatrix`) is optional but strongly recommended for large datasets. If it is not compiled, evaluation falls back to a pure-Python implementation that is significantly slower.

---

## Dataset Layout

The toolkit expects the following directory structure. Use symlinks if your original folders have different names:

```
UAVidDataset/
├── train/
│   ├── seq1/
│   │   ├── Images/
│   │   └── Labels/
│   └── seq2/ ...
├── valid/
│   └── seq<N>/ ...
├── test/
│   └── seq<N>/ ...
└── UAVidToolKit/
```

To create symlinks for non-standard folder names:

```bash
ln -s <original train dir> train
ln -s <original valid dir> valid
ln -s <original test  dir> test
```

---

## Label Colour Map

The eight UAVid semantic classes and their corresponding RGB colours:

| Train ID | Class | RGB |
|:---:|---|:---:|
| 0 | Clutter | `(0, 0, 0)` |
| 1 | Building | `(128, 0, 0)` |
| 2 | Road | `(128, 64, 128)` |
| 3 | Static Car | `(192, 0, 192)` |
| 4 | Tree | `(0, 128, 0)` |
| 5 | Vegetation | `(128, 128, 0)` |
| 6 | Human | `(64, 64, 0)` |
| 7 | Moving Car | `(64, 0, 128)` |

---

## Usage

All commands are run from the **UAVid dataset root** (the parent of `UAVidToolKit/`).

### 1. RGB Labels → Train-ID Labels

Converts 3-channel RGB label images to single-channel uint8 train-ID images.

```bash
python UAVidToolKit/prepareTrainIdFiles.py -s <source_dir> -t <target_dir>
```

| Argument | Description |
|---|---|
| `-s` | Source directory containing `seq*/Labels/` (RGB labels) |
| `-t` | Target directory where `seq*/TrainId/` folders are written |

**Example:**
```bash
python UAVidToolKit/prepareTrainIdFiles.py -s valid/ -t tooltest/
```

---

### 2. Train-ID Labels → RGB Labels

Converts single-channel train-ID images back to 3-channel RGB colour images.

```bash
python UAVidToolKit/convertTrainIdFiles2Color.py -s <source_dir> -t <target_dir> [-f <subdir>]
```

| Argument | Description | Default |
|---|---|---|
| `-s` | Source directory containing `seq*/TrainId/` | — |
| `-t` | Target directory for colour output | — |
| `-f` | Output subdirectory name inside each sequence folder | `color` |

**Example:**
```bash
python UAVidToolKit/convertTrainIdFiles2Color.py -s tooltest/ -t tooltest/ -f color
```

---

### 3. Blend Images with Labels

Alpha-composites camera images with their colour-coded label overlays.
The blend formula is: `output = image × alpha + label × beta + gamma`.

```bash
python UAVidToolKit/blendImageAndLabel.py \
    -i <image_dir> -l <label_dir> -o <output_dir> \
    [-id <image_subdir>] [-ld <label_subdir>] [-od <output_subdir>] \
    [-alpha <float>] [-beta <float>] [-gamma <float>]
```

| Argument | Description | Default |
|---|---|---|
| `-i` | Root directory of camera images | — |
| `-l` | Root directory of label images | — |
| `-o` | Root directory for blended outputs | — |
| `-id` | Image subdirectory name | `Images` |
| `-ld` | Label subdirectory name | `Labels` |
| `-od` | Output subdirectory name | `Blend` |
| `-alpha` | Weight applied to the camera image | `0.6` |
| `-beta` | Weight applied to the label image | `0.4` |
| `-gamma` | Scalar offset added to every pixel | `0.0` |

**Example:**
```bash
python UAVidToolKit/blendImageAndLabel.py \
    -i valid/ -l tooltest/ -o tooltest/ \
    -id Images -ld color -od blend
```

---

### 4. Performance Evaluation

Computes per-class IoU, mean IoU, pixel accuracy, and per-class GT pixel counts by comparing predictions against ground truth.

```bash
python UAVidToolKit/evaluate.py -gt <gt_dir> -p <pred_dir> [-v] [-o <output_dir>]
```

| Argument | Description | Default |
|---|---|---|
| `-gt` | Ground-truth directory | — |
| `-p` | Prediction directory (same structure as `-gt`) | — |
| `-v` | Save IoU bar-chart and confusion matrix as PNG files | off |
| `-o` | Output directory for saved figures | `./` |

**Example:**
```bash
python UAVidToolKit/evaluate.py -gt valid/ -p pred_valid/ -v -o results/
```

Sample output:
```
IOUs: {'Clutter': 0.812, 'Building': 0.743, ...}
mIOU: 0.756
acc:  0.891
GT pixel counts per class: {'Clutter': 1024000, ...}
```

---

### 5. Generate Path-Pair Files

Writes absolute image–label path pairs to a text file, one pair per line separated by a space. Useful as input for training and inference scripts.

**Training / validation pairs** — outputs `./img_lbl_pair.txt`:

```bash
python UAVidToolKit/writeImageLabelPathPairsToTxtFile.py -d <dataset_dir> [-t] [-v]
```

| Argument | Description |
|---|---|
| `-d` | Dataset root directory |
| `-t` | Include the training split |
| `-v` | Include the validation split |

**Test / prediction pairs** — outputs `./test_pred_pair.txt` and/or `./valid_pred_pair.txt`:

```bash
python UAVidToolKit/writeImageLabelPathPairsToTxtFile.py \
    -d <dataset_dir> --test --pred-dir <pred_dir>
```

| Argument | Description |
|---|---|
| `--test` | Include the test split |
| `--pred-dir` | Root directory where predictions are (or will be) stored |

**Examples:**
```bash
# Training + validation ground-truth pairs
python UAVidToolKit/writeImageLabelPathPairsToTxtFile.py -d ./ -t -v

# Test image → expected prediction path pairs
python UAVidToolKit/writeImageLabelPathPairsToTxtFile.py \
    -d ./ --test --pred-dir pred_test/
```

---

## Contributing

Bug reports and feature suggestions are welcome. Please open an issue on the [GitHub issue tracker](https://github.com/dronefreak/UAVidToolKit/issues) with a clear description and, where applicable, a minimal reproducible example.

