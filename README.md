# nnUNet for Mouse Brain Segmentation

<p align="center">
  <img src="zhang_lab_logo.png" alt="Zhang Lab logo" width="160">
</p>

<p align="center">
  <strong>Zhang Lab · Shanghai Jiao Tong University School of Medical Technology</strong>
</p>

This repository contains the training code for **multi-modal mouse brain atlas segmentation** using [nnUNet v1](https://github.com/MIC-DKFZ/nnUNet).

---

## Overview

We trained nnUNet models to segment **50 brain regions plus background** from mouse MRI across multiple modalities and configurations:

| Task | Modality | Network | Notes |
|------|----------|---------|-------|
| Task020_Mouse5mod | QSMs + T1 + T2 + T2s + iMag (5-mod) | 3D fullres + 2D | Multi-modal |
| Task021_Mouse_T2 | T2 | 3D fullres | Single-modal |
| Task022_Mouse_T2s | T2s | 3D fullres | Single-modal |
| Task023_Mouse_T1 | T1 | 3D fullres | Single-modal |
| Task024_Mouse_iMag | iMag | 3D fullres | Single-modal |
| Task025_Mouse_QSM | QSMs | 3D fullres | Single-modal |
| Task026_ContinueLearn_Fold0 | QSMs + T1 + T2 + T2s + iMag | 3D fullres + 2D | Continual learning from Task020 |

**Modality channel order** (for multi-modal tasks): `0=QSMs, 1=T1, 2=T2, 3=T2s, 4=iMag`

---

## Repository Structure

```
nnunet_training_code/
├── nnunet/                         # Modified nnUNet v1 source code
│   ├── network_architecture/       # Generic U-Net
│   ├── training/
│   │   ├── network_training/
│   │   │   ├── nnUNetTrainerV2.py              # Standard trainer (used for all tasks)
│   │   │   └── nnUNetTrainerV2_ContinueLearn.py  # Custom continual learning trainer (Task026)
│   │   ├── data_augmentation/      # moreDA augmentation pipeline
│   │   ├── loss_functions/         # Dice + Cross-Entropy loss
│   │   └── dataloading/
│   ├── preprocessing/              # Cropping, resampling, z-score normalization
│   ├── inference/                  # Sliding-window prediction + ensemble
│   ├── postprocessing/             # Connected-component post-processing
│   └── evaluation/                 # Dice, HD95 metrics
│
├── 01_prepare_dataset.py           # Generate dataset.json for all Tasks
├── 02_train.sh                     # Full training pipeline (preprocess → train)
├── 04_inference.sh                 # Inference + 3D/2D ensemble
└── README.md
```

---

## Modifications to nnUNet

This codebase is based on **nnUNet v1** (Apache 2.0, © DKFZ Heidelberg). The only addition is:

- `nnunet/training/network_training/nnUNetTrainerV2_ContinueLearn.py` — a custom Trainer subclass that loads a pretrained checkpoint from Task020 and fine-tunes on new data, with an optional EWC (Elastic Weight Consolidation) regularization term to reduce catastrophic forgetting.

All other code is unmodified from the original nnUNet v1.

---

## Installation

```bash
# 1. Install nnUNet v1
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .

# 2. Replace the nnunet/ source folder with this repo's version
#    (or install this repo directly)
cd ..
git clone <this-repo>
cd nnunet_training_code
pip install -e .

# 3. Set environment variables
export nnUNet_raw_data_base="/path/to/nnUNet_raw_data_base"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export RESULTS_FOLDER="/path/to/RESULTS_FOLDER"
```

---

## Usage

### Step 1 — Prepare datasets

Organize your NIfTI files into nnUNet raw data format, then run:

```bash
python 01_prepare_dataset.py
```

This generates `dataset.json` for all 7 Tasks. Edit `BASE_DIR` at the top of the script to match your paths.

File naming convention:
- Training images: `imagesTr/<case_id>_0000.nii.gz` (single-modal) or `_0000` … `_0004` (5-modal)
- Training labels: `labelsTr/<case_id>.nii.gz`

### Step 2 — Preprocess and train

```bash
bash 02_train.sh [gpu_id]   # default GPU 0
```

This runs `nnUNet_plan_and_preprocess` followed by `nnUNet_train` for all Tasks in order.

### Step 3 — Inference

```bash
bash 04_inference.sh [gpu_id]
```

For Task020 and Task026, predictions from 3D fullres and 2D models are automatically ensembled.

---

## Training Configuration

Derived from saved `debug.json` files:

| Parameter | Value |
|-----------|-------|
| Trainer | `nnUNetTrainerV2` |
| Plans | `nnUNetPlansv2.1` |
| Network | 3D U-Net (6 pooling levels) |
| Patch size (3D) | 80 × 192 × 128 |
| Patch size (2D) | 224 × 160 |
| Batch size (3D / 2D) | 2 / 78 |
| Optimizer | SGD, Nesterov momentum=0.99 |
| Initial LR | 0.01 (poly decay) |
| Weight decay | 3e-5 |
| Max epochs | 1000 |
| Loss | Dice + Cross-Entropy (deep supervision) |
| Mixed precision | fp16 |
| Normalization | z-score within foreground mask |
| Dataset split | 37 train / 10 val (fold 0) |

---

## License

The nnUNet source code is licensed under the **Apache License 2.0** (© German Cancer Research Center, DKFZ).  
Custom additions in this repository are released under the same license.
