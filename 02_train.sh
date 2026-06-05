#!/bin/bash
# =============================================================================
# nnUNet v1 -- Mouse Brain Segmentation Training Pipeline
# Configuration restored from debug.json / training_log
#
# Trainer:  nnUNetTrainerV2
# Plans:    nnUNetPlansv2.1
# Key training params (auto-planned by nnUNet):
#   3D fullres: patch [80,192,128], batch 2, epochs 1000, lr 0.01
#   2D:         patch [224,160],   batch 78, epochs 1000, lr 0.01
#
# Usage:
#   bash 02_train.sh [gpu_id]   # default GPU=0
# =============================================================================

set -euo pipefail
GPU=${1:-0}

# ── Set to your local data path ───────────────────────────────────────────────
export nnUNet_raw_data_base="/path/to/data/nnUNet_raw_data_base"
export nnUNet_preprocessed="/path/to/data/nnUNet_preprocessed"
export RESULTS_FOLDER="/path/to/data/RESULTS_FOLDER"
# ─────────────────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES=$GPU

TRAINER="nnUNetTrainerV2"
PLANS="nnUNetPlansv2.1"
FOLD=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# =============================================================================
# STEP 1 -- Dataset verification & preprocessing (all Tasks)
# =============================================================================
log "=== STEP 1: Preprocessing ==="

for TASK in \
    Task020_Mouse5mod \
    Task021_Mouse_T2 \
    Task022_Mouse_T2s \
    Task023_Mouse_T1 \
    Task024_Mouse_iMag \
    Task025_Mouse_QSM
do
    log "Verifying $TASK ..."
    nnUNet_plan_and_preprocess -t ${TASK##Task} \
        --verify_dataset_integrity \
        -pl3d ExperimentPlanner3D_v21 \
        -pl2d ExperimentPlanner2D_v21 \
        -tf 8 -tl 8          # preprocessing threads
done

# Task026 shares plans with Task020; re-run plan_and_preprocess only if it has new data:
# nnUNet_plan_and_preprocess -t 26 --verify_dataset_integrity

# =============================================================================
# STEP 2 -- Train single-modality models, 3D fullres (Task021-025)
# =============================================================================
log "=== STEP 2: Single-modality training (3D fullres) ==="

for TASK_ID in 21 22 23 24 25; do
    log "Training Task0${TASK_ID} -- 3d_fullres fold $FOLD ..."
    nnUNet_train 3d_fullres $TRAINER $TASK_ID $FOLD \
        -p $PLANS \
        --npz            # also save softmax .npz for ensemble
done

# =============================================================================
# STEP 3 -- Train 5-modality model, 3D fullres + 2D (Task020)
# =============================================================================
log "=== STEP 3: 5-modality training (3D fullres + 2D) ==="

log "Training Task020_Mouse5mod -- 3d_fullres fold $FOLD ..."
nnUNet_train 3d_fullres $TRAINER 20 $FOLD \
    -p $PLANS \
    --npz

log "Training Task020_Mouse5mod -- 2d fold $FOLD ..."
nnUNet_train 2d $TRAINER 20 $FOLD \
    -p $PLANS \
    --npz

# =============================================================================
# STEP 4 -- Continual learning: Task026 (fine-tuned from Task020 3D checkpoint)
# Requires the custom trainer; see nnunet/training/network_training/nnUNetTrainerV2_ContinueLearn.py
# =============================================================================
log "=== STEP 4: Continual learning Task026 ==="

# 3D fullres
log "Training Task026_ContinueLearn_Fold0 -- 3d_fullres ..."
nnUNet_train 3d_fullres nnUNetTrainerV2_ContinueLearn 26 $FOLD \
    -p $PLANS \
    --npz

# 2D
log "Training Task026_ContinueLearn_Fold0 -- 2d ..."
nnUNet_train 2d nnUNetTrainerV2_ContinueLearn 26 $FOLD \
    -p $PLANS \
    --npz

log "=== All training complete ==="
