#!/bin/bash
# =============================================================================
# nnUNet v1 -- Mouse Brain Segmentation Inference Script
# Covers single-modality, 5-modality, and continual-learning Tasks
#
# Usage:
#   bash 04_inference.sh [gpu_id]   # default GPU=0
# =============================================================================

set -euo pipefail
GPU=${1:-0}

export nnUNet_raw_data_base="/path/to/data/nnUNet_raw_data_base"
export nnUNet_preprocessed="/path/to/data/nnUNet_preprocessed"
export RESULTS_FOLDER="/path/to/data/RESULTS_FOLDER"
export CUDA_VISIBLE_DEVICES=$GPU

TRAINER="nnUNetTrainerV2"
PLANS="nnUNetPlansv2.1"
FOLD=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# =============================================================================
# Single-modality inference (Task021-025, 3D fullres)
# =============================================================================
for TASK_ID in 21 22 23 24 25; do
    TASK_DIR="${nnUNet_raw_data_base}/nnUNet_raw_data/Task0${TASK_ID}_Mouse_*"
    INPUT_DIR=$(ls -d ${TASK_DIR} 2>/dev/null | head -1)/imagesTs
    OUTPUT_DIR="${RESULTS_FOLDER}/predictions/Task0${TASK_ID}_3d"

    if [ ! -d "$INPUT_DIR" ] || [ -z "$(ls -A $INPUT_DIR 2>/dev/null)" ]; then
        log "Skipping Task0${TASK_ID}: imagesTs is empty"
        continue
    fi

    log "Predicting Task0${TASK_ID} -- 3d_fullres ..."
    nnUNet_predict \
        -i "$INPUT_DIR" \
        -o "$OUTPUT_DIR" \
        -t $TASK_ID \
        -m 3d_fullres \
        -tr $TRAINER \
        -p $PLANS \
        -f $FOLD \
        --save_npz
done

# =============================================================================
# 5-modality inference (Task020, 3D fullres + 2D ensemble)
# =============================================================================
TASK020_INPUT="${nnUNet_raw_data_base}/nnUNet_raw_data/Task020_Mouse5mod/imagesTs"

if [ -d "$TASK020_INPUT" ] && [ -n "$(ls -A $TASK020_INPUT 2>/dev/null)" ]; then
    log "Predicting Task020 -- 3d_fullres ..."
    nnUNet_predict \
        -i "$TASK020_INPUT" \
        -o "${RESULTS_FOLDER}/predictions/Task020_3d" \
        -t 20 -m 3d_fullres -tr $TRAINER -p $PLANS -f $FOLD --save_npz

    log "Predicting Task020 -- 2d ..."
    nnUNet_predict \
        -i "$TASK020_INPUT" \
        -o "${RESULTS_FOLDER}/predictions/Task020_2d" \
        -t 20 -m 2d -tr $TRAINER -p $PLANS -f $FOLD --save_npz

    # Ensemble 3D + 2D
    log "Ensembling Task020 (3d + 2d) ..."
    nnUNet_ensemble \
        -f "${RESULTS_FOLDER}/predictions/Task020_3d" \
           "${RESULTS_FOLDER}/predictions/Task020_2d" \
        -o "${RESULTS_FOLDER}/predictions/Task020_ensemble" \
        -pp "${RESULTS_FOLDER}/nnUNet/3d_fullres/Task020_Mouse5mod/${TRAINER}__${PLANS}/fold_${FOLD}/postprocessing.json"
else
    log "Skipping Task020: imagesTs is empty"
fi

# =============================================================================
# Continual learning inference (Task026)
# =============================================================================
TASK026_INPUT="${nnUNet_raw_data_base}/nnUNet_raw_data/Task026_ContinueLearn_Fold0/imagesTs"
CL_TRAINER="nnUNetTrainerV2_ContinueLearn"

if [ -d "$TASK026_INPUT" ] && [ -n "$(ls -A $TASK026_INPUT 2>/dev/null)" ]; then
    log "Predicting Task026 -- 3d_fullres (ContinueLearn) ..."
    nnUNet_predict \
        -i "$TASK026_INPUT" \
        -o "${RESULTS_FOLDER}/predictions/Task026_3d" \
        -t 26 -m 3d_fullres -tr $CL_TRAINER -p $PLANS -f $FOLD --save_npz

    log "Predicting Task026 -- 2d (ContinueLearn) ..."
    nnUNet_predict \
        -i "$TASK026_INPUT" \
        -o "${RESULTS_FOLDER}/predictions/Task026_2d" \
        -t 26 -m 2d -tr $CL_TRAINER -p $PLANS -f $FOLD --save_npz

    log "Ensembling Task026 (3d + 2d) ..."
    nnUNet_ensemble \
        -f "${RESULTS_FOLDER}/predictions/Task026_3d" \
           "${RESULTS_FOLDER}/predictions/Task026_2d" \
        -o "${RESULTS_FOLDER}/predictions/Task026_ensemble"
else
    log "Skipping Task026: imagesTs is empty"
fi

log "=== Inference complete ==="
