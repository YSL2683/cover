#!/bin/bash
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

# Ensure conda environment 'cover' is activated
if [ -f "/home/ysl2683/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/home/ysl2683/anaconda3/etc/profile.d/conda.sh"
    conda activate cover
elif [ -f "/home/moai/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/moai/miniconda3/etc/profile.d/conda.sh"
    conda activate cover
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/lane:$PYTHONPATH

DEMO_DIR="${PROJECT_ROOT}/lane/demo/robomimic_square/50"
SAVE_DIR="${PROJECT_ROOT}/lane/pretrained_e2c/square_z8"
Z_DIM=8
MODE="decoupled"

echo "=================================================="
echo "Pretraining Decoupled E2C with z_dim=${Z_DIM} (50 demos)"
echo "Demo Dir : ${DEMO_DIR}"
echo "Save Dir : ${SAVE_DIR}"
echo "Mode     : ${MODE}"
echo "Z Dim    : ${Z_DIM}"
echo "=================================================="

mkdir -p "${SAVE_DIR}"

python lane/pretrain_e2c.py \
    --demo_dir "${DEMO_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --mode "${MODE}" \
    --z_dim "${Z_DIM}"

echo "=================================================="
echo "Pretraining completed! Models saved to ${SAVE_DIR}"
echo "=================================================="
