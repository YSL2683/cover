#!/bin/bash
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export CACHE_DIR=resfit/my_lerobot_data
export PYTHONPATH=${PROJECT_ROOT}

# Ensure conda environment 'cover' is activated
if [ -f "/home/moai/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/moai/miniconda3/etc/profile.d/conda.sh"
    conda activate cover
fi

python resfit/lerobot/scripts/train_bc_dexmg.py \
    --dataset ysl2683/robomimic_can_v15_10 \
    --policy diffusion \
    --policy_kwargs '{"crop_shape": [112, 112]}' \
    --steps 200000 \
    --batch_size 256 \
    --rollout_freq 1000 \
    --save_freq 1000 \
    --eval_env Can \
    --eval_camera_size 128 \
    --eval_num_episodes 100 \
    --eval_num_envs 16 \
    --num_workers 8 \
    --seed 42 \
    --wandb_enable --wandb_project train_diffusion_can
