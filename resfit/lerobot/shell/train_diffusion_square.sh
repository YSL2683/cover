#!/bin/bash
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export CACHE_DIR=resfit/my_lerobot_data
export PYTHONPATH=${PROJECT_ROOT}

python resfit/lerobot/scripts/train_bc_dexmg.py \
    --dataset ysl2683/robomimic_square_v15_50 \
    --policy diffusion \
    --policy_kwargs '{"crop_shape": [112, 112]}' \
    --steps 100000 \
    --batch_size 256 \
    --rollout_freq 2000 \
    --save_freq 2000 \
    --eval_env Square \
    --eval_camera_size 128 \
    --eval_num_episodes 20 \
    --eval_num_envs 10 \
    --num_workers 8 \
    --wandb_enable --wandb_project train_diffusion_square
