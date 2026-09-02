#!/bin/bash
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export CACHE_DIR=resfit/my_lerobot_data
export PYTHONPATH=${PROJECT_ROOT}

python resfit/lerobot/scripts/train_bc_dexmg.py \
    --dataset ysl2683/robomimic_square_v15_50 \
    --policy diffusion \
    --policy_kwargs '{"crop_shape": [112, 112]}' \
    --steps 200000 \
    --batch_size 256 \
    --rollout_freq 1000 \
    --save_freq 1000 \
    --resume_ckpt ${PROJECT_ROOT}/resfit/my_lerobot_data/bc_run_2026-08-31_09-08-37_robomimic_square_v15_50_diffusion/latest \
    --eval_env Square \
    --eval_camera_size 128 \
    --eval_num_episodes 100 \
    --eval_num_envs 16 \
    --num_workers 8 \
    --seed 42 \
    --wandb_enable --wandb_project train_diffusion_square
