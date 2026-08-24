#!/bin/bash
export CACHE_DIR=resfit/my_lerobot_data
export PYTHONPATH=/home/moai/ysl_ws/cover

python resfit/lerobot/scripts/train_bc_dexmg.py \
    --dataset ysl2683/robomimic_square_v15_50 \
    --policy diffusion \
    --steps 100000 \
    --batch_size 256 \
    --rollout_freq 2000 \
    --save_freq 2000 \
    --resume_ckpt /home/moai/ysl_ws/cover/resfit/my_lerobot_data/bc_run_2026-08-23_18-24-26_robomimic_square_v15_50_diffusion/latest \
    --eval_env Square \
    --eval_camera_size 128 \
    --eval_num_episodes 20 \
    --eval_num_envs 10 \
    --num_workers 8 \
    --wandb_enable --wandb_project train_diffusion_square
