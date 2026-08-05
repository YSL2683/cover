#!/bin/bash
export CACHE_DIR=resfit/my_lerobot_data
export PYTHONPATH=/home/moai/ysl_ws/cover

python resfit/lerobot/scripts/train_bc_dexmg.py \
    --dataset ysl2683/lane_nut_assembly_square_id_20 \
    --policy diffusion \
    --steps 30000 \
    --rollout_freq 5000 \
    --eval_env SquareID \
    --eval_camera_size 128 \
    --eval_num_episodes 20 \
    --wandb_enable --wandb_project train_diffusion_square
