#!/bin/bash
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
# Script to run Residual TD3 WITHOUT dense reward (ablation study)
# Adapted for a 5x5cm spawn area and Black Table environment

# Default parameters
REWARD_TYPE="none"
SEED=42
BASE_POLICY_PATH="resfit/my_lerobot_data/bc_run_2026-07-30_16-20-35_lane_lift_id_20_aligned_diffusion/policy_step_5000/policy"
E2C_DIR=""  # We don't need E2C for sparse reward

# Name for Weights & Biases
WANDB_NAME="sparse_black_table"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --reward_type) REWARD_TYPE="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --wandb_name) WANDB_NAME="$2"; shift ;;
        --base_policy_path) BASE_POLICY_PATH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "=================================================="
echo "Starting Residual TD3 Training (NO DENSE REWARD, Black Table)"
echo "Reward Type     : $REWARD_TYPE"
echo "Seed            : $SEED"
echo "Base Policy Path: $BASE_POLICY_PATH"
echo "WandB Name      : $WANDB_NAME"
echo "=================================================="

# Clear scratch memory buffers
rm -rf ${PROJECT_ROOT}/scratch/online_buffer_cache/* ${PROJECT_ROOT}/scratch/offline_buffer_cache/*
rm -rf ${PROJECT_ROOT}/scratch/online/* ${PROJECT_ROOT}/scratch/offline/*

# Environment variables
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export CACHE_DIR=${PROJECT_ROOT}/scratch
export HF_HUB_OFFLINE=1
export LEROBOT_OFFLINE=1

# Run training
python resfit/rl_finetuning/scripts/train_residual_td3.py \
    env_modifier.mode=visual_ood \
    env_modifier.disturbance=null \
    env_modifier.ood_position.x_bounds="[-0.025, 0.025]" \
    env_modifier.ood_position.y_bounds="[-0.025, 0.025]" \
    env_modifier.visual_ood.table_color="black" \
    task="Lift" \
    rl_camera="['observation.images.frontview','observation.images.robot0_eye_in_hand']" \
    wandb.project="lift_residual_rl" \
    wandb.name="${WANDB_NAME}" \
    seed="${SEED}" \
    algo.reward_type="${REWARD_TYPE}" \
    base_policy_path="${BASE_POLICY_PATH}" \
    e2c_dir="${E2C_DIR}" \
    eval_interval_every_steps=2000
