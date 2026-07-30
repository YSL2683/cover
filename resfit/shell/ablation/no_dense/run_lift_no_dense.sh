#!/bin/bash
# Script to run Residual TD3 WITHOUT dense reward (ablation study)

# Default parameters (can be overridden by command line arguments)
REWARD_TYPE="none"
SEED=42
BASE_POLICY_PATH="resfit/my_lerobot_data/bc_run_2026-07-30_16-20-35_lane_lift_id_20_aligned_diffusion/policy_step_5000/policy"
E2C_DIR=""  # We don't need E2C for sparse reward

# Name for Weights & Biases
WANDB_NAME="residual_td3_sparse_only"

# Parse command line arguments if provided
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
echo "Starting Residual TD3 Training (NO DENSE REWARD)"
echo "Reward Type     : $REWARD_TYPE"
echo "Seed            : $SEED"
echo "Base Policy Path: $BASE_POLICY_PATH"
echo "WandB Name      : $WANDB_NAME"
echo "=================================================="

# Clear scratch memory buffers
rm -rf /home/moai/ysl_ws/cover/scratch/online/* /home/moai/ysl_ws/cover/scratch/offline/*

# Environment variables
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/moai/ysl_ws/cover:$PYTHONPATH
export CACHE_DIR=/home/moai/ysl_ws/cover/scratch
export HF_HUB_OFFLINE=1
export LEROBOT_OFFLINE=1

# Run training
python resfit/rl_finetuning/scripts/train_residual_td3.py \
    task="Lift" \
    rl_camera="['observation.images.frontview','observation.images.robot0_eye_in_hand']" \
    wandb.project="lift_residual_rl" \
    wandb.name="${WANDB_NAME}" \
    seed="${SEED}" \
    algo.reward_type="${REWARD_TYPE}" \
    base_policy_path="${BASE_POLICY_PATH}" \
    e2c_dir="${E2C_DIR}" \
    eval_interval_every_steps=2000
