#!/bin/bash
# Script to run Residual TD3 with configurable reward shaping parameters and OOD range of 0.1

# Default parameters (can be overridden by command line arguments)
REWARD_TYPE="reward_2"
BETA=1.0
ALPHA=0.98
W_M=0.3
W_W=0.7
SEED=42
FREEZE_E2C="True"
BASE_POLICY_PATH="resfit/my_lerobot_data/bc_run_2026-07-30_16-20-35_lane_lift_id_20_aligned_diffusion/policy_step_5000/policy"
E2C_DIR="/home/moai/ysl_ws/cover/lane/pretrained_e2c/lift"

# Name for Weights & Biases
WANDB_NAME="reward2_beta${BETA}_freeze_ood_0.1"

# Parse command line arguments if provided
# Example: ./run_lift_ood_0.1.sh --beta 0.1 --reward_type reward_2
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --reward_type) REWARD_TYPE="$2"; shift ;;
        --beta) BETA="$2"; shift ;;
        --alpha) ALPHA="$2"; shift ;;
        --w_m) W_M="$2"; shift ;;
        --w_w) W_W="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --wandb_name) WANDB_NAME="$2"; shift ;;
        --freeze_e2c) FREEZE_E2C="$2"; shift ;;
        --base_policy_path) BASE_POLICY_PATH="$2"; shift ;;
        --e2c_dir) E2C_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "=================================================="
echo "Starting Residual TD3 Training (OOD 0.1)"
echo "Reward Type     : $REWARD_TYPE"
echo "Beta            : $BETA"
echo "Alpha           : $ALPHA"
echo "Main Weight     : $W_M"
echo "Wrist Weight    : $W_W"
echo "Seed            : $SEED"
echo "Freeze E2C      : $FREEZE_E2C"
echo "Base Policy Path: $BASE_POLICY_PATH"
echo "E2C Dir         : $E2C_DIR"
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

# OOD Initialization Range (0.1 means +/- 10cm, effectively 20x20cm range)
export OOD_RANGE="0.1"

# Run training
python resfit/rl_finetuning/scripts/train_residual_td3.py \
    task="Lift" \
    rl_camera="['observation.images.frontview','observation.images.robot0_eye_in_hand']" \
    wandb.project="lift_residual_rl" \
    wandb.name="${WANDB_NAME}" \
    seed="${SEED}" \
    algo.reward_type="${REWARD_TYPE}" \
    algo.reward_beta="${BETA}" \
    algo.reward_alpha="${ALPHA}" \
    algo.reward_w_m="${W_M}" \
    algo.reward_w_w="${W_W}" \
    algo.freeze_e2c="${FREEZE_E2C}" \
    base_policy_path="${BASE_POLICY_PATH}" \
    e2c_dir="${E2C_DIR}" \
    eval_interval_every_steps=2000
