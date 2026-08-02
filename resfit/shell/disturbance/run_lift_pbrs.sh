#!/bin/bash
# Script to run Residual TD3 with Potential-Based Reward Shaping (PBRS) under Physical Disturbances

# Default parameters
REWARD_TYPE="reward_pbrs"
BETA=1.0
ALPHA=0.98
W_M=0.3
W_W=0.7
P_REWARD=100.0  # Scaling factor for PBRS difference magnitude
SEED=42
FREEZE_E2C="True"
BASE_POLICY_PATH="resfit/my_lerobot_data/bc_run_2026-07-30_16-20-35_lane_lift_id_20_aligned_diffusion/policy_step_5000/policy"
E2C_DIR="/home/moai/ysl_ws/cover/lane/pretrained_e2c/lift"

# Disturbance parameters
DIST_STEP_RANGE="[0, 17]"
DIST_FORCE_RANGE="[250, 250]"
NUM_DISTURBANCES=1

# Name for Weights & Biases
WANDB_NAME="disturb_pbrs_beta${BETA}_scale${P_REWARD}_f${NUM_DISTURBANCES}"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --reward_type) REWARD_TYPE="$2"; shift ;;
        --beta) BETA="$2"; shift ;;
        --alpha) ALPHA="$2"; shift ;;
        --w_m) W_M="$2"; shift ;;
        --w_w) W_W="$2"; shift ;;
        --p_reward) P_REWARD="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --wandb_name) WANDB_NAME="$2"; shift ;;
        --freeze_e2c) FREEZE_E2C="$2"; shift ;;
        --base_policy_path) BASE_POLICY_PATH="$2"; shift ;;
        --e2c_dir) E2C_DIR="$2"; shift ;;
        --dist_step_range) DIST_STEP_RANGE="$2"; shift ;;
        --dist_force_range) DIST_FORCE_RANGE="$2"; shift ;;
        --num_disturbances) NUM_DISTURBANCES="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "=================================================="
echo "Starting Residual TD3 Training with V-PBRS (Disturbance Env)"
echo "Reward Type     : $REWARD_TYPE"
echo "Reward Scale    : $P_REWARD"
echo "Beta            : $BETA"
echo "Alpha           : $ALPHA"
echo "Main Weight     : $W_M"
echo "Wrist Weight    : $W_W"
echo "Dist. Steps     : $DIST_STEP_RANGE"
echo "Dist. Force     : $DIST_FORCE_RANGE"
echo "Dist. Count     : $NUM_DISTURBANCES"
echo "Seed            : $SEED"
echo "Freeze E2C      : $FREEZE_E2C"
echo "WandB Name      : $WANDB_NAME"
echo "=================================================="

# Clear scratch memory buffers
rm -rf /home/moai/ysl_ws/cover/scratch/online_buffer_cache/* /home/moai/ysl_ws/cover/scratch/offline_buffer_cache/*
rm -rf /home/moai/ysl_ws/cover/scratch/online/* /home/moai/ysl_ws/cover/scratch/offline/*

# Environment variables
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/moai/ysl_ws/cover:$PYTHONPATH
export CACHE_DIR=/home/moai/ysl_ws/cover/scratch
export HF_HUB_OFFLINE=1
export LEROBOT_OFFLINE=1

# Run training
python resfit/rl_finetuning/scripts/train_residual_td3.py \
    env_modifier.ood_position.x_bounds="[-0.025, 0.025]" \
    env_modifier.ood_position.y_bounds="[-0.025, 0.025]" \
    env_modifier.disturbance.step_range="${DIST_STEP_RANGE}" \
    env_modifier.disturbance.force_range="${DIST_FORCE_RANGE}" \
    env_modifier.disturbance.num_disturbances=${NUM_DISTURBANCES} \
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
    algo.p_reward="${P_REWARD}" \
    algo.freeze_e2c="${FREEZE_E2C}" \
    base_policy_path="${BASE_POLICY_PATH}" \
    e2c_dir="${E2C_DIR}" \
    eval_interval_every_steps=2000
