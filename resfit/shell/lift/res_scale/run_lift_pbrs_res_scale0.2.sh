#!/bin/bash
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
# Script to run Residual TD3 with Potential-Based Reward Shaping (PBRS)
# with increased residual action scale (0.2)

# Default parameters
REWARD_TYPE="reward_pbrs"
BETA=1.0
ALPHA=0.98
W_M=0.3
W_W=0.7
P_REWARD=100.0  # Increased scaling factor for PBRS difference magnitude
SEED=42
FREEZE_E2C="True"
BASE_POLICY_PATH="resfit/my_lerobot_data/bc_run_2026-07-30_16-20-35_lane_lift_id_20_aligned_diffusion/policy_step_5000/policy"
E2C_DIR="${PROJECT_ROOT}/lane/pretrained_e2c/lift"
ACTION_SCALE=0.2

# Name for Weights & Biases
WANDB_NAME="pbrs_beta${BETA}_scale${P_REWARD}_res_scale${ACTION_SCALE}"

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
        --action_scale) ACTION_SCALE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "=================================================="
echo "Starting Residual TD3 Training with V-PBRS (Increased Scale)"
echo "Reward Type     : $REWARD_TYPE"
echo "Reward Scale    : $P_REWARD"
echo "Action Scale    : $ACTION_SCALE"
echo "Beta            : $BETA"
echo "Alpha           : $ALPHA"
echo "Main Weight     : $W_M"
echo "Wrist Weight    : $W_W"
echo "Seed            : $SEED"
echo "Freeze E2C      : $FREEZE_E2C"
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
    env_modifier.mode=ood_position \
    env_modifier.ood_position.x_bounds="[-0.1, 0.1]" \
    env_modifier.ood_position.y_bounds="[-0.1, 0.1]" \
    env_modifier.disturbance=null \
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
    agent.actor.action_scale="${ACTION_SCALE}" \
    base_policy_path="${BASE_POLICY_PATH}" \
    e2c_dir="${E2C_DIR}" \
    eval_interval_every_steps=2000
