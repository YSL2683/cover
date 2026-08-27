#!/bin/bash
# Script to run Residual TD3 with Potential-Based Reward Shaping (PBRS) for SquareID
# Note: Uses task-isolated CACHE_DIR to support concurrent multi-task Residual RL training.

# Default parameters
REWARD_TYPE="reward_pbrs"
BETA=1.0
ALPHA=0.98
W_M=0.5
W_W=0.5
P_REWARD=1.0  # Scaling factor for PBRS difference magnitude
SEED=42
FREEZE_E2C="True"
TASK="Square"
RES_ACTION_REG=0.0005  # Regularization for residual action magnitude

# Base policy path (placeholder pointing to policy in resfit/my_lerobot_data)
BASE_POLICY_PATH="/home/moai/ysl_ws/cover/resfit/my_lerobot_data/bc_run_2026-08-23_21-14-35_robomimic_square_v15_50_diffusion/best_step_14000/policy"
E2C_DIR="/home/moai/ysl_ws/cover/lane/pretrained_e2c/square"
OFFLINE_DATA_DIR="/home/moai/ysl_ws/cover/resfit/my_lerobot_data/ysl2683/robomimic_square_v15_50"

# Name for Weights & Biases
WANDB_PROJECT="square_residual_rl"
WANDB_NAME="${TASK}_${REWARD_TYPE}_beta${BETA}_scale${P_REWARD}_qclipping_w55"

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
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "=================================================="
echo "Starting Residual TD3 Training for Square with V-PBRS (Q-Clipping + 5:5 Weights)"
echo "Target Task     : $TASK (In-Distribution Position & Orientation)"
echo "Reward Type     : $REWARD_TYPE"
echo "Reward Scale    : $P_REWARD"
echo "Beta            : $BETA"
echo "Alpha           : $ALPHA"
echo "Main Weight     : $W_M"
echo "Wrist Weight    : $W_W"
echo "Seed            : $SEED"
echo "Freeze E2C      : $FREEZE_E2C"
echo "WandB Project   : $WANDB_PROJECT"
echo "WandB Name      : $WANDB_NAME"
echo "Base Policy Path: $BASE_POLICY_PATH"
echo "E2C Dir         : $E2C_DIR"
echo "=================================================="

# Environment variables & Isolated Cache Directory for Multi-Task Concurrency
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/moai/ysl_ws/cover:$PYTHONPATH
export HF_HUB_OFFLINE=1
export LEROBOT_OFFLINE=1
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
export CACHE_DIR=/home/moai/ysl_ws/cover/scratch/square_${CURRENT_TIME}

# Clear isolated scratch memory buffers for this task only
mkdir -p ${CACHE_DIR}

# Run training with task="SquareOOD" and wandb.project="square_residual_rl"
python resfit/rl_finetuning/scripts/train_residual_td3.py \
    env_modifier.mode=none \
    env_modifier.disturbance=null \
    task="${TASK}" \
    rl_camera="['observation.images.agentview','observation.images.robot0_eye_in_hand']" \
    wandb.project="${WANDB_PROJECT}" \
    wandb.name="${WANDB_NAME}" \
    seed="${SEED}" \
    algo.reward_type="${REWARD_TYPE}" \
    algo.reward_beta="${BETA}" \
    algo.reward_alpha="${ALPHA}" \
    algo.reward_w_m="${W_M}" \
    algo.reward_w_w="${W_W}" \
    algo.p_reward="${P_REWARD}" \
    agent.actor.action_l2_reg_weight="${RES_ACTION_REG}" \
    agent.q_target_clip_max=1.0 \
    algo.freeze_e2c="${FREEZE_E2C}" \
    base_policy_path="${BASE_POLICY_PATH}" \
    e2c_dir="${E2C_DIR}" \
    offline_data.name="${OFFLINE_DATA_DIR}" \
    eval_interval_every_steps=2000
