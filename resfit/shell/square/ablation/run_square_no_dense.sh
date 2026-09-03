#!/bin/bash
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
# Ablation Script: Residual TD3 WITHOUT Dense Reward (No PBRS)
# This serves as a baseline (Sparse Reward only) for SquareID.
# Note: Uses task-isolated CACHE_DIR to support concurrent multi-task Residual RL training.

# Default parameters
REWARD_TYPE="none"   # "none" bypasses the shaper completely (no dense reward)
BETA=1.0
ALPHA=0.98
W_M=0.3
W_W=0.7
P_REWARD=0.0  # Set to 0.0 just to be explicit that there is no dense reward
SEED=42
FREEZE_E2C="True"
TASK="Square"
RES_ACTION_REG=0.0005  # Regularization for residual action magnitude

# Base policy path (placeholder pointing to policy in resfit/my_lerobot_data)
BASE_POLICY_PATH="${PROJECT_ROOT}/resfit/my_lerobot_data/bc_run_2026-08-29_14-38-11_robomimic_square_v15_50_diffusion/policy_step_66000/policy"
E2C_DIR="${PROJECT_ROOT}/lane/pretrained_e2c/square"
OFFLINE_DATA_DIR="${PROJECT_ROOT}/resfit/my_lerobot_data/ysl2683/robomimic_square_v15_50"

# Name for Weights & Biases
WANDB_PROJECT="square_residual_rl"
WANDB_NAME="${TASK}_ablation_no_dense_seed${SEED}"

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
echo "Starting Ablation Training for Square WITHOUT Dense Reward"
echo "Target Task     : $TASK (In-Distribution Position & Orientation)"
echo "Reward Type     : $REWARD_TYPE (No PBRS)"
echo "Reward Scale    : $P_REWARD"
echo "Seed            : $SEED"
echo "Freeze E2C      : $FREEZE_E2C"
echo "WandB Project   : $WANDB_PROJECT"
echo "WandB Name      : $WANDB_NAME"
echo "Base Policy Path: $BASE_POLICY_PATH"
echo "E2C Dir         : $E2C_DIR"
echo "=================================================="

# Environment variables & Isolated Cache Directory for Multi-Task Concurrency
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export HF_HUB_OFFLINE=1
export LEROBOT_OFFLINE=1
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
export CACHE_DIR=${PROJECT_ROOT}/scratch/square_${CURRENT_TIME}

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
    algo.freeze_e2c="${FREEZE_E2C}" \
    base_policy_path="${BASE_POLICY_PATH}" \
    e2c_dir="${E2C_DIR}" \
    offline_data.name="${OFFLINE_DATA_DIR}" \
    eval_interval_every_steps=2000 \
    torch_deterministic=true
