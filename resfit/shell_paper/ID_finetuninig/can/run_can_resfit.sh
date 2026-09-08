#!/bin/bash
# Robust PROJECT_ROOT detection
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT=$(cd "$SCRIPT_DIR" && while [ ! -d "resfit" ] && [ "$PWD" != "/" ]; do cd ..; done; pwd)
fi

# Script to run Residual TD3 (ResFiT Baseline) WITHOUT Additional Reward Shaping (Reward None) for Can
# Note: Uses task-isolated CACHE_DIR to support concurrent multi-task Residual RL training.

# Default parameters
REWARD_TYPE="none"
BETA=1.0
ALPHA=0.98
W_M=0.0
W_W=0.0
P_REWARD=0.0  # Set to 0.0 as no additional reward shaping is used
SEED=42
FREEZE_E2C="True"
TASK="Can"
RES_ACTION_REG=0.0 # Regularization for residual action magnitude
NUM_EPISODES=10
DDIM_STEPS=20

# Base policy path (pointing to policy in resfit/my_lerobot_data)
BASE_POLICY_PATH="${PROJECT_ROOT}/resfit/my_lerobot_data/bc_run_2026-09-06_21-23-18_robomimic_can_v15_10_diffusion/best/policy"
E2C_DIR="${PROJECT_ROOT}/lane/pretrained_e2c/can"
OFFLINE_DATA_DIR="${PROJECT_ROOT}/resfit/my_lerobot_data/ysl2683/robomimic_can_v15_10"

# Name for Weights & Biases
WANDB_PROJECT="can_residual_rl"
WANDB_NAME="${TASK}_ID_resfit_seed${SEED}"

# Parse command line arguments
CUSTOM_WANDB_NAME=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --reward_type) REWARD_TYPE="$2"; shift ;;
        --beta) BETA="$2"; shift ;;
        --alpha) ALPHA="$2"; shift ;;
        --w_m) W_M="$2"; shift ;;
        --w_w) W_W="$2"; shift ;;
        --p_reward) P_REWARD="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --wandb_project) WANDB_PROJECT="$2"; shift ;;
        --wandb_name) CUSTOM_WANDB_NAME="$2"; shift ;;
        --freeze_e2c) FREEZE_E2C="$2"; shift ;;
        --base_policy_path) BASE_POLICY_PATH="$2"; shift ;;
        --e2c_dir) E2C_DIR="$2"; shift ;;
        --offline_data_dir) OFFLINE_DATA_DIR="$2"; shift ;;
        --num_episodes) NUM_EPISODES="$2"; shift ;;
        --ddim_steps) DDIM_STEPS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -n "$CUSTOM_WANDB_NAME" ]; then
    WANDB_NAME="$CUSTOM_WANDB_NAME"
else
    WANDB_NAME="${TASK}_ID_resfit_seed${SEED}"
fi

echo "=================================================="
echo "Starting Residual TD3 (ResFiT) Training for Can WITHOUT Additional Reward"
echo "Target Task     : $TASK (In-Distribution Position & Orientation)"
echo "Reward Type     : $REWARD_TYPE (No additional reward shaping)"
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
echo "Offline Data Dir: $OFFLINE_DATA_DIR"
echo "Num Episodes    : $NUM_EPISODES"
echo "DDIM Steps      : $DDIM_STEPS"
echo "=================================================="

# Ensure conda environment 'cover' is activated
if [ -f "/home/moai/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/moai/miniconda3/etc/profile.d/conda.sh"
    conda activate cover
fi

# Environment variables & Isolated Cache Directory for Multi-Task Concurrency
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export HF_HUB_OFFLINE=1
export LEROBOT_OFFLINE=1
export PYTHONHASHSEED=0
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
export CACHE_DIR=${PROJECT_ROOT}/scratch/can_${CURRENT_TIME}

# Clear isolated scratch memory buffers for this task only
mkdir -p ${CACHE_DIR}

# Run training with task="Can" and wandb.project="can_residual_rl"
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
    offline_data.num_episodes="${NUM_EPISODES}" \
    eval_interval_every_steps=2000 \
    torch_deterministic=false \
    base_policy.diffusion_ddim_steps="${DDIM_STEPS}"
