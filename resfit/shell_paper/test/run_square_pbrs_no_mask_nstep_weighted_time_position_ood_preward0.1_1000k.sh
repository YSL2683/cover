#!/bin/bash
# Robust PROJECT_ROOT detection
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT=$(cd "$SCRIPT_DIR" && while [ ! -d "resfit" ] && [ "$PWD" != "/" ]; do cd ..; done; pwd)
fi

# Script to run Residual TD3 with Potential-Based Reward Shaping using Similarity-Weighted Remaining Timesteps (1000k Steps)
# Environment: Square Position OOD (ID span + R_nut (4.375cm total per axis): X: [-0.136875, -0.088125], Y: [0.088125, 0.246875])
# Note: Uses task-isolated CACHE_DIR to support concurrent multi-task Residual RL training.

# Default parameters
REWARD_TYPE="reward_pbrs_no_mask_nstep_weighted_time"
BETA=1.0
ALPHA=0.98
W_M=0.3
W_W=0.7
P_REWARD=0.1
SEED=42
FREEZE_E2C="True"
TASK="Square"
RES_ACTION_REG=0.00005
DDIM_STEPS=20
TOTAL_TIMESTEPS=1000000
EVAL_INTERVAL=10000

# Base policy path
BASE_POLICY_PATH="${PROJECT_ROOT}/resfit/my_lerobot_data/bc_run_2026-08-29_14-38-11_robomimic_square_v15_50_diffusion/policy_step_66000/policy"
E2C_DIR="${PROJECT_ROOT}/lane/pretrained_e2c/square"
OFFLINE_DATA_DIR="${PROJECT_ROOT}/resfit/my_lerobot_data/ysl2683/robomimic_square_v15_50"

# Name for Weights & Biases
WANDB_PROJECT="square_residual_rl"
WANDB_NAME="${TASK}_position_ood_4.875x15.875_${REWARD_TYPE}_beta${BETA}_scale${P_REWARD}_1000k_seed${SEED}"
CUSTOM_WANDB_NAME=""

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
        --total_timesteps) TOTAL_TIMESTEPS="$2"; shift ;;
        --eval_interval) EVAL_INTERVAL="$2"; shift ;;
        --wandb_project) WANDB_PROJECT="$2"; shift ;;
        --wandb_name) CUSTOM_WANDB_NAME="$2"; shift ;;
        --freeze_e2c) FREEZE_E2C="$2"; shift ;;
        --base_policy_path) BASE_POLICY_PATH="$2"; shift ;;
        --e2c_dir) E2C_DIR="$2"; shift ;;
        --offline_data_dir) OFFLINE_DATA_DIR="$2"; shift ;;
        --ddim_steps) DDIM_STEPS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -n "$CUSTOM_WANDB_NAME" ]; then
    WANDB_NAME="$CUSTOM_WANDB_NAME"
else
    WANDB_NAME="${TASK}_position_ood_4.875x15.875_${REWARD_TYPE}_beta${BETA}_scale${P_REWARD}_1000k_seed${SEED}"
fi

echo "=================================================="
echo "Starting Residual TD3 Training for Square Position OOD with Similarity-Weighted Timestep PBRS (1000k Steps)"
echo "Target Task     : $TASK (Position OOD: X[-0.136875, -0.088125], Y[0.088125, 0.246875])"
echo "Total Timesteps : $TOTAL_TIMESTEPS"
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
echo "Offline Data Dir: $OFFLINE_DATA_DIR"
echo "DDIM Steps      : $DDIM_STEPS"
echo "Eval Interval   : $EVAL_INTERVAL"
echo "=================================================="

# Ensure conda environment 'cover' is activated
if [ -d "/home/ysl2683/anaconda3/envs/cover/bin" ]; then
    export PATH="/home/ysl2683/anaconda3/envs/cover/bin:${PATH}"
elif [ -d "/home/moai/miniconda3/envs/cover/bin" ]; then
    export PATH="/home/moai/miniconda3/envs/cover/bin:${PATH}"
fi
if [ -f "/home/ysl2683/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/home/ysl2683/anaconda3/etc/profile.d/conda.sh"
    conda activate cover
elif [ -f "/home/moai/miniconda3/etc/profile.d/conda.sh" ]; then
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
export CACHE_DIR=${PROJECT_ROOT}/scratch/square_pos_ood_weighted_time_${CURRENT_TIME}

# Clear isolated scratch memory buffers for this task only
mkdir -p ${CACHE_DIR}

# Run training
python resfit/rl_finetuning/scripts/train_residual_td3.py \
    env_modifier.mode=ood_position \
    env_modifier.ood_position.x_bounds="[-0.136875, -0.088125]" \
    env_modifier.ood_position.y_bounds="[0.088125, 0.246875]" \
    env_modifier.disturbance=null \
    task="${TASK}" \
    rl_camera="['observation.images.agentview','observation.images.robot0_eye_in_hand']" \
    wandb.project="${WANDB_PROJECT}" \
    wandb.name="${WANDB_NAME}" \
    seed="${SEED}" \
    algo.total_timesteps="${TOTAL_TIMESTEPS}" \
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
    eval_interval_every_steps="${EVAL_INTERVAL}" \
    torch_deterministic=false \
    base_policy.diffusion_ddim_steps="${DDIM_STEPS}"
