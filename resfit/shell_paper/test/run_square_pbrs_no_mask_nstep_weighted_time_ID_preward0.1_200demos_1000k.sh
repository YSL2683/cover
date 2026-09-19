#!/bin/bash
# Robust PROJECT_ROOT detection
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT=$(cd "$SCRIPT_DIR" && while [ ! -d "resfit" ] && [ "$PWD" != "/" ]; do cd ..; done; pwd)
fi

# Script to run Residual TD3 with Potential-Based Reward Shaping using Similarity-Weighted Remaining Timesteps (1000k Steps, 200 Demos)
# Base Policy : 200 Demos Diffusion Policy
# E2C Encoder : lane/pretrained_e2c/square_200
# Reward      : reward_pbrs_no_mask_nstep_weighted_time (P_REWARD=0.1, W_M=0.3, W_W=0.7, BETA=1.0)
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
NUM_EPISODES=200
EVAL_INTERVAL=10000

# Base policy path (pointing to trained 200-demo diffusion policy)
BASE_POLICY_DIR="${PROJECT_ROOT}/resfit/my_lerobot_data/bc_run_2026-09-17_10-28-08_robomimic_square_v15_200_diffusion"
if [ -d "${BASE_POLICY_DIR}/best/policy" ]; then
    BASE_POLICY_PATH="${BASE_POLICY_DIR}/best/policy"
elif [ -d "${BASE_POLICY_DIR}/policy" ]; then
    BASE_POLICY_PATH="${BASE_POLICY_DIR}/policy"
else
    BASE_POLICY_PATH="${BASE_POLICY_DIR}"
fi

# E2C directory and Offline data directory for 200 demos
E2C_DIR="${PROJECT_ROOT}/lane/pretrained_e2c/square_200"
OFFLINE_DATA_DIR="${PROJECT_ROOT}/resfit/my_lerobot_data/ysl2683/robomimic_square_v15_200"

# Name for Weights & Biases
WANDB_PROJECT="square_residual_rl"
WANDB_NAME="${TASK}_${REWARD_TYPE}_beta${BETA}_scale${P_REWARD}_200demos_1000k_seed${SEED}"
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
        --num_episodes) NUM_EPISODES="$2"; shift ;;
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

# Normalize base_policy_path if user passes directory containing best/policy or policy
if [ -d "${BASE_POLICY_PATH}/best/policy" ]; then
    BASE_POLICY_PATH="${BASE_POLICY_PATH}/best/policy"
elif [ -d "${BASE_POLICY_PATH}/policy" ]; then
    BASE_POLICY_PATH="${BASE_POLICY_PATH}/policy"
fi

if [ -n "$CUSTOM_WANDB_NAME" ]; then
    WANDB_NAME="$CUSTOM_WANDB_NAME"
else
    WANDB_NAME="${TASK}_${REWARD_TYPE}_beta${BETA}_scale${P_REWARD}_200demos_1000k_seed${SEED}"
fi

echo "=================================================="
echo "Starting Residual TD3 Training for Square with Similarity-Weighted Timestep PBRS (200 Demos, 1000k Steps)"
echo "Target Task      : $TASK (In-Distribution Position & Orientation)"
echo "Total Timesteps  : $TOTAL_TIMESTEPS"
echo "Reward Type      : $REWARD_TYPE"
echo "Reward Scale     : $P_REWARD"
echo "Beta             : $BETA"
echo "Alpha            : $ALPHA"
echo "Main Weight      : $W_M"
echo "Wrist Weight     : $W_W"
echo "Action L2 Reg    : $RES_ACTION_REG"
echo "Seed             : $SEED"
echo "Freeze E2C       : $FREEZE_E2C"
echo "WandB Project    : $WANDB_PROJECT"
echo "WandB Name       : $WANDB_NAME"
echo "Base Policy Path : $BASE_POLICY_PATH"
echo "E2C Dir          : $E2C_DIR"
echo "Offline Data Dir : $OFFLINE_DATA_DIR"
echo "Offline Episodes : $NUM_EPISODES"
echo "DDIM Steps       : $DDIM_STEPS"
echo "Eval Interval    : $EVAL_INTERVAL"
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
export CACHE_DIR=${PROJECT_ROOT}/scratch/square_weighted_time_200demos_${CURRENT_TIME}

# Clear isolated scratch memory buffers for this task only
mkdir -p ${CACHE_DIR}

# Run training
python resfit/rl_finetuning/scripts/train_residual_td3.py \
    env_modifier.mode=none \
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
    offline_data.num_episodes="${NUM_EPISODES}" \
    eval_interval_every_steps="${EVAL_INTERVAL}" \
    torch_deterministic=false \
    base_policy.diffusion_ddim_steps="${DDIM_STEPS}"
