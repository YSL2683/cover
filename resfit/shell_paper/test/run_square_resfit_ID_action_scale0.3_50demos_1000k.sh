#!/bin/bash
# Robust PROJECT_ROOT detection
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT=$(cd "$SCRIPT_DIR" && while [ ! -d "resfit" ] && [ "$PWD" != "/" ]; do cd ..; done; pwd)
fi

# Script to run Residual TD3 (Vanilla ResFiT Baseline) on Square In-Distribution
# Action Scale: 0.3 (testing larger residual action bounds without visual guidance)
# Base Policy : 50 Demos Diffusion Policy
# Reward      : none (No reward shaping - sparse environment reward only)

# Default parameters
REWARD_TYPE="none"
BETA=1.0
ALPHA=0.98
W_M=0.0
W_W=0.0
P_REWARD=0.0
SEED=42
FREEZE_E2C="True"
TASK="Square"
ACTION_SCALE=0.3
RES_ACTION_REG=0.0
DDIM_STEPS=20
TOTAL_TIMESTEPS=1000000
NUM_EPISODES=50
EVAL_INTERVAL=10000

# Base policy path (50-demo diffusion policy)
BASE_POLICY_PATH="${PROJECT_ROOT}/resfit/my_lerobot_data/bc_run_2026-08-29_14-38-11_robomimic_square_v15_50_diffusion/policy_step_66000/policy"
E2C_DIR="${PROJECT_ROOT}/lane/pretrained_e2c/square"
OFFLINE_DATA_DIR="${PROJECT_ROOT}/resfit/my_lerobot_data/ysl2683/robomimic_square_v15_50"

# Name for Weights & Biases
WANDB_PROJECT="square_residual_rl"
WANDB_NAME="${TASK}_ID_resfit_action_scale${ACTION_SCALE}_1000k_seed${SEED}"
CUSTOM_WANDB_NAME=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --seed) SEED="$2"; shift ;;
        --total_timesteps) TOTAL_TIMESTEPS="$2"; shift ;;
        --eval_interval) EVAL_INTERVAL="$2"; shift ;;
        --wandb_project) WANDB_PROJECT="$2"; shift ;;
        --wandb_name) CUSTOM_WANDB_NAME="$2"; shift ;;
        --base_policy_path) BASE_POLICY_PATH="$2"; shift ;;
        --e2c_dir) E2C_DIR="$2"; shift ;;
        --offline_data_dir) OFFLINE_DATA_DIR="$2"; shift ;;
        --ddim_steps) DDIM_STEPS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Normalize base_policy_path
if [ -d "${BASE_POLICY_PATH}/best/policy" ]; then
    BASE_POLICY_PATH="${BASE_POLICY_PATH}/best/policy"
elif [ -d "${BASE_POLICY_PATH}/policy" ]; then
    BASE_POLICY_PATH="${BASE_POLICY_PATH}/policy"
fi

if [ -n "$CUSTOM_WANDB_NAME" ]; then
    WANDB_NAME="$CUSTOM_WANDB_NAME"
else
    WANDB_NAME="${TASK}_ID_resfit_action_scale${ACTION_SCALE}_1000k_seed${SEED}"
fi

echo "=================================================="
echo "Starting Residual TD3 Training for Square ID (Vanilla ResFiT, Action Scale 0.3)"
echo "Target Task      : $TASK (In-Distribution)"
echo "Action Scale     : $ACTION_SCALE"
echo "Total Timesteps  : $TOTAL_TIMESTEPS"
echo "Reward Type      : $REWARD_TYPE"
echo "Reward Scale     : $P_REWARD"
echo "Action L2 Reg    : $RES_ACTION_REG"
echo "Seed             : $SEED"
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

# Environment variables & Isolated Cache Directory
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export HF_HUB_OFFLINE=1
export LEROBOT_OFFLINE=1
export PYTHONHASHSEED=0
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
export CACHE_DIR=${PROJECT_ROOT}/scratch/square_id_resfit_ascale03_${CURRENT_TIME}

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
    agent.actor.action_scale="${ACTION_SCALE}" \
    agent.actor.action_l2_reg_weight="${RES_ACTION_REG}" \
    algo.freeze_e2c="${FREEZE_E2C}" \
    base_policy_path="${BASE_POLICY_PATH}" \
    e2c_dir="${E2C_DIR}" \
    offline_data.name="${OFFLINE_DATA_DIR}" \
    offline_data.num_episodes="${NUM_EPISODES}" \
    eval_interval_every_steps="${EVAL_INTERVAL}" \
    torch_deterministic=false \
    base_policy.diffusion_ddim_steps="${DDIM_STEPS}"
