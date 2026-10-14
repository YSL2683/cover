#!/bin/bash
# ==============================================================================
# Script     : run_square_id_weak_unified_seed62.sh
# Category   : Exp 1: ID - Weak Base (50 Demos)
# Title      : In-Distribution Weak Base Unified Latent PBRS
# Seed       : 62
# ==============================================================================
set -e

# Robust PROJECT_ROOT detection
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT=$(cd "$SCRIPT_DIR" && while [ ! -d "resfit" ] && [ "$PWD" != "/" ]; do cd ..; done; pwd)
fi

# Default run configuration
TASK="Square"
SEED=62
TOTAL_TIMESTEPS=1000000
EVAL_INTERVAL=10000
DDIM_STEPS=20
ACTION_SCALE=0.1
RES_ACTION_REG=5e-05
REWARD_TYPE="reward_pbrs_unified_no_mask_nstep"
P_REWARD=0.1
BETA=1.0
ALPHA=0.98
W_M=0.3
W_W=0.7
Z_DIM=16
NUM_EPISODES=50
FREEZE_E2C="True"

# WandB Logging
WANDB_PROJECT="paper_square"
WANDB_NAME="Square_ID_weak_50d_unified_pbrs_scale0.1_1000k_seed62"

# Paths
BASE_POLICY_PATH="${PROJECT_ROOT}/resfit/my_lerobot_data/bc_run_2026-08-29_14-38-11_robomimic_square_v15_50_diffusion/policy_step_66000/policy"
OFFLINE_DATA_DIR="${PROJECT_ROOT}/resfit/my_lerobot_data/ysl2683/robomimic_square_v15_50"
E2C_DIR="${PROJECT_ROOT}/lane/pretrained_e2c_unified/square"

# Conda environment activation (compatible with PC1 and PC2)
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

# Environment variables & thread caps
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/lane:$PYTHONPATH
export HF_HUB_OFFLINE=1
export LEROBOT_OFFLINE=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OFFLINE_CACHE_DIR="${PROJECT_ROOT}/scratch/offline_buffers"

# Isolated Cache Directory
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
export CACHE_DIR=${PROJECT_ROOT}/scratch/square_id_weak_unified_seed${SEED}_${CURRENT_TIME}
mkdir -p "${CACHE_DIR}"

echo "=============================================================================="
echo "Starting Residual TD3 Run: ${WANDB_NAME}"
echo "Category    : Exp 1: ID - Weak Base (50 Demos)"
echo "Task        : ${TASK}"
echo "Seed        : ${SEED}"
echo "Timesteps   : ${TOTAL_TIMESTEPS}"
echo "Reward Type : ${REWARD_TYPE} (scale: ${P_REWARD})"
echo "Action Scale: ${ACTION_SCALE} (reg: ${RES_ACTION_REG})"
echo "Base Policy : ${BASE_POLICY_PATH}"
echo "Offline Data: ${OFFLINE_DATA_DIR} (${NUM_EPISODES} demos)"
echo "E2C Model   : ${E2C_DIR}"
echo "Cache Dir   : ${CACHE_DIR}"
echo "Start Time  : $(date)"
echo "=============================================================================="

# Launch training
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
    algo.e2c_mode="unified" \
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

echo "=============================================================================="
echo "Finished Residual TD3 Run: ${WANDB_NAME}"
echo "End Time  : $(date)"
echo "=============================================================================="
