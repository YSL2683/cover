#!/bin/bash
# ==============================================================================
# Post-Hoc Latent Space Visualization: Base Policy vs. Residual RL Policy
# Task: Square (Robomimic Square Nut Assembly)
# Target Model: resfit/outputs/2026-09-02_11-33-44_Square_reward_pbrs_no_mask_nstep_beta1.0_scale0.1
# ==============================================================================

# Determine project root dynamically
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# Activate Conda environment if available
if [ -f "/home/moai/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/moai/miniconda3/etc/profile.d/conda.sh"
    conda activate cover
fi

# Environment variables
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export LEROBOT_OFFLINE=1

# Default Paths (Relative to PROJECT_ROOT)
TASK="Square"
CHECKPOINT_PATH="resfit/outputs/2026-09-02_11-33-44_Square_reward_pbrs_no_mask_nstep_beta1.0_scale0.1/models/agent_best.pt"
BASE_POLICY_PATH="resfit/my_lerobot_data/bc_run_2026-08-29_14-38-11_robomimic_square_v15_50_diffusion/policy_step_66000/policy"
E2C_DIR="lane/pretrained_e2c/square"
OFFLINE_DATA_NAME="resfit/my_lerobot_data/ysl2683/robomimic_square_v15_50"
NUM_EPISODES=50
NUM_ENVS=5
ACTION_SCALE=0.1
SEED=42
BETA=1.0
OUTPUT_PATH="visualization/outputs/base_vs_residual_square_50ep.png"
CACHE_DIR="visualization/cache"

echo "=================================================================="
echo "Running Post-Hoc Latent Visualization"
echo "Project Root       : ${PROJECT_ROOT}"
echo "Task               : ${TASK}"
echo "Seed               : ${SEED}"
echo "Action Scale       : ${ACTION_SCALE}"
echo "Checkpoint         : ${CHECKPOINT_PATH}"
echo "Base Policy        : ${BASE_POLICY_PATH}"
echo "E2C Directory      : ${E2C_DIR}"
echo "Success Target     : ${NUM_EPISODES} episodes per policy"
echo "Output Path        : ${OUTPUT_PATH}"
echo "=================================================================="

# Run Python script passing all configured defaults and user override arguments
python "${SCRIPT_DIR}/visualize_base_vs_residual.py" \
    --task "${TASK}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --base_policy_path "${BASE_POLICY_PATH}" \
    --e2c_dir "${E2C_DIR}" \
    --offline_data_name "${OFFLINE_DATA_NAME}" \
    --num_episodes "${NUM_EPISODES}" \
    --num_envs "${NUM_ENVS}" \
    --action_scale "${ACTION_SCALE}" \
    --seed "${SEED}" \
    --beta "${BETA}" \
    --output_path "${OUTPUT_PATH}" \
    --cache_dir "${CACHE_DIR}" \
    "$@"

