#!/bin/bash
# ==============================================================================
# Run Case Study Visual Comparison on Square Task
# Comparing:
# 1. Base Policy (Diffusion BC, zero residual)
# 2. ResFiT (Sparse RL baseline)
# 3. Ours (V-PBRS Residual RL)
# 4. Decoupled View Similarities (S_m in Red, S_w in Green)
# ==============================================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# Ensure conda environment 'cover' is activated
if [ -f "/home/moai/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/moai/miniconda3/etc/profile.d/conda.sh"
    conda activate cover
fi

export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export LEROBOT_OFFLINE=1

# Checkpoint paths specified by user
BASE_POLICY_PATH="resfit/my_lerobot_data/bc_run_2026-08-29_14-38-11_robomimic_square_v15_50_diffusion/policy_step_66000/policy"
RESFIT_CKPT="resfit/outputs/2026-08-31_16-01-18_Square_ablation_no_dense_seed42/models/agent_best.pt"
OURS_CKPT="resfit/outputs/2026-09-02_11-33-44_Square_reward_pbrs_no_mask_nstep_beta1.0_scale0.1/models/agent_best.pt"
E2C_DIR="lane/pretrained_e2c/square"
OUTPUT_DIR="visualization/outputs/case_study"

echo "=================================================================="
echo "Starting Case Study Analysis on Square Task"
echo "Project Root   : ${PROJECT_ROOT}"
echo "Base Policy    : ${BASE_POLICY_PATH}"
echo "ResFiT Ckpt    : ${RESFIT_CKPT}"
echo "Ours Ckpt      : ${OURS_CKPT}"
echo "Output Dir     : ${OUTPUT_DIR}"
echo "=================================================================="

python "${SCRIPT_DIR}/case_study_analysis.py" \
    --task "Square" \
    --base_policy_path "${BASE_POLICY_PATH}" \
    --resfit_ckpt "${RESFIT_CKPT}" \
    --ours_ckpt "${OURS_CKPT}" \
    --e2c_dir "${E2C_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --start_seed 0 \
    --max_search_seeds 30 \
    --ddim_steps 20 \
    "$@"
