#!/bin/bash
# ==============================================================================
# Queue Script: queue_exp3_ablation.sh
# Title       : Exp 3: Reward Component Ablations (Weak 50d, 9 runs)
# Total Runs  : 9
# ==============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT=$(cd "$SCRIPT_DIR" && while [ ! -d "resfit" ] && [ "$PWD" != "/" ]; do cd ..; done; pwd)
fi

echo "=============================================================================="
echo "Starting Execution Queue: Exp 3: Reward Component Ablations (Weak 50d, 9 runs)"
echo "Total Runs to Execute: 9"
echo "Project Root         : ${PROJECT_ROOT}"
echo "Queue Start Time     : $(date)"
echo "=============================================================================="

echo "=============================================================================="
echo ">>> [1/9] STARTING: run_square_ablation_weak_no_pbrs_seed42.sh"
echo ">>> Path: resfit/shell_final/square/exp3_ablation/run_square_ablation_weak_no_pbrs_seed42.sh"
echo ">>> Time: $(date)"
echo "=============================================================================="
bash "${PROJECT_ROOT}/resfit/shell_final/square/exp3_ablation/run_square_ablation_weak_no_pbrs_seed42.sh"
echo ">>> [1/9] FINISHED: run_square_ablation_weak_no_pbrs_seed42.sh"
echo ">>> Time: $(date)"

echo "=============================================================================="
echo ">>> [2/9] STARTING: run_square_ablation_weak_no_pbrs_seed52.sh"
echo ">>> Path: resfit/shell_final/square/exp3_ablation/run_square_ablation_weak_no_pbrs_seed52.sh"
echo ">>> Time: $(date)"
echo "=============================================================================="
bash "${PROJECT_ROOT}/resfit/shell_final/square/exp3_ablation/run_square_ablation_weak_no_pbrs_seed52.sh"
echo ">>> [2/9] FINISHED: run_square_ablation_weak_no_pbrs_seed52.sh"
echo ">>> Time: $(date)"

echo "=============================================================================="
echo ">>> [3/9] STARTING: run_square_ablation_weak_no_pbrs_seed62.sh"
echo ">>> Path: resfit/shell_final/square/exp3_ablation/run_square_ablation_weak_no_pbrs_seed62.sh"
echo ">>> Time: $(date)"
echo "=============================================================================="
bash "${PROJECT_ROOT}/resfit/shell_final/square/exp3_ablation/run_square_ablation_weak_no_pbrs_seed62.sh"
echo ">>> [3/9] FINISHED: run_square_ablation_weak_no_pbrs_seed62.sh"
echo ">>> Time: $(date)"

echo "=============================================================================="
echo ">>> [4/9] STARTING: run_square_ablation_weak_no_action_reg_seed42.sh"
echo ">>> Path: resfit/shell_final/square/exp3_ablation/run_square_ablation_weak_no_action_reg_seed42.sh"
echo ">>> Time: $(date)"
echo "=============================================================================="
bash "${PROJECT_ROOT}/resfit/shell_final/square/exp3_ablation/run_square_ablation_weak_no_action_reg_seed42.sh"
echo ">>> [4/9] FINISHED: run_square_ablation_weak_no_action_reg_seed42.sh"
echo ">>> Time: $(date)"

echo "=============================================================================="
echo ">>> [5/9] STARTING: run_square_ablation_weak_no_action_reg_seed52.sh"
echo ">>> Path: resfit/shell_final/square/exp3_ablation/run_square_ablation_weak_no_action_reg_seed52.sh"
echo ">>> Time: $(date)"
echo "=============================================================================="
bash "${PROJECT_ROOT}/resfit/shell_final/square/exp3_ablation/run_square_ablation_weak_no_action_reg_seed52.sh"
echo ">>> [5/9] FINISHED: run_square_ablation_weak_no_action_reg_seed52.sh"
echo ">>> Time: $(date)"

echo "=============================================================================="
echo ">>> [6/9] STARTING: run_square_ablation_weak_no_action_reg_seed62.sh"
echo ">>> Path: resfit/shell_final/square/exp3_ablation/run_square_ablation_weak_no_action_reg_seed62.sh"
echo ">>> Time: $(date)"
echo "=============================================================================="
bash "${PROJECT_ROOT}/resfit/shell_final/square/exp3_ablation/run_square_ablation_weak_no_action_reg_seed62.sh"
echo ">>> [6/9] FINISHED: run_square_ablation_weak_no_action_reg_seed62.sh"
echo ">>> Time: $(date)"

echo "=============================================================================="
echo ">>> [7/9] STARTING: run_square_ablation_weak_no_time_weight_seed42.sh"
echo ">>> Path: resfit/shell_final/square/exp3_ablation/run_square_ablation_weak_no_time_weight_seed42.sh"
echo ">>> Time: $(date)"
echo "=============================================================================="
bash "${PROJECT_ROOT}/resfit/shell_final/square/exp3_ablation/run_square_ablation_weak_no_time_weight_seed42.sh"
echo ">>> [7/9] FINISHED: run_square_ablation_weak_no_time_weight_seed42.sh"
echo ">>> Time: $(date)"

echo "=============================================================================="
echo ">>> [8/9] STARTING: run_square_ablation_weak_no_time_weight_seed52.sh"
echo ">>> Path: resfit/shell_final/square/exp3_ablation/run_square_ablation_weak_no_time_weight_seed52.sh"
echo ">>> Time: $(date)"
echo "=============================================================================="
bash "${PROJECT_ROOT}/resfit/shell_final/square/exp3_ablation/run_square_ablation_weak_no_time_weight_seed52.sh"
echo ">>> [8/9] FINISHED: run_square_ablation_weak_no_time_weight_seed52.sh"
echo ">>> Time: $(date)"

echo "=============================================================================="
echo ">>> [9/9] STARTING: run_square_ablation_weak_no_time_weight_seed62.sh"
echo ">>> Path: resfit/shell_final/square/exp3_ablation/run_square_ablation_weak_no_time_weight_seed62.sh"
echo ">>> Time: $(date)"
echo "=============================================================================="
bash "${PROJECT_ROOT}/resfit/shell_final/square/exp3_ablation/run_square_ablation_weak_no_time_weight_seed62.sh"
echo ">>> [9/9] FINISHED: run_square_ablation_weak_no_time_weight_seed62.sh"
echo ">>> Time: $(date)"

echo "=============================================================================="
echo ">>> ALL 9 RUNS IN QUEUE COMPLETED SUCCESSFULLY!"
echo "Queue Finish Time: $(date)"
echo "=============================================================================="
