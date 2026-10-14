#!/bin/bash
# ==============================================================================
# Queue Script: queue_pc1_exp1_id_strong.sh
# Title       : PC1 - Exp 1: In-Distribution Strong Base (200 Demos, 9 runs)
# Total Runs  : 9
# ==============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT=$(cd "$SCRIPT_DIR" && while [ ! -d "resfit" ] && [ "$PWD" != "/" ]; do cd ..; done; pwd)
fi

echo "=============================================================================="
echo "Starting Execution Queue: PC1 - Exp 1: In-Distribution Strong Base (200 Demos, 9 runs)"
echo "Total Runs to Execute: 9"
echo "Project Root         : ${PROJECT_ROOT}"
echo "Queue Start Time     : $(date)"
echo "=============================================================================="

echo "=============================================================================="
echo ">>> [1/9] STARTING: run_square_id_strong_resfit_seed42.sh"
echo ">>> Path: resfit/shell_final/square/exp1_id/strong_200d/run_square_id_strong_resfit_seed42.sh"
echo ">>> Time: $(date)"
echo "=============================================================================="
bash "${PROJECT_ROOT}/resfit/shell_final/square/exp1_id/strong_200d/run_square_id_strong_resfit_seed42.sh"
echo ">>> [1/9] FINISHED: run_square_id_strong_resfit_seed42.sh"
echo ">>> Time: $(date)"

echo "=============================================================================="
echo ">>> [2/9] STARTING: run_square_id_strong_resfit_seed52.sh"
echo ">>> Path: resfit/shell_final/square/exp1_id/strong_200d/run_square_id_strong_resfit_seed52.sh"
echo ">>> Time: $(date)"
echo "=============================================================================="
bash "${PROJECT_ROOT}/resfit/shell_final/square/exp1_id/strong_200d/run_square_id_strong_resfit_seed52.sh"
echo ">>> [2/9] FINISHED: run_square_id_strong_resfit_seed52.sh"
echo ">>> Time: $(date)"

echo "=============================================================================="
echo ">>> [3/9] STARTING: run_square_id_strong_resfit_seed62.sh"
echo ">>> Path: resfit/shell_final/square/exp1_id/strong_200d/run_square_id_strong_resfit_seed62.sh"
echo ">>> Time: $(date)"
echo "=============================================================================="
bash "${PROJECT_ROOT}/resfit/shell_final/square/exp1_id/strong_200d/run_square_id_strong_resfit_seed62.sh"
echo ">>> [3/9] FINISHED: run_square_id_strong_resfit_seed62.sh"
echo ">>> Time: $(date)"

echo "=============================================================================="
echo ">>> [4/9] STARTING: run_square_id_strong_unified_seed42.sh"
echo ">>> Path: resfit/shell_final/square/exp1_id/strong_200d/run_square_id_strong_unified_seed42.sh"
echo ">>> Time: $(date)"
echo "=============================================================================="
bash "${PROJECT_ROOT}/resfit/shell_final/square/exp1_id/strong_200d/run_square_id_strong_unified_seed42.sh"
echo ">>> [4/9] FINISHED: run_square_id_strong_unified_seed42.sh"
echo ">>> Time: $(date)"

echo "=============================================================================="
echo ">>> [5/9] STARTING: run_square_id_strong_unified_seed52.sh"
echo ">>> Path: resfit/shell_final/square/exp1_id/strong_200d/run_square_id_strong_unified_seed52.sh"
echo ">>> Time: $(date)"
echo "=============================================================================="
bash "${PROJECT_ROOT}/resfit/shell_final/square/exp1_id/strong_200d/run_square_id_strong_unified_seed52.sh"
echo ">>> [5/9] FINISHED: run_square_id_strong_unified_seed52.sh"
echo ">>> Time: $(date)"

echo "=============================================================================="
echo ">>> [6/9] STARTING: run_square_id_strong_unified_seed62.sh"
echo ">>> Path: resfit/shell_final/square/exp1_id/strong_200d/run_square_id_strong_unified_seed62.sh"
echo ">>> Time: $(date)"
echo "=============================================================================="
bash "${PROJECT_ROOT}/resfit/shell_final/square/exp1_id/strong_200d/run_square_id_strong_unified_seed62.sh"
echo ">>> [6/9] FINISHED: run_square_id_strong_unified_seed62.sh"
echo ">>> Time: $(date)"

echo "=============================================================================="
echo ">>> [7/9] STARTING: run_square_id_strong_decoupled_seed42.sh"
echo ">>> Path: resfit/shell_final/square/exp1_id/strong_200d/run_square_id_strong_decoupled_seed42.sh"
echo ">>> Time: $(date)"
echo "=============================================================================="
bash "${PROJECT_ROOT}/resfit/shell_final/square/exp1_id/strong_200d/run_square_id_strong_decoupled_seed42.sh"
echo ">>> [7/9] FINISHED: run_square_id_strong_decoupled_seed42.sh"
echo ">>> Time: $(date)"

echo "=============================================================================="
echo ">>> [8/9] STARTING: run_square_id_strong_decoupled_seed52.sh"
echo ">>> Path: resfit/shell_final/square/exp1_id/strong_200d/run_square_id_strong_decoupled_seed52.sh"
echo ">>> Time: $(date)"
echo "=============================================================================="
bash "${PROJECT_ROOT}/resfit/shell_final/square/exp1_id/strong_200d/run_square_id_strong_decoupled_seed52.sh"
echo ">>> [8/9] FINISHED: run_square_id_strong_decoupled_seed52.sh"
echo ">>> Time: $(date)"

echo "=============================================================================="
echo ">>> [9/9] STARTING: run_square_id_strong_decoupled_seed62.sh"
echo ">>> Path: resfit/shell_final/square/exp1_id/strong_200d/run_square_id_strong_decoupled_seed62.sh"
echo ">>> Time: $(date)"
echo "=============================================================================="
bash "${PROJECT_ROOT}/resfit/shell_final/square/exp1_id/strong_200d/run_square_id_strong_decoupled_seed62.sh"
echo ">>> [9/9] FINISHED: run_square_id_strong_decoupled_seed62.sh"
echo ">>> Time: $(date)"

echo "=============================================================================="
echo ">>> ALL 9 RUNS IN QUEUE COMPLETED SUCCESSFULLY!"
echo "Queue Finish Time: $(date)"
echo "=============================================================================="
