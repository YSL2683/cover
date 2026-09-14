#!/usr/bin/env python3
"""
Compare Robosuite Square Residual RL Methods:
- square_unified (3 seeds: seed42, seed52, seed62)
- square_ours    (3 seeds: seed42, seed52, seed62)
- square_resfit  (3 seeds: seed42, seed52, seed62)

Plots evaluation success rate (eval/success_rate) vs environment steps.
Uses Seaborn: solid line for 3-seed mean, shaded region for ±1 standard deviation.
"""

import os
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualization.plot_wandb_success_rate import plot_wandb_runs

ENTITY = "ysl2683-seoul-national-university-ofscience-and-technology"
PROJECT = "square_residual_rl"

METHODS = {
    "square_unified": [
        "lv0vr121",  # seed62 (Square_reward_pbrs_unified_no_mask_nstep_beta1.0_scale0.1_seed62__2026-09-12_22-09-09...)
        "01ag9o17",  # seed52 (Square_reward_pbrs_unified_no_mask_nstep_beta1.0_scale0.1_seed52__2026-09-12_16-43-56...)
        "hw3am9mz",  # seed42 (Square_reward_pbrs_unified_no_mask_nstep_beta1.0_scale0.1__2026-09-08_12-15-49...)
    ],
    "square_ours": [
        "1naafe3f",  # seed42 (Square_reward_pbrs_no_mask_nstep_beta1.0_scale0.1__2026-09-02_11-33-44...)
        "ujvx4mzn",  # seed62 (Square_reward_pbrs_no_mask_nstep_beta1.0_scale0.1_seed62__2026-09-10_15-51-16...)
        "1ezbu0jq",  # seed52 (Square_reward_pbrs_no_mask_nstep_beta1.0_scale0.1_seed52__2026-09-10_01-41-01...)
    ],
    "square_resfit": [
        "32wrjn3v",  # seed62 (Square_ID_resfit_seed62__2026-09-11_13-58-20...)
        "whv3dtmy",  # seed52 (Square_ID_resfit_seed52__2026-09-10_14-34-21...)
        "ri6hp3in",  # seed42 (Square_ID_resfit_seed42__2026-08-31_16-01-18...)
    ],
}

def main():
    output_dir = Path(__file__).resolve().parent
    brain_dir = Path("/home/ysl2683/.gemini/antigravity-cli/brain/29de231d-dbed-4f21-8374-6037be748a6e")

    print("Generating Min-Max shaded plots...")
    # 1. Min-Max Shaded (Raw)
    minmax_png = output_dir / "square_success_rate_comparison_minmax.png"
    minmax_pdf = output_dir / "square_success_rate_comparison_minmax.pdf"
    plot_wandb_runs(
        project=PROJECT,
        entity=ENTITY,
        methods_dict=METHODS,
        metric_key="eval/success_rate",
        output_path=str(minmax_png),
        title="Robosuite Square: Evaluation Success Rate (3 Seeds, Mean & Min-Max Range)",
        xlabel="Environment Steps",
        ylabel="Success Rate (%)",
        as_percentage=True,
        smooth=1.0,
        max_step=300000,
        errorbar="minmax",
        figsize=(9.0, 5.5),
    )
    plot_wandb_runs(
        project=PROJECT,
        entity=ENTITY,
        methods_dict=METHODS,
        metric_key="eval/success_rate",
        output_path=str(minmax_pdf),
        title="Robosuite Square: Evaluation Success Rate (3 Seeds, Mean & Min-Max Range)",
        xlabel="Environment Steps",
        ylabel="Success Rate (%)",
        as_percentage=True,
        smooth=1.0,
        max_step=300000,
        errorbar="minmax",
        figsize=(9.0, 5.5),
    )
    if brain_dir.exists():
        shutil.copy(str(minmax_png), str(brain_dir / "square_success_rate_comparison_minmax.png"))

    # 2. Min-Max Shaded (EMA Smoothed alpha=0.6)
    minmax_smooth_png = output_dir / "square_success_rate_comparison_minmax_smoothed.png"
    plot_wandb_runs(
        project=PROJECT,
        entity=ENTITY,
        methods_dict=METHODS,
        metric_key="eval/success_rate",
        output_path=str(minmax_smooth_png),
        title="Robosuite Square: Evaluation Success Rate (3 Seeds, Mean & Min-Max Range, EMA Smoothed)",
        xlabel="Environment Steps",
        ylabel="Success Rate (%)",
        as_percentage=True,
        smooth=0.6,
        max_step=300000,
        errorbar="minmax",
        figsize=(9.0, 5.5),
    )
    if brain_dir.exists():
        shutil.copy(str(minmax_smooth_png), str(brain_dir / "square_success_rate_comparison_minmax_smoothed.png"))

    print("All Min-Max plots generated successfully!")

if __name__ == "__main__":
    main()
