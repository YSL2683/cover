import os
import glob
import re
import argparse
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import json

from resfit.lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

import sys
sys.path.append(os.getcwd())

from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.lerobot.scripts.train_bc_dexmg import _run_rollouts
import torch

def main():
    checkpoints_dir = "resfit/my_lerobot_data/bc_run_2026-08-29_14-38-11_robomimic_square_v15_50_diffusion"
    
    # Find all policy_step_* folders
    steps_folders = []
    for d in os.listdir(checkpoints_dir):
        if d.startswith("policy_step_"):
            step = int(d.replace("policy_step_", ""))
            steps_folders.append((step, os.path.join(checkpoints_dir, d)))
    
    steps_folders.sort(key=lambda x: x[0])
    
    if not steps_folders:
        logger.error(f"No policy_step_* folders found in {checkpoints_dir}")
        return

    eval_env_name = "Square"
    eval_camera_size = 128
    eval_render_size = 128
    eval_num_episodes = 100
    eval_num_envs = 16
    eval_video_key = "observation.images.frontview"
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Creating evaluation environment: {eval_env_name}")
    eval_env = create_vectorized_env(
        env_name=eval_env_name,
        num_envs=eval_num_envs,
        device=device_str,
        camera_size=eval_camera_size,
        render_size=eval_render_size,
        video_key=eval_video_key,
        debug=False,
    )
    
    results = {}
    
    output_dir = Path("eval_results_diffusion_square")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "success_rates.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            try:
                results = json.load(f)
                results = {int(k): v for k, v in results.items()}
            except:
                results = {}
    
    run_start_time = "eval_run"

    for step, ckpt_dir in steps_folders:
        if step in results:
            logger.info(f"Step {step} already evaluated: {results[step]*100:.1f}%")
            continue
            
        logger.info(f"Evaluating step {step} from {ckpt_dir}...")
        policy_dir = Path(ckpt_dir) / "policy"
        if not policy_dir.exists():
            policy_dir = Path(ckpt_dir)
            
        try:
            policy = DiffusionPolicy.from_pretrained(str(policy_dir))
            policy.to(device_str)
            policy.eval()
            
            success_rate, _, _ = _run_rollouts(
                policy=policy,
                env=eval_env,
                save_dir=output_dir,
                step=step,
                num_episodes=eval_num_episodes,
                run_start_time=run_start_time,
            )
            logger.info(f"Step {step} success rate: {success_rate * 100:.1f}%")
            results[step] = float(success_rate)
            
            with open(results_file, "w") as f:
                json.dump(results, f, indent=4)
                
        except Exception as e:
            logger.error(f"Failed to evaluate step {step}: {e}")
            
    print("====================================")
    print("Final Evaluation Results:")
    for step in sorted(results.keys()):
        print(f"Step {step}: {results[step]*100:.1f}%")
    print("====================================")
    
    with open(output_dir / "results_table.md", "w") as f:
        f.write("| Step | Success Rate |\n")
        f.write("|------|--------------|\n")
        for step in sorted(results.keys()):
            f.write(f"| {step} | {results[step]*100:.1f}% |\n")

if __name__ == "__main__":
    main()
