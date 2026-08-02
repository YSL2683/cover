import numpy as np
import imageio
import os
from omegaconf import OmegaConf
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from resfit.dexmg.environments.dexmg import create_vectorized_env

def main():
    # 1. Setup config for multi-disturbance and OOD environment
    config = OmegaConf.create({
        "ood_position": {
            "x_bounds": [-0.05, 0.05],
            "y_bounds": [-0.05, 0.05]
        },
        "disturbance": {
            "step_range": [0, 100], 
            "force_range": [250, 250],
            "num_disturbances": 1
        }
    })
    
    print("Creating vectorized environment...")
    # 2. Create Vectorized Environment
    env = create_vectorized_env(
        env_name="Lift",
        num_envs=1,
        debug=True,
        env_modifier_config=config,
        video_key="observation.images.frontview"
    )
    
    obs, info = env.reset()
    frames = []
    
    print("Running episode...")
    # 3. Step through the environment
    for step in range(80):
        # Base policy action (keep it still mostly, to clearly see disturbance)
        action = np.zeros((1, 7))
        obs, reward, terminated, truncated, info = env.step(action)
        
        frames.append(env.render()[0])
        
        if terminated[0] or truncated[0]:
            print(f"Episode ended at step {step}")
            break
            
    env.close()
    
    # 4. Save video
    save_path = 'disturbance_test.mp4'
    imageio.mimsave(save_path, frames, fps=10)
    print(f"Saved video to {os.path.abspath(save_path)}")

if __name__ == "__main__":
    main()
