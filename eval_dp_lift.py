import argparse
import torch
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
from resfit.lerobot.utils.load_policy import load_policy
from pathlib import Path
import imageio

def eval_policy(policy_path, n_episodes=20, max_steps=300, video_path="eval_video.mp4", cube_range=0.025):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading policy from {policy_path}...")
    print(f"Evaluating with cube range [-{cube_range}, {cube_range}] ({(cube_range*2*100):.1f}cm x {(cube_range*2*100):.1f}cm)")
    policy = load_policy(Path(policy_path))
    policy.eval()
    policy.to(device)
    
    config = load_controller_config(default_controller="OSC_POSE")
    
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=config,
        has_renderer=False,
        has_offscreen_renderer=True,
        control_freq=20,
        horizon=max_steps,
        use_object_obs=True,
        use_camera_obs=True,
        camera_names=["frontview", "robot0_eye_in_hand", "agentview"],
        camera_heights=[128, 128, 512],
        camera_widths=[128, 128, 512],
        reward_shaping=True,
    )
    
    successes = 0
    all_frames = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        env.sim.forward()
        
        # Override initial position
        cube_joint = env.cube.joints[0]
        cube_x = np.random.uniform(-cube_range, cube_range)
        cube_y = np.random.uniform(-cube_range, cube_range)
        
        qpos = env.sim.data.get_joint_qpos(cube_joint)
        qpos[0] = cube_x
        qpos[1] = cube_y
        env.sim.data.set_joint_qpos(cube_joint, qpos)
        env.sim.forward()
        obs = env._get_observations(force_update=True)
        
        policy.reset()
        ep_success = False
        
        # Record video of all episodes to see what's happening
        record_video = True
        success_counter = 0
        
        ep_frames = []
        for step in range(max_steps):
            if record_video:
                ep_frames.append(obs["agentview_image"][::-1])
                
            front_img = obs["frontview_image"][::-1].copy()
            wrist_img = obs["robot0_eye_in_hand_image"][::-1].copy()
            
            # Prepare inputs to lerobot policy
            # LeRobot expects (C, H, W) float32 in [0, 1]
            front_t = torch.from_numpy(front_img.transpose((2, 0, 1))).float() / 255.0
            wrist_t = torch.from_numpy(wrist_img.transpose((2, 0, 1))).float() / 255.0
            
            state = np.concatenate([
                obs["robot0_eef_pos"],
                obs["robot0_eef_quat"],
                obs["robot0_gripper_qpos"]
            ])
            state_t = torch.from_numpy(state).float()
            
            obs_dict = {
                "observation.images.frontview": front_t.unsqueeze(0).to(device),
                "observation.images.robot0_eye_in_hand": wrist_t.unsqueeze(0).to(device),
                "observation.state": state_t.unsqueeze(0).to(device)
            }
            
            with torch.no_grad():
                action = policy.select_action(obs_dict)
                action = action.squeeze(0).cpu().numpy()
                
            next_obs, r, d, info = env.step(action)
            
            if env._check_success():
                success_counter += 1
                if success_counter >= 5:
                    ep_success = True
                    break
            else:
                success_counter = 0
                
            obs = next_obs
            
            if d:
                break
                
        if ep_success:
            successes += 1
            print(f"Episode {ep+1}: SUCCESS")
        else:
            print(f"Episode {ep+1}: FAILURE")
            if record_video:
                all_frames.extend(ep_frames)
            
    success_rate = successes / n_episodes
    print(f"Total Success Rate: {success_rate * 100:.1f}% ({successes}/{n_episodes})")
    
    if all_frames:
        imageio.mimsave(video_path, all_frames, fps=10)
        print(f"Saved evaluation video to {video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--video_path", type=str, default="eval_video.mp4")
    parser.add_argument("--cube_range", type=float, default=0.025)
    args = parser.parse_args()
    eval_policy(args.policy_path, args.n_episodes, video_path=args.video_path, cube_range=args.cube_range)
