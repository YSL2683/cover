import numpy as np
import os
import torch
import h5py
import robosuite as suite
from robosuite import load_controller_config
from tqdm import tqdm

def main():
    hdf5_path = "/home/moai/ysl_ws/cover/lane/demo/demo_v15.hdf5"
    
    # Target folder (as specified: @[lane/demo/robosuite_nut_assembly_square] + number of demos)
    num_demos = 0
    with h5py.File(hdf5_path, "r") as f:
        num_demos = len(f["data"].keys())
        
    target_folder = f"/home/moai/ysl_ws/cover/lane/demo/robosuite_nut_assembly_square/{num_demos}"
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)

    config = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name="NutAssemblySquare",
        robots="Panda",
        controller_configs=config,
        camera_names=["frontview", "robot0_eye_in_hand"],
        camera_heights=128,
        camera_widths=128,
        control_freq=10,
        horizon=300,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
    )
    
    # Hide all sites (nut center, gripper visualizers, green lines, etc.)
    env.sim.model.site_rgba[:, 3] = 0.0

    f = h5py.File(hdf5_path, "r")
    demos = list(f["data"].keys())
    # Sort demos numerically
    demos = sorted(demos, key=lambda x: int(x.split("_")[1]))

    obs_list = []
    next_obs_list = []
    action_list = []
    reward_list = []
    not_done_list = []
    state_list = []

    demo_starts = []
    demo_ends = []

    for idx, demo in enumerate(tqdm(demos, desc="Rendering demos")):
        states = f["data"][demo]["states"][:]
        actions = f["data"][demo]["actions"][:]
        
        ep_obs, ep_next_obs, ep_actions, ep_rewards, ep_not_dones, ep_states = [], [], [], [], [], []
        
        # Downsample to 10Hz (take every 2nd frame)
        indices = list(range(0, len(states), 2))
        
        rendered_images = []
        is_success = []
        
        # Render the selected frames
        for i in indices:
            env.sim.set_state_from_flattened(states[i])
            env.sim.forward()
            obs = env._get_observations(force_update=True)
            
            img_obs = np.concatenate(
                [obs["frontview_image"][::-1], obs["robot0_eye_in_hand_image"][::-1]], axis=2
            ).transpose((2, 0, 1))
            
            rob_state = np.concatenate([
                obs["robot0_eef_pos"],
                obs["robot0_eef_quat"],
                obs["robot0_gripper_qpos"]
            ])
            
            rendered_images.append(img_obs)
            ep_states.append(rob_state)
            is_success.append(env._check_success())
            
        # Get one more frame for the final next_obs
        last_idx = indices[-1] + 2
        if last_idx >= len(states):
            last_idx = len(states) - 1
            
        env.sim.set_state_from_flattened(states[last_idx])
        env.sim.forward()
        obs = env._get_observations(force_update=True)
        img_obs = np.concatenate(
            [obs["frontview_image"][::-1], obs["robot0_eye_in_hand_image"][::-1]], axis=2
        ).transpose((2, 0, 1))
        rendered_images.append(img_obs)
        final_succ = env._check_success()
        
        # Construct transitions
        for i in range(len(indices)):
            ep_obs.append(rendered_images[i])
            ep_next_obs.append(rendered_images[i+1])
            
            orig_idx = indices[i]
            # Aggregate actions for downsampling: sum translations/rotations, take last gripper action
            if orig_idx + 1 < len(actions):
                a1 = actions[orig_idx]
                a2 = actions[orig_idx+1]
                agg_action = np.zeros(7)
                agg_action[:6] = a1[:6] + a2[:6]
                agg_action[6] = a2[6]
            else:
                agg_action = actions[orig_idx]
                
            ep_actions.append(agg_action)
            
            next_succ = is_success[i+1] if (i + 1 < len(is_success)) else final_succ
            
            done = next_succ
            reward = 100.0 if next_succ else -1.0
            
            ep_rewards.append([reward])
            ep_not_dones.append([not done])
            
        demo_starts.append(len(obs_list))
        obs_list.extend(ep_obs)
        next_obs_list.extend(ep_next_obs)
        action_list.extend(ep_actions)
        reward_list.extend(ep_rewards)
        not_done_list.extend(ep_not_dones)
        state_list.extend(ep_states)
        demo_ends.append(len(obs_list))

    print("Saving dataset...")
    payload = [
        np.array(obs_list, dtype=np.uint8),
        np.array(next_obs_list, dtype=np.uint8),
        np.array(action_list, dtype=np.float32),
        np.array(reward_list, dtype=np.float32),
        np.array(not_done_list, dtype=bool),
        np.array(state_list, dtype=np.float32),
    ]
    
    file_path = f"{target_folder}/0_{len(obs_list)}.pt"
    torch.save(payload, file_path)
    np.save(f"{target_folder}/demo_starts.npy", np.array(demo_starts))
    np.save(f"{target_folder}/demo_ends.npy", np.array(demo_ends))
    
    print(f"Successfully saved {len(demos)} demos ({len(obs_list)} frames) to {target_folder}")

if __name__ == '__main__':
    main()
