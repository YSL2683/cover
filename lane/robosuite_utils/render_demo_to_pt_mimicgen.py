import argparse
import os
import sys
import h5py
import numpy as np
import torch
from tqdm import tqdm

import robosuite as suite
from robosuite import load_controller_config
import mimicgen
from mimicgen.envs.robosuite.coffee import Coffee_D0
from mimicgen.envs.robosuite.mug_cleanup import MugCleanup_D0
from mimicgen.envs.robosuite.three_piece_assembly import ThreePieceAssembly_D0

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TASK_CONFIGS = {
    "coffee": {
        "env_name": "Coffee_D0",
        "horizon": 400,
        "default_demos": 50,
    },
    "mug_cleanup": {
        "env_name": "MugCleanup_D0",
        "horizon": 500,
        "default_demos": 50,
    },
    "three_piece_assembly": {
        "env_name": "ThreePieceAssembly_D0",
        "horizon": 500,
        "default_demos": 100,
    },
}


def render_task_demos(task_name, num_demos=None, hdf5_path=None, output_dir=None, camera_size=128):
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_CONFIGS.keys())}")
    
    cfg = TASK_CONFIGS[task_name]
    env_name = cfg["env_name"]
    horizon = cfg["horizon"]
    if num_demos is None:
        num_demos = cfg["default_demos"]
        
    if hdf5_path is None:
        hdf5_path = os.path.join(PROJECT_ROOT, f"lane/demo/mimicgen_dataset/{task_name}/demo_d0.hdf5")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, f"lane/demo/mimicgen_{task_name}/{num_demos}")
        
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f"HDF5 dataset not found at {hdf5_path}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Rendering {task_name} ({env_name}) ===")
    print(f"Source HDF5: {hdf5_path}")
    print(f"Target dir : {output_dir}")
    print(f"Num demos  : {num_demos}")
    print(f"Resolution : {camera_size}x{camera_size}")

    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller_config,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=camera_size,
        camera_widths=camera_size,
        control_freq=20,
        horizon=horizon,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
    )
    
    # Hide all sites to avoid visual artifacts in observation images
    if hasattr(env.sim.model, "site_rgba"):
        env.sim.model.site_rgba[:, 3] = 0.0

    f = h5py.File(hdf5_path, "r")
    demos = list(f["data"].keys())
    # Sort demos numerically (e.g., demo_0, demo_1, ...)
    demos = sorted(demos, key=lambda x: int(x.split("_")[1]))[:num_demos]
    actual_demos = len(demos)
    print(f"Processing {actual_demos} demonstrations...")

    obs_list = []
    next_obs_list = []
    action_list = []
    reward_list = []
    not_done_list = []
    state_list = []

    demo_starts = []
    demo_ends = []

    for idx, demo in enumerate(tqdm(demos, desc=f"Rendering {task_name}")):
        states = f["data"][demo]["states"][:]
        actions = f["data"][demo]["actions"][:]
        
        # Reset environment with specific model_file if present
        if "model_file" in f[f"data/{demo}"].attrs:
            xml_str = f[f"data/{demo}"].attrs["model_file"]
            if hasattr(env, "edit_model_xml"):
                xml_str = env.edit_model_xml(xml_str)
            try:
                env.reset_from_xml_string(xml_str)
            except Exception as e:
                print(f"Warning: reset_from_xml_string failed ({e}), falling back to env.reset()")
                env.reset()
        else:
            env.reset()

        ep_obs = []
        ep_next_obs = []
        ep_actions = []
        ep_rewards = []
        ep_not_dones = []
        ep_states = []

        rendered_images = []
        is_success = []

        # Render each frame in trajectory
        for i in range(len(states)):
            env.sim.reset()
            env.sim.set_state_from_flattened(states[i])
            env.sim.forward()
            obs = env._get_observations(force_update=True)

            # Invert vertically to align MuJoCo coordinate convention with OpenCV/torch
            img_obs = np.concatenate(
                [obs["agentview_image"][::-1], obs["robot0_eye_in_hand_image"][::-1]], axis=2
            ).transpose((2, 0, 1))

            rob_state = np.concatenate([
                obs["robot0_eef_pos"],
                obs["robot0_eef_quat"],
                obs["robot0_gripper_qpos"]
            ])

            rendered_images.append(img_obs)
            ep_states.append(rob_state)
            is_success.append(env._check_success())

        # Render one additional frame for the final next_obs
        last_idx = len(states) - 1
        env.sim.reset()
        env.sim.set_state_from_flattened(states[last_idx])
        env.sim.forward()
        obs = env._get_observations(force_update=True)
        img_obs = np.concatenate(
            [obs["agentview_image"][::-1], obs["robot0_eye_in_hand_image"][::-1]], axis=2
        ).transpose((2, 0, 1))
        rendered_images.append(img_obs)
        final_succ = env._check_success()

        # Construct transitions
        for i in range(len(states)):
            ep_obs.append(rendered_images[i])
            ep_next_obs.append(rendered_images[i + 1])
            ep_actions.append(actions[i])

            next_succ = is_success[i + 1] if (i + 1 < len(is_success)) else final_succ
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

    f.close()
    env.close()

    print("Packaging and saving dataset...")
    payload = [
        np.array(obs_list, dtype=np.uint8),
        np.array(next_obs_list, dtype=np.uint8),
        np.array(action_list, dtype=np.float32),
        np.array(reward_list, dtype=np.float32),
        np.array(not_done_list, dtype=bool),
        np.array(state_list, dtype=np.float32),
    ]

    pt_file_path = os.path.join(output_dir, f"0_{len(obs_list)}.pt")
    import pickle
    torch.save(payload, pt_file_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(output_dir, "demo_starts.npy"), np.array(demo_starts))
    np.save(os.path.join(output_dir, "demo_ends.npy"), np.array(demo_ends))

    print(f"Successfully saved {actual_demos} demos ({len(obs_list)} frames) to {output_dir}")
    print(f"Payload file: {pt_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render MimicGen HDF5 dataset to LaNE/E2C .pt format")
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_CONFIGS.keys()), help="Task name")
    parser.add_argument("--num_demos", type=int, default=None, help="Number of demos to render")
    parser.add_argument("--hdf5_path", type=str, default=None, help="Path to input HDF5 file")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to output directory")
    parser.add_argument("--camera_size", type=int, default=128, help="Camera resolution (square)")
    args = parser.parse_args()

    render_task_demos(
        task_name=args.task,
        num_demos=args.num_demos,
        hdf5_path=args.hdf5_path,
        output_dir=args.output_dir,
        camera_size=args.camera_size,
    )
