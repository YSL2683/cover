import numpy as np
import os
import torch
import robosuite as suite
from robosuite import load_controller_config
import robosuite.utils.transform_utils as T

config = load_controller_config(default_controller="OSC_POSE")

NUM_DEMOS = 20
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_FOLDER = os.path.abspath(os.path.join(SCRIPT_DIR, "../demo/robosuite_lift/")) + "/"
target_folder = ROOT_FOLDER + str(NUM_DEMOS)
if not os.path.isdir(target_folder):
    os.makedirs(target_folder)

env = suite.make(
    env_name="Lift",
    robots="Panda",
    controller_configs=config,
    camera_names=["frontview", "robot0_eye_in_hand"],
    camera_heights=128,
    camera_widths=128,
    control_freq=10,
    horizon=40,
    has_renderer=True,
    has_offscreen_renderer=True,
    render_camera="frontview",
)

obs_list = []
next_obs_list = []
action_list = []
reward_list = []
not_done_list = []
state_list = []

stage = 0
stage_counter = 0

demo_starts = []
demo_ends = []

successful_demos = 0
while successful_demos < NUM_DEMOS:
    obs = env.reset()
    
    # Manually place the cube exactly within 5cm x 5cm area at center (0, 0)
    new_x = np.random.uniform(-0.025, 0.025)
    new_y = np.random.uniform(-0.025, 0.025)
    
    qpos = env.sim.data.get_joint_qpos(env.cube.joints[0])
    qpos[0] = new_x
    qpos[1] = new_y
    env.sim.data.set_joint_qpos(env.cube.joints[0], qpos)
    
    env.sim.forward()
    obs = env._get_observations(force_update=True)

    ep_obs, ep_next_obs, ep_actions, ep_rewards, ep_not_dones, ep_states = [], [], [], [], [], []
    demo_frames = []
    
    img_obs = np.concatenate(
        [obs["frontview_image"][::-1], obs["robot0_eye_in_hand_image"][::-1]], axis=2
    ).transpose((2, 0, 1))
    
    stage = 0
    stage_counter = 0
    success = False
    
    while True:
        demo_frames.append(obs["frontview_image"][::-1])
        cube_pos = env.sim.data.body_xpos[env.cube_body_id]
        gripper_pos = np.array(
            env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
        )

        action = np.zeros(7)

        if stage == 0:
            action[:3] = cube_pos - gripper_pos
            action[-1] = -1
            
            # Align Gripper Yaw to Cube Yaw
            from scipy.spatial.transform import Rotation as R
            cube_quat = env.sim.data.body_xquat[env.cube_body_id] # [w, x, y, z]
            cube_quat_xyzw = np.array([cube_quat[1], cube_quat[2], cube_quat[3], cube_quat[0]])
            cube_yaw = R.from_quat(cube_quat_xyzw).as_euler('xyz', degrees=False)[2]
            
            gripper_quat_xyzw = obs["robot0_eef_quat"]
            gripper_yaw = R.from_quat(gripper_quat_xyzw).as_euler('xyz', degrees=False)[2]
            
            yaw_error = cube_yaw - gripper_yaw
            # Cube is symmetric every 90 degrees (pi/2)
            yaw_error = (yaw_error + np.pi/4) % (np.pi/2) - np.pi/4
            
            # OSC_POSE uses action[3:6] as angular velocity (axis-angle)
            action[5] = yaw_error * 5.0 # P-control for yaw alignment
            
            if (action[:3] ** 2).sum() < 0.0001 and abs(yaw_error) < 0.05:
                stage = 1
            action[:3] *= 10

        if stage == 1:
            action[:] = 0
            action[-1] = 1
            stage_counter += 1
            if stage_counter == 3:
                stage = 2
                stage_counter = 0

        if stage == 2:
            action[:] = 0
            action[2] = 0.25
            action[-1] = 1
            stage_counter += 1
            if stage_counter >= 10:
                action[2] = 0
        state = np.concatenate([
            obs["robot0_eef_pos"],
            obs["robot0_eef_quat"],
            obs["robot0_gripper_qpos"]
        ])
        ep_states.append(state)

        next_obs, r, d, info = env.step(action)
        env.render()
        
        next_img_obs = np.concatenate(
            [
                next_obs["frontview_image"][::-1],
                next_obs["robot0_eye_in_hand_image"][::-1],
            ],
            axis=2,
        ).transpose((2, 0, 1))
        
        ep_obs.append(img_obs)
        ep_next_obs.append(next_img_obs)
        ep_actions.append(action)
        
        r = -1 if r <= 0 else 100
        if r == 100:
            d = True
            success = True
            
        ep_rewards.append([r])
        ep_not_dones.append([not d])
        
        img_obs = next_img_obs
        obs = next_obs

        if d:
            if success:
                demo_starts.append(len(obs_list))
                obs_list.extend(ep_obs)
                next_obs_list.extend(ep_next_obs)
                action_list.extend(ep_actions)
                reward_list.extend(ep_rewards)
                not_done_list.extend(ep_not_dones)
                state_list.extend(ep_states)
                demo_ends.append(len(obs_list))
                
                if successful_demos < 5:
                    if not os.path.isdir(target_folder):
                        os.makedirs(target_folder)
                    import imageio
                    imageio.mimsave(target_folder + f"/demo_{successful_demos}.mp4", demo_frames, fps=10)
                    
                successful_demos += 1
                print(f"Collected successful demo {successful_demos}/{NUM_DEMOS}")
            else:
                print("Episode failed! Retrying...")
            break

payload = [
    np.array(obs_list),
    np.array(next_obs_list),
    np.array(action_list),
    np.array(reward_list),
    np.array(not_done_list),
    np.array(state_list),
]
torch.save(payload, target_folder + "/0_" + str(len(obs_list)) + ".pt")
np.save(target_folder + "/demo_starts.npy", np.array(demo_starts))
np.save(target_folder + "/demo_ends.npy", np.array(demo_ends))

env.close()
