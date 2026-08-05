import numpy as np
import os
import torch
import imageio
import robosuite as suite
from robosuite import load_controller_config
from scipy.spatial.transform import Rotation as R

config = load_controller_config(default_controller="OSC_POSE")

NUM_DEMOS = 20
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_FOLDER = os.path.abspath(os.path.join(SCRIPT_DIR, "../demo/robosuite_nut_assembly_square/")) + "/"
target_folder = ROOT_FOLDER + str(NUM_DEMOS)
if not os.path.isdir(target_folder):
    os.makedirs(target_folder)

env = suite.make(
    env_name="NutAssemblySquare",
    robots="Panda",
    controller_configs=config,
    camera_names=["frontview", "robot0_eye_in_hand"],
    camera_heights=128,
    camera_widths=128,
    control_freq=10,
    horizon=300,
    has_renderer=True,
    has_offscreen_renderer=True,
    render_camera="frontview",
)

# Set SquareNut initial placement range (x uses default [-0.115, -0.11], y uses equivalent width [0.11, 0.115])
if hasattr(env, "placement_initializer") and hasattr(env.placement_initializer, "samplers"):
    if "SquareNutSampler" in env.placement_initializer.samplers:
        env.placement_initializer.samplers["SquareNutSampler"].x_range = [-0.115, -0.11]
        env.placement_initializer.samplers["SquareNutSampler"].y_range = [0.11, 0.115]

obs_list = []
next_obs_list = []
action_list = []
reward_list = []
not_done_list = []
state_list = []

demo_starts = []
demo_ends = []

successful_demos = 0
attempts = 0

while successful_demos < NUM_DEMOS:
    obs = env.reset()
    attempts += 1
    
    ep_obs, ep_next_obs, ep_actions, ep_rewards, ep_not_dones, ep_states = [], [], [], [], [], []
    demo_frames = []
    
    img_obs = np.concatenate(
        [obs["frontview_image"][::-1], obs["robot0_eye_in_hand_image"][::-1]], axis=2
    ).transpose((2, 0, 1))
    
    stage = 0
    prev_stage = -1
    stage_counter = 0
    success = False
    
    for step in range(300):
        demo_frames.append(obs["frontview_image"][::-1])
        nut_body_id = env.sim.model.body_name2id("SquareNut_main")
        peg_body_id = env.sim.model.body_name2id("peg1")
        
        handle_pos = env.sim.data.site_xpos[env.sim.model.site_name2id("SquareNut_handle_site")]
        center_pos = env.sim.data.site_xpos[env.sim.model.site_name2id("SquareNut_center_site")]
        peg_pos = env.sim.data.body_xpos[peg_body_id]
        
        gripper_pos = np.array(
            env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
        )

        action = np.zeros(7)

        if stage == 0:
            target_pos = handle_pos.copy()
            target_pos[2] += 0.05
            action[:3] = target_pos - gripper_pos
            action[-1] = -1
            
            nut_quat = env.sim.data.body_xquat[nut_body_id] # [w, x, y, z]
            nut_quat_xyzw = np.array([nut_quat[1], nut_quat[2], nut_quat[3], nut_quat[0]])
            nut_yaw = R.from_quat(nut_quat_xyzw).as_euler('xyz', degrees=False)[2]
            
            gripper_quat_xyzw = obs["robot0_eef_quat"]
            gripper_yaw = R.from_quat(gripper_quat_xyzw).as_euler('xyz', degrees=False)[2]
            
            yaw_error = nut_yaw - gripper_yaw
            yaw_error = (yaw_error + np.pi/2) % np.pi - np.pi/2
            action[5] = yaw_error * 5.0
            
            if np.linalg.norm(action[:3]) < 0.01 and abs(yaw_error) < 0.05:
                stage = 1
            action[:3] *= 5.0

        elif stage == 1:
            target_pos = handle_pos.copy()
            action[:3] = target_pos - gripper_pos
            action[-1] = -1
            
            nut_quat = env.sim.data.body_xquat[nut_body_id]
            nut_quat_xyzw = np.array([nut_quat[1], nut_quat[2], nut_quat[3], nut_quat[0]])
            nut_yaw = R.from_quat(nut_quat_xyzw).as_euler('xyz', degrees=False)[2]
            gripper_quat_xyzw = obs["robot0_eef_quat"]
            gripper_yaw = R.from_quat(gripper_quat_xyzw).as_euler('xyz', degrees=False)[2]
            
            yaw_error = nut_yaw - gripper_yaw
            yaw_error = (yaw_error + np.pi/2) % np.pi - np.pi/2
            action[5] = yaw_error * 5.0
            
            if np.linalg.norm(action[:3]) < 0.005:
                stage = 2
            action[:3] *= 5.0

        elif stage == 2:
            action[:] = 0
            action[-1] = 1
            stage_counter += 1
            if stage_counter == 8:
                stage = 3
                stage_counter = 0

        elif stage == 3:
            action[:] = 0
            action[2] = 0.25
            action[-1] = 1
            stage_counter += 1
            if stage_counter >= 15:
                stage = 4
                stage_counter = 0

        elif stage == 4:
            # move nut center to target_pos smoothly
            target_pos = peg_pos.copy()
            target_pos[2] += 0.15 
            
            action[:3] = np.clip((target_pos - center_pos) * 3.0, -0.4, 0.4)
            action[-1] = 1
            
            peg_quat = env.sim.data.body_xquat[peg_body_id]
            peg_quat_xyzw = np.array([peg_quat[1], peg_quat[2], peg_quat[3], peg_quat[0]])
            peg_yaw = R.from_quat(peg_quat_xyzw).as_euler('xyz', degrees=False)[2]
            gripper_quat_xyzw = obs["robot0_eef_quat"]
            gripper_yaw = R.from_quat(gripper_quat_xyzw).as_euler('xyz', degrees=False)[2]
            yaw_error = peg_yaw - gripper_yaw
            yaw_error = (yaw_error + np.pi/2) % np.pi - np.pi/2
            action[5] = yaw_error * 5.0
            
            stage_counter += 1
            if np.linalg.norm(target_pos - center_pos) < 0.01:
                stage = 5
                stage_counter = 0

        elif stage == 5:
            # Lower nut until peg tip just engages the hole (~nut half-thickness above peg top)
            # midpoint between 0.06 and 0.12 -> release at peg_pos+0.09
            target_pos = peg_pos.copy()
            target_pos[2] += 0.09
            
            action[:3] = np.clip((target_pos - center_pos) * 3.0, -0.4, 0.4)
            action[-1] = 1
            
            peg_quat = env.sim.data.body_xquat[peg_body_id]
            peg_quat_xyzw = np.array([peg_quat[1], peg_quat[2], peg_quat[3], peg_quat[0]])
            peg_yaw = R.from_quat(peg_quat_xyzw).as_euler('xyz', degrees=False)[2]
            gripper_quat_xyzw = obs["robot0_eef_quat"]
            gripper_yaw = R.from_quat(gripper_quat_xyzw).as_euler('xyz', degrees=False)[2]
            yaw_error = peg_yaw - gripper_yaw
            yaw_error = (yaw_error + np.pi/2) % np.pi - np.pi/2
            action[5] = yaw_error * 5.0
            
            stage_counter += 1
            if np.linalg.norm(target_pos - center_pos) < 0.01:
                stage = 6
                stage_counter = 0
            
        elif stage == 6:
            action[:] = 0
            action[-1] = -1
            stage_counter += 1
            if stage_counter == 5:
                stage = 7
                stage_counter = 0
                
        elif stage == 7:
            action[:] = 0
            action[2] = 0.25
            action[-1] = -1
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
        
        if r == 1 or r == 100 or env._check_success():
            d = True
            success = True
            
        r_val = 100 if success else -1
        ep_rewards.append([r_val])
        ep_not_dones.append([not d])
        
        img_obs = next_img_obs
        obs = next_obs

        if d or success:
            break
            
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
            imageio.mimsave(target_folder + f"/demo_{successful_demos}.mp4", demo_frames, fps=10)
            
        successful_demos += 1
        print(f"Collected successful demo {successful_demos}/{NUM_DEMOS} (Attempts: {attempts})")
    else:
        print(f"Episode failed! Retrying... (Attempts: {attempts})")

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
