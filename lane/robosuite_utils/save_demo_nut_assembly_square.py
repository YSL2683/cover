import numpy as np
import os
import torch
import imageio
import robosuite as suite
from robosuite import load_controller_config
from scipy.spatial.transform import Rotation as R

config = load_controller_config(default_controller="OSC_POSE")

NUM_DEMOS = 50
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
    stage_counter = 0
    success = False
    
    target_yaw_global = 0.0
    
    # We will interpolate pitch/roll from the initial values
    init_roll = None
    init_pitch = None
    
    for step in range(400):
        if step == 0:
            initial_R = R.from_quat(obs["robot0_eef_quat"])
            euler0 = initial_R.as_euler('xyz', degrees=False)
            init_roll, init_pitch = euler0[0], euler0[1]
            
        # Smoothly interpolate target roll/pitch to perfectly vertical over 50 steps
        alpha = min(1.0, step / 50.0)
        target_roll_final = np.pi if init_roll > 0 else -np.pi
        target_roll = init_roll + alpha * (target_roll_final - init_roll)
        target_pitch = init_pitch + alpha * (0.0 - init_pitch)
        demo_frames.append(obs["frontview_image"][::-1])
        nut_body_id = env.sim.model.body_name2id("SquareNut_main")
        peg_body_id = env.sim.model.body_name2id("peg1")
        
        handle_pos = env.sim.data.site_xpos[env.sim.model.site_name2id("SquareNut_handle_site")]
        center_pos = env.sim.data.body_xpos[env.sim.model.body_name2id("SquareNut_main")]
        peg_pos = env.sim.data.body_xpos[peg_body_id]
        
        gripper_pos = obs["robot0_eef_pos"]

        action = np.zeros(7)
        
        if stage == 0:
            vec = handle_pos - center_pos
            vec[2] = 0
            vec_dir = vec / (np.linalg.norm(vec) + 1e-6)
            
            hole_dir = center_pos - handle_pos
            # STRICTLY align the X axis to the hole direction so the camera sees the hole
            target_yaw_global = np.arctan2(hole_dir[1], hole_dir[0])
        
            gripper_quat_xyzw = obs["robot0_eef_quat"]
            gripper_yaw = R.from_quat(gripper_quat_xyzw).as_euler('xyz', degrees=False)[2]
            yaw_error = (target_yaw_global - gripper_yaw + np.pi) % (2 * np.pi) - np.pi
            
            # Target pos calculation moved above
            target_pos = handle_pos + vec_dir * 0.015
            target_pos[2] = 0.95
            
            action[:3] = target_pos - gripper_pos
            action[-1] = -1
            

            if np.linalg.norm(action[:3]) < 0.015 and abs(yaw_error) < 0.08:
                stage = 1
                
            # Slow down translation if yaw error is large so it aligns and approaches smoothly
            speed_factor = 1.0 / (1.0 + abs(yaw_error) * 3.0)
            action[:3] = np.clip(action[:3] * (5.0 * speed_factor), -0.4, 0.4)

        elif stage == 1:
            vec = handle_pos - center_pos
            vec[2] = 0
            vec_dir = vec / (np.linalg.norm(vec) + 1e-6)
            target_pos = handle_pos + vec_dir * 0.015
            # Go down to grasp
            target_pos[2] -= 0.01
            
            action[:3] = target_pos - gripper_pos
            action[-1] = -1
            
            if np.linalg.norm(action[:3]) < 0.015:
                stage = 2
            action[:3] = np.clip(action[:3] * 5.0, -0.4, 0.4)

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
            
            if stage_counter == 0:
                gripper_quat_xyzw = obs["robot0_eef_quat"]
                gripper_yaw = R.from_quat(gripper_quat_xyzw).as_euler('xyz', degrees=False)[2]
                
                v_peg = peg_pos[:2] - center_pos[:2]
                angle_peg = np.arctan2(v_peg[1], v_peg[0])
                
                candidates = [peg_yaw + k * np.pi/2 for k in range(4)]
                best_yaw = candidates[0]
                min_cost = 10000
                
                for c in candidates:
                    rel_angle = (c - angle_peg + np.pi) % (2 * np.pi) - np.pi
                    
                    # Forbid backwards alignment (handle pointing to peg)
                    # This leaves ONLY Front (abs(rel_angle) <= pi/4) and Side (pi/4 < abs(rel_angle) < 3*pi/4)
                    if abs(rel_angle) > np.pi * 0.75:
                        continue
                        
                    err = (c - gripper_yaw + np.pi) % (2 * np.pi) - np.pi
                    total_cost = abs(err)
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_yaw = c
                        
                target_yaw_global = best_yaw
            
            stage_counter += 1
            if np.linalg.norm(target_pos[:2] - center_pos[:2]) < 0.02 and abs((target_yaw_global - R.from_quat(obs["robot0_eef_quat"]).as_euler('xyz', degrees=False)[2] + np.pi) % (2 * np.pi) - np.pi) < 0.08:
                stage = 5
                stage_counter = 0

        elif stage == 5:
            # Lower nut until peg tip just engages the hole (~nut half-thickness above peg top)
            # midpoint between 0.06 and 0.12 -> release at peg_pos+0.09
            target_pos = peg_pos.copy()
            target_pos[2] += 0.09
            
            action[:3] = np.clip((target_pos - center_pos) * 3.0, -0.4, 0.4)
            action[-1] = 1
            
            # Uses target_yaw_global computed in stage 4
            
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
                
        R_target = R.from_euler('xyz', [target_roll, target_pitch, target_yaw_global])
        R_current = R.from_quat(obs["robot0_eef_quat"])
        rot_err = (R_target * R_current.inv()).as_rotvec()
        rot_action = rot_err * 5.0
        max_rot = np.max(np.abs(rot_action))
        if max_rot > 1.0:
            rot_action = rot_action / max_rot
        action[3:6] = rot_action
        action[:3] = np.clip(action[:3], -1.0, 1.0)

        state = np.concatenate([
            obs["robot0_eef_pos"],
            obs["robot0_eef_quat"],
            obs["robot0_gripper_qpos"]
        ])
        ep_states.append(state)

        next_obs, r, d, info = env.step(action)
        env.render()
        demo_frames.append(obs["frontview_image"][::-1])
        
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
            
        if d or step == 399:
            if not success:
                print(f"Episode failed at stage {stage}, step {step}! Retrying... (Attempts: {attempts})")
                pos_err = np.linalg.norm(target_pos - gripper_pos)
                print(f"Failed. Pos err: {pos_err:.4f}, Yaw err: {abs(yaw_error):.4f}")
                print(f"target_pos: {target_pos}, gripper_pos: {gripper_pos}")
                print(f"target_yaw: {target_yaw_global}, gripper_yaw: {gripper_yaw}")
            r_val = 100 if success else -1
            ep_rewards.append([r_val])
            ep_not_dones.append([not d])
            break
        
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
