import numpy as np
import os
import imageio
import torch
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler

config = load_controller_config(default_controller="OSC_POSE")

NUM_DEMOS = 20
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_FOLDER = os.path.abspath(os.path.join(SCRIPT_DIR, "../demo/robosuite_pick_place_can/")) + "/"
target_folder = ROOT_FOLDER + str(NUM_DEMOS)

reset_sampler = UniformRandomSampler(
    name="ObjectSampler",
    mujoco_objects=None,
    x_range=[-0.05, 0.05],
    y_range=[-0.05, 0.05],
    rotation=None,
    ensure_object_boundary_in_range=False,
    ensure_valid_placement=True,
    reference_pos=np.array((0, 0, 0.8)),
    z_offset=0.01,
)

env = suite.make(
    env_name="PickPlace",
    robots="Panda",
    controller_configs=config,
    camera_names=["frontview", "robot0_eye_in_hand"],
    camera_heights=128,
    camera_widths=128,
    control_freq=10,
    horizon=120,
    single_object_mode=2,
    object_type="can",
    has_renderer=True,
    has_offscreen_renderer=True,
    render_camera="frontview",
    # bin1_pos=(0.1, -0.27, 0.8),
    # bin2_pos=(0.1, 0.27, 0.8),
)

# (Placement initializer modification removed because it is reset by the environment internally)

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
    
    # Manually place the Can exactly within 3cm x 3cm area
    new_x = env.bin1_pos[0] + np.random.uniform(-0.015, 0.015)
    new_y = env.bin1_pos[1] + np.random.uniform(-0.015, 0.015)
    
    for obj in env.objects:
        if obj.name == "Can":
            qpos = env.sim.data.get_joint_qpos(obj.joints[0])
            qpos[0] = new_x
            qpos[1] = new_y
            env.sim.data.set_joint_qpos(obj.joints[0], qpos)
            break
            
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
        obj_pos = env.sim.data.body_xpos[env.obj_body_id["Can"]]
        goal_pos = env.target_bin_placements[env.object_to_id["can"]]
        gripper_pos = np.array(
            env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
        )

        action = np.zeros(7)

        if stage == 0:
            action[:3] = obj_pos - gripper_pos
            action[2] = 0
            action[-1] = -1
            if (action[:3] ** 2).sum() < 0.0001:
                stage = 1
            action[:3] *= 10

        if stage == 1:
            action[:3] = obj_pos + np.array([0, 0, 0.015]) - gripper_pos
            action[-1] = -1
            if (action[:3] ** 2).sum() < 0.0001:
                stage = 2
            action[:3] *= 10

        if stage == 2:
            action[:] = 0
            action[-1] = 1
            stage_counter += 1
            if stage_counter == 3:
                stage = 3
                stage_counter = 0

        if stage == 3:
            action[:] = 0
            action[2] = 1
            action[-1] = 1
            stage_counter += 1
            if stage_counter == 8:
                stage = 4
                stage_counter = 0

        if stage == 4:
            action[:2] = goal_pos[:2] - obj_pos[:2]
            action[-1] = 1
            if (action[:3] ** 2).sum() < 0.0001:
                stage = 5
            action[:3] *= 10

        if stage == 5:
            action[:3] = goal_pos + np.array([0, 0, 0.05]) - obj_pos
            action[-1] = 1
            if (action[:3] ** 2).sum() < 0.0001:
                stage = 6
                stage_counter = 0
            action[:3] *= 10

        if stage == 6:
            action[:] = 0
            action[-1] = -1
            stage_counter += 1
            if stage_counter == 8:
                stage = 7
                stage_counter = 0

        if stage == 7:
            action[:] = 0
            action[2] = 1
            action[-1] = -1
            stage_counter += 1
            if stage_counter >= 5:
                action[2] = 0

        next_obs, r, d, info = env.step(action)
        env.render()
        
        state = np.concatenate([
            next_obs["robot0_eef_pos"],
            next_obs["robot0_eef_quat"],
            next_obs["robot0_gripper_qpos"]
        ])
        ep_states.append(state)
        
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
if not os.path.isdir(target_folder):
    os.makedirs(target_folder)
torch.save(payload, target_folder + "/0_" + str(len(obs_list)) + ".pt")
np.save(target_folder + "/demo_starts.npy", np.array(demo_starts))
np.save(target_folder + "/demo_ends.npy", np.array(demo_ends))

env.close()
