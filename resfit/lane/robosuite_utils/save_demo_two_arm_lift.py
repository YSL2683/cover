import numpy as np
import os
import torch
import robosuite as suite
from robosuite import load_controller_config

config = load_controller_config(default_controller="OSC_POSE")

NUM_DEMOS = 15
ROOT_FOLDER = "./demo/robosuite_two_arm_lift/"
target_folder = ROOT_FOLDER + str(NUM_DEMOS)

env = suite.make(
    env_name="TwoArmLift",
    robots=["Panda", "Panda"], # 양팔 환경
    controller_configs=config,
    camera_names=["frontview", "robot0_eye_in_hand", "robot1_eye_in_hand"],
    camera_heights=128,
    camera_widths=128,
    control_freq=10,
    horizon=80,
    has_renderer=True,           # 실시간 뷰어 활성화
    has_offscreen_renderer=True, # 카메라 이미지 렌더링 유지
    hard_reset=False,
)

env.placement_initializer.x_range = [-0.03, 0.03]
env.placement_initializer.y_range = [-0.03, 0.03]
env.placement_initializer.rotation = (np.pi - np.pi/36, np.pi + np.pi/36)

obs_list = []
next_obs_list = []
action_list = []
reward_list = []
not_done_list = []

demo_starts = []
demo_ends = []

num_saved_demos = 0
attempts = 0

print(f"[{NUM_DEMOS}개의 성공 데모 수집을 시작합니다 (Two Arm Lift)...]")

while num_saved_demos < NUM_DEMOS:
    attempts += 1
    obs = env.reset()
    
    temp_obs_list = []
    temp_next_obs_list = []
    temp_action_list = []
    temp_reward_list = []
    temp_not_done_list = []
    
    stage = 0
    stage_counter = 0
    
    img_obs = np.concatenate(
        [
            obs["frontview_image"][::-1],
            obs["robot0_eye_in_hand_image"][::-1],
            obs["robot1_eye_in_hand_image"][::-1]
        ],
        axis=2
    ).transpose((2, 0, 1))
    
    success = False
    
    for _ in range(79):
        # 양쪽 손잡이 위치 추적
        handle0_pos = env._handle0_xpos
        handle1_pos = env._handle1_xpos
        
        gripper0_pos = np.array(env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")])
        gripper1_pos = np.array(env.sim.data.site_xpos[env.sim.model.site_name2id("gripper1_grip_site")])
        
        # 냄비의 회전 각도(Yaw) 계산 (handle0 -> handle1 벡터 기준)
        vec = handle1_pos[:2] - handle0_pos[:2]
        pot_yaw = np.arctan2(vec[1], vec[0])
        
        # 로봇 0의 그리퍼 방향 오차 계산
        gripper0_mat = np.array(env.sim.data.site_xmat[env.sim.model.site_name2id("gripper0_grip_site")]).reshape(3, 3)
        gripper0_yaw = np.arctan2(gripper0_mat[1, 0], gripper0_mat[0, 0])
        yaw_err_0 = (pot_yaw - gripper0_yaw + np.pi) % (2 * np.pi) - np.pi
        
        # 로봇 1의 그리퍼 방향 오차 계산
        gripper1_mat = np.array(env.sim.data.site_xmat[env.sim.model.site_name2id("gripper1_grip_site")]).reshape(3, 3)
        gripper1_yaw = np.arctan2(gripper1_mat[1, 0], gripper1_mat[0, 0])
        yaw_err_1 = (pot_yaw - gripper1_yaw + np.pi) % (2 * np.pi) - np.pi
        
        action = np.zeros(14)
        action[6] = -1 # 로봇 0 그립 열기
        action[13] = -1 # 로봇 1 그립 열기

        # Stage 0: 로봇 0, 1 모두 각자의 손잡이 바로 위로 이동 및 Yaw 정렬
        if stage == 0:
            target0 = handle0_pos + np.array([0, 0, 0.05])
            target1 = handle1_pos + np.array([0, 0, 0.05])
            
            action[:3] = target0 - gripper0_pos
            action[7:10] = target1 - gripper1_pos
            
            action[5] = np.clip(yaw_err_0 * 5.0, -1, 1)
            action[12] = np.clip(yaw_err_1 * 5.0, -1, 1)
            
            if (action[:3] ** 2).sum() < 0.0001 and (action[7:10] ** 2).sum() < 0.0001 and abs(yaw_err_0) < 0.05 and abs(yaw_err_1) < 0.05:
                stage = 1
                
            action[:3] *= 10
            action[7:10] *= 10

        # Stage 1: 수직 하강하여 손잡이를 향함
        elif stage == 1:
            # 손잡이 중심보다 살짝 아래로 내려가서 안정적으로 파지
            target0 = handle0_pos + np.array([0, 0, -0.01])
            target1 = handle1_pos + np.array([0, 0, -0.01])
            
            action[:3] = target0 - gripper0_pos
            action[7:10] = target1 - gripper1_pos
            
            action[5] = np.clip(yaw_err_0 * 5.0, -1, 1)
            action[12] = np.clip(yaw_err_1 * 5.0, -1, 1)
            
            if (action[:3] ** 2).sum() < 0.0001 and (action[7:10] ** 2).sum() < 0.0001:
                stage = 2
                
            action[:3] *= 10
            action[7:10] *= 10

        # Stage 2: 양팔 그립 닫기 (파지 완료 후 대기)
        elif stage == 2:
            action[:3] = 0
            action[7:10] = 0
            action[6] = 1  # 로봇 0 그립 닫기
            action[13] = 1 # 로봇 1 그립 닫기
            
            stage_counter += 1
            if stage_counter == 10: # 물리적 안정화를 위해 충분히 대기
                stage = 3
                stage_counter = 0

        # Stage 3: 양팔 동시에 위로 들어올림 (Lifting)
        elif stage == 3:
            action[:3] = 0
            action[7:10] = 0
            action[2] = 1  # 로봇 0 Z상승
            action[9] = 1  # 로봇 1 Z상승
            action[6] = 1
            action[13] = 1

        next_obs, r, d, info = env.step(action)
        env.render() # 뷰어 업데이트
        
        next_img_obs = np.concatenate(
            [
                next_obs["frontview_image"][::-1],
                next_obs["robot0_eye_in_hand_image"][::-1],
                next_obs["robot1_eye_in_hand_image"][::-1],
            ],
            axis=2,
        ).transpose((2, 0, 1))
        
        temp_obs_list.append(img_obs)
        temp_next_obs_list.append(next_img_obs)
        temp_action_list.append(action)
        
        r = -1 if r <= 0 else 100
        if r == 100:
            d = True
            success = True
            
        temp_reward_list.append([r])
        temp_not_done_list.append([not d])
        img_obs = next_img_obs

        if d:
            break

    # 성공 여부 판별
    if success:
        demo_starts.append(len(obs_list))
        obs_list.extend(temp_obs_list)
        next_obs_list.extend(temp_next_obs_list)
        action_list.extend(temp_action_list)
        reward_list.extend(temp_reward_list)
        not_done_list.extend(temp_not_done_list)
        demo_ends.append(len(obs_list))
        
        num_saved_demos += 1
        print(f"✅ 성공! (수집: {num_saved_demos}/{NUM_DEMOS}, 시도 횟수: {attempts})")
    else:
        print(f"❌ 실패 (버려짐) - 시도 횟수: {attempts}")

payload = [
    np.array(obs_list),
    np.array(next_obs_list),
    np.array(action_list),
    np.array(reward_list),
    np.array(not_done_list),
]
if not os.path.isdir(target_folder):
    os.makedirs(target_folder)
torch.save(payload, target_folder + "/0_" + str(len(obs_list)) + ".pt")
np.save(target_folder + "/demo_starts.npy", np.array(demo_starts))
np.save(target_folder + "/demo_ends.npy", np.array(demo_ends))

env.close()
