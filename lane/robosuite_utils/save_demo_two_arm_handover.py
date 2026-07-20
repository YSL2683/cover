import numpy as np
import os
import torch
import robosuite as suite
from robosuite import load_controller_config

config = load_controller_config(default_controller="OSC_POSE")

NUM_DEMOS = 20
ROOT_FOLDER = "./demo/robosuite_two_arm_handover/"
target_folder = ROOT_FOLDER + str(NUM_DEMOS)

env = suite.make(
    env_name="TwoArmHandover",
    robots=["Panda", "Panda"], # 양팔 환경
    controller_configs=config,
    camera_names=["frontview", "robot0_eye_in_hand", "robot1_eye_in_hand"],
    camera_heights=128,
    camera_widths=128,
    control_freq=10,
    horizon=200,
    has_renderer=True,           # 실시간 뷰어 활성화
    has_offscreen_renderer=True, # 카메라 이미지 렌더링 유지
    hard_reset=False,
)

# 초기화 조건 변경: 망치 위치 +-0.01 범위 내 생성, 회전 고정
env.placement_initializer.x_range = [-0.01, 0.01]
env.placement_initializer.y_range = [-0.01, 0.01]
env.placement_initializer.rotation = np.pi/2

obs_list = []
next_obs_list = []
action_list = []
reward_list = []
not_done_list = []

demo_starts = []
demo_ends = []

num_saved_demos = 0
attempts = 0

print(f"[{NUM_DEMOS}개의 성공 데모 수집을 시작합니다 (Two Arm Handover)...]")

while num_saved_demos < NUM_DEMOS:
    attempts += 1
    obs = env.reset()
    
    # 망치 머리 방향 강제 고정 (로봇 0을 향하도록 쿼터니언 직접 주입)
    try:
        qpos = env.sim.data.get_joint_qpos(env.hammer.joints[0]).copy()
        qpos[3:] = [0, 0, -0.70710677, 0.70710677]
        env.sim.data.set_joint_qpos(env.hammer.joints[0], qpos)
        env.sim.forward()
        obs = env._get_observations() # 업데이트된 observation 가져오기
    except Exception as e:
        pass
    
    temp_obs_list = []
    temp_next_obs_list = []
    temp_action_list = []
    temp_reward_list = []
    temp_not_done_list = []
    
    stage = 0
    stage_counter = 0
    success_counter = 0
    
    img_obs = np.concatenate(
        [
            obs["frontview_image"][::-1],
            obs["robot0_eye_in_hand_image"][::-1],
            obs["robot1_eye_in_hand_image"][::-1]
        ],
        axis=2
    ).transpose((2, 0, 1))
    
    success = False
    
    try:
        head_id = env.sim.model.geom_name2id(env.hammer.head_geoms[0])
    except:
        head_id = env.sim.model.geom_name2id("hammer_head")
        
    for _ in range(199):
        obj_pos = env.sim.data.geom_xpos[head_id] # 망치 머리 (로봇 0이 잡을 곳)
        handle_pos = env._handle_xpos # 망치 손잡이 (로봇 1이 잡을 곳)
        
        gripper0_pos = np.array(env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")])
        gripper1_pos = np.array(env.sim.data.site_xpos[env.sim.model.site_name2id("gripper1_grip_site")])
        
        # 망치 손잡이의 방향(Yaw) 계산
        vec = handle_pos[:2] - obj_pos[:2]
        handle_yaw = np.arctan2(vec[1], vec[0])
        
        # 로봇 0의 그리퍼 방향 오차 계산 (사용자 요청대로 길이 방향과 수직(np.pi/2)으로 정렬)
        gripper0_mat = np.array(env.sim.data.site_xmat[env.sim.model.site_name2id("gripper0_grip_site")]).reshape(3, 3)
        gripper0_yaw = np.arctan2(gripper0_mat[1, 0], gripper0_mat[0, 0])
        target_yaw_0 = handle_yaw + np.pi/2
        yaw_err_0 = (target_yaw_0 - gripper0_yaw + np.pi/2) % np.pi - np.pi/2
        
        # 로봇 1의 그리퍼 방향 오차 계산 (손잡이를 잡기 위해 손잡이 방향과 수직(np.pi/2)으로 정렬)
        gripper1_mat = np.array(env.sim.data.site_xmat[env.sim.model.site_name2id("gripper1_grip_site")]).reshape(3, 3)
        gripper1_yaw = np.arctan2(gripper1_mat[1, 0], gripper1_mat[0, 0])
        target_yaw_1 = handle_yaw + np.pi/2
        yaw_err_1 = (target_yaw_1 - gripper1_yaw + np.pi/2) % np.pi - np.pi/2
        
        action = np.zeros(14)
        action[6] = -1 # 로봇 0 그립 열기
        action[13] = -1 # 로봇 1 그립 열기

        # 로봇 0이 잡을 목표 위치 및 Handover 시 회전 목표 결정
        # 로봇 0은 -Y에 있고 로봇 1은 +Y에 있습니다.
        if obj_pos[1] < handle_pos[1]:
            # 망치 머리(obj_pos)가 로봇 0쪽에 가까움 -> 로봇 0은 목을 잡고, 로봇 1은 손잡이 끝부분 아래를 잡음
            grab_target0 = obj_pos + np.array([vec[0]*0.55, vec[1]*0.55, 0])
            grab_target1 = obj_pos + np.array([vec[0]*1.45, vec[1]*1.45, 0])
            handle_target_yaw_val = np.pi/2 # 로봇 1에게 손잡이를 향하게 함 (+Y)
        else:
            # 망치 손잡이 끝부분이 로봇 0쪽에 가까움 -> 로봇 0은 손잡이 끝부분을 잡고, 로봇 1은 목을 잡음
            grab_target0 = obj_pos + np.array([vec[0]*1.45, vec[1]*1.45, 0])
            grab_target1 = obj_pos + np.array([vec[0]*0.55, vec[1]*0.55, 0])
            handle_target_yaw_val = -np.pi/2 # 로봇 1에게 머리 부분을 향하게 함 (+Y)


        # Stage 0: 로봇 0, 망치 목 부분 위로 이동 및 Yaw 정렬
        if stage == 0:
            target0 = grab_target0 + np.array([0, 0, 0.1])
            action[:3] = target0 - gripper0_pos
            action[5] = np.clip(yaw_err_0 * 5.0, -1, 1)
            
            if (action[:3] ** 2).sum() < 0.0001 and abs(yaw_err_0) < 0.05:
                stage = 1
            action[:3] *= 10

        # Stage 1: 로봇 0, 망치 목 부분으로 하강
        elif stage == 1:
            # 조금 더 깊게 잡기 위해 Z축으로 -0.01 더 내려갑니다
            target0 = grab_target0 + np.array([0, 0, -0.01])
            action[:3] = target0 - gripper0_pos
            action[5] = np.clip(yaw_err_0 * 5.0, -1, 1)
            
            if (action[:3] ** 2).sum() < 0.0001:
                stage = 2
            action[:3] *= 10

        # Stage 2: 로봇 0 파지
        elif stage == 2:
            action[:3] = 0
            action[6] = 1
            stage_counter += 1
            if stage_counter >= 10:
                stage = 3
                stage_counter = 0

        # Stage 3: 로봇 0이 망치를 들어올려 가운데 지점으로 이동하며, 손잡이를 로봇 1쪽(+Y)으로 회전
        elif stage == 3:
            target0 = np.array([0.0, -0.05, 1.05]) # 로봇 두 대 사이의 가운데 허공
            action[:3] = target0 - gripper0_pos
            action[6] = 1
            
            # 망치가 로봇 0의 팔과 충돌하지 않도록 미리 계산된 최적의 방향으로 회전시킵니다.
            handle_target_yaw = handle_target_yaw_val
            yaw_err_rotate = (handle_target_yaw - handle_yaw + np.pi) % (2*np.pi) - np.pi
            action[5] = np.clip(yaw_err_rotate * 5.0, -1, 1)
            
            if (action[:3] ** 2).sum() < 0.0001 and abs(yaw_err_rotate) < 0.05:
                stage = 4
            action[:3] *= 10

        # Stage 4: 로봇 1, 대상 위치(grab_target1) 위로 이동 (공중에 떠있는 손잡이를 실시간 추적)
        elif stage == 4:
            # 로봇 0은 제자리 유지
            action[:3] = 0
            action[6] = 1
            
            target1 = grab_target1 + np.array([0, 0, 0.1])
            action[7:10] = target1 - gripper1_pos
            action[12] = np.clip(yaw_err_1 * 5.0, -1, 1) # 로봇 1도 Yaw 정렬
            
            if (action[7:10] ** 2).sum() < 0.0005 and abs(yaw_err_1) < 0.05:
                stage = 5
            action[7:10] *= 10

        # Stage 5: 로봇 1, 대상 위치로 하강하여 정확히 파지
        elif stage == 5:
            action[:3] = 0
            action[6] = 1
            
            target1 = grab_target1
            action[7:10] = target1 - gripper1_pos
            action[12] = np.clip(yaw_err_1 * 5.0, -1, 1)
            
            if (action[7:10] ** 2).sum() < 0.0001:
                stage = 6
            action[7:10] *= 10

        # Stage 6: 로봇 1 파지 (양쪽 모두 쥐고 있는 상태)
        elif stage == 6:
            action[:3] = 0
            action[6] = 1
            action[7:10] = 0
            action[13] = 1
            stage_counter += 1
            if stage_counter >= 10:
                stage = 7
                stage_counter = 0

        # Stage 7: 로봇 0이 놓기 (Handover 완료)
        elif stage == 7:
            action[:3] = 0
            action[6] = -1 # 로봇 0 그립 열기
            action[7:10] = 0
            action[13] = 1 # 로봇 1 그립 유지
            stage_counter += 1
            if stage_counter >= 10:
                stage = 8

        # Stage 8: 로봇 1이 망치를 들고 위로 살짝 올라감
        elif stage == 8:
            action[:3] = np.array([0, 0, 0.1]) # 로봇 0은 위로 빠짐
            action[6] = -1
            action[7:10] = 0
            action[9] = 0.5 # 로봇 1은 위로 조금 들기
            action[13] = 1

        next_obs, r, d, info = env.step(action)
        env.render()
        
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
        
        # Handover 태스크는 r이 1이면 성공 (Robosuite 기본 설정)
        # 하지만 스파스 보상 표준을 위해 100으로 스케일링
        r = -1 if r <= 0 else 100
        if r == 100:
            success = True
            # 성공 판정이 났더라도 바로 종료하지 않고, 
            # Stage 8에서 로봇 0이 확실히 빠져나가는 모습을 데이터에 담기 위해 대기
            if stage == 8:
                success_counter += 1
                if success_counter >= 5:
                    d = True
            
        temp_reward_list.append([r])
        temp_not_done_list.append([not d])
        img_obs = next_img_obs

        if d or _ == 198:
            if not d and success:
                d = True
            break

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
