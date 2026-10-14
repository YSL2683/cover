import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import cv2
import imageio
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"

from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy, _make_noise_scheduler

class OodPositionConfig:
    def __init__(self, x_bounds, y_bounds):
        self.x_bounds = list(x_bounds)
        self.y_bounds = list(y_bounds)

class EnvModifierConfig:
    def __init__(self, mode, ood_position):
        self.mode = mode
        self.ood_position = ood_position
        self.disturbance = None
        self.visual_ood = None
        self.camera_ood = None
        self.robot_pose_ood = None

def load_policy(policy_path, device="cuda", ddim_steps=20):
    policy = DiffusionPolicy.from_pretrained(policy_path)
    policy.to(device)
    policy.eval()
    if ddim_steps and hasattr(policy, "noise_scheduler"):
        policy.config.noise_scheduler_type = "DDIM"
        policy.config.num_inference_steps = ddim_steps
        policy.noise_scheduler = _make_noise_scheduler(
            "DDIM",
            num_train_timesteps=policy.config.num_train_timesteps,
            beta_schedule=policy.config.beta_schedule,
            beta_start=policy.config.beta_start,
            beta_end=policy.config.beta_end,
            clip_sample=policy.config.clip_sample,
            clip_sample_range=policy.config.clip_sample_range,
            prediction_type=policy.config.prediction_type,
        )
    return policy

def draw_hud(frame, scenario_title, step, max_steps, eef_pos, nut_pos, is_success, color_theme):
    H, W, _ = frame.shape
    banner_h = 75
    banner = np.zeros((banner_h, W, 3), dtype=np.uint8)
    banner[:] = (20, 20, 25) # Dark gray background

    # Calculate distance
    dist = np.linalg.norm(eef_pos - nut_pos) if (eef_pos is not None and nut_pos is not None) else 0.0

    # Draw border line
    cv2.line(banner, (0, banner_h - 2), (W, banner_h - 2), color_theme, 2)

    # Title & Step
    cv2.putText(banner, f"{scenario_title}", (15, 24), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    
    status_text = "TASK SUCCESS" if is_success else ("FAIL (Timeout)" if step >= max_steps else "RUNNING (Evaluating)")
    status_color = (0, 230, 118) if is_success else ((0, 69, 255) if step >= max_steps else (255, 200, 0))
    cv2.putText(banner, f"Step: {step:03d}/{max_steps} | Status: {status_text}", (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)

    # Coordinates
    eef_str = f"EEF: ({eef_pos[0]:.2f}, {eef_pos[1]:.2f}, {eef_pos[2]:.2f})" if eef_pos is not None else ""
    nut_str = f"Nut: ({nut_pos[0]:.2f}, {nut_pos[1]:.2f}, {nut_pos[2]:.2f})" if nut_pos is not None else ""
    dist_str = f"Target Gap: {dist*100:4.1f} cm" if eef_pos is not None else ""
    
    cv2.putText(banner, f"{eef_str} | {nut_str}", (W - 470, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(banner, f"{dist_str}", (W - 470, 48), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 230, 255) if dist < 0.08 else (100, 140, 255), 1, cv2.LINE_AA)

    # Annotate view labels on frame
    view_banner_h = 24
    half_w = W // 2
    cv2.rectangle(frame, (10, 5), (160, 5 + view_banner_h), (0, 0, 0), -1)
    cv2.putText(frame, "Agentview (Global)", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (half_w + 10, 5), (half_w + 200, 5 + view_banner_h), (0, 0, 0), -1)
    cv2.putText(frame, "Wristview (Eye-in-Hand)", (half_w + 15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    composite = np.vstack([banner, frame])
    return composite

def record_episode(policy, env_modifier_config, scenario_title, color_theme, seed=42, render_wh=(480, 480)):
    env = create_vectorized_env(
        env_name="Square",
        num_envs=1,
        device="cuda",
        camera_size=128,
        render_size=128,
        video_key="observation.images.agentview",
        debug=True,
        env_modifier_config=env_modifier_config,
    )

    policy.reset()
    obs, _ = env.reset(seed=seed)
    
    sim = env.envs[0].env.sim
    nut_body_id = sim.model.body_name2id("SquareNut_main")
    eef_body_id = sim.model.body_name2id("gripper0_eef")

    frames = []
    step = 0
    max_steps = 300
    is_success = False

    while step < max_steps:
        # High-res render
        img_agent = sim.render(camera_name="agentview", height=render_wh[1], width=render_wh[0])[::-1]
        img_wrist = sim.render(camera_name="robot0_eye_in_hand", height=render_wh[1], width=render_wh[0])[::-1]
        raw_side_by_side = np.hstack([img_agent, img_wrist])

        # State extraction
        nut_pos = np.array(sim.data.body_xpos[nut_body_id])
        eef_pos = np.array(sim.data.body_xpos[eef_body_id])

        hud_frame = draw_hud(raw_side_by_side, scenario_title, step, max_steps, eef_pos, nut_pos, is_success, color_theme)
        frames.append(hud_frame)

        with torch.inference_mode():
            action = policy.select_action(obs)

        obs, rewards, terminated, truncated, info = env.step(action)
        step += 1

        if bool(rewards[0].item() >= 1.0):
            is_success = True
            # Update last frame status to SUCCESS and freeze for 15 frames (0.75s)
            success_frame = draw_hud(raw_side_by_side, scenario_title, step, max_steps, eef_pos, nut_pos, True, color_theme)
            frames[-1] = success_frame
            for _ in range(15):
                frames.append(success_frame)
            break

    env.close()
    return frames, is_success, step

def main():
    policy_dir = str(PROJECT_ROOT / "resfit/my_lerobot_data/bc_run_2026-09-17_10-28-08_robomimic_square_v15_200_diffusion/best/policy")
    policy = load_policy(policy_dir, device="cuda", ddim_steps=20)

    out_dir = PROJECT_ROOT / "visualization" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    brain_dir = Path("/home/moai/.gemini/antigravity-cli/brain/f279ab21-5659-4869-bcd5-c8600b0f2725")

    # 1. Record True Table Center Failure
    true_center_cfg = EnvModifierConfig("ood_position", OodPositionConfig([-0.0025, 0.0025], [-0.0575, 0.0575]))
    print("\nRecording Episode 1: TRUE Table Center (0, 0)...")
    frames_tc1, success_tc1, steps_tc1 = record_episode(
        policy, true_center_cfg,
        scenario_title="TRUE Table Center Shift (0, 0) | Failure Mode",
        color_theme=(0, 23, 255), # Red
        seed=42,
    )
    print(f"Recorded TRUE Table Center Ep 1: Success={success_tc1}, Steps={steps_tc1}, Frames={len(frames_tc1)}")

    # 2. Record ID Baseline Success (for contrast)
    print("\nRecording Episode 2: ID Baseline (Default)...")
    frames_id, success_id, steps_id = record_episode(
        policy, None,
        scenario_title="ID Baseline (Default Demo Area) | Reference Success",
        color_theme=(0, 230, 118), # Green
        seed=42,
    )
    print(f"Recorded ID Baseline: Success={success_id}, Steps={steps_id}, Frames={len(frames_id)}")

    # Save MP4s
    mp4_tc = out_dir / "true_table_center_failure.mp4"
    imageio.mimsave(str(mp4_tc), frames_tc1, fps=20)
    print(f"Saved: {mp4_tc}")

    mp4_id = out_dir / "id_baseline_success.mp4"
    imageio.mimsave(str(mp4_id), frames_id, fps=20)
    print(f"Saved: {mp4_id}")

    # Save Optimized GIF for True Table Center Failure
    # Downsample slightly to 640px width and skip every 2nd frame for compact file size (~6MB)
    print("Generating optimized animated GIF for True Table Center failure...")
    gif_frames_tc = []
    for i, f in enumerate(frames_tc1):
        if i % 2 == 0:
            h, w = f.shape[:2]
            new_w = 640
            new_h = int(h * (new_w / w))
            resized = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA)
            gif_frames_tc.append(resized)

    gif_tc = out_dir / "true_table_center_failure.gif"
    imageio.mimsave(str(gif_tc), gif_frames_tc, fps=12, loop=0)
    print(f"Saved: {gif_tc} ({gif_tc.stat().st_size / 1024 / 1024:.1f} MB)")

    # Save Comparison GIF (ID vs True Table Center side-by-side or stacked)
    # We create a stacked comparison GIF: Top = ID Success, Bottom = True Center Failure
    print("Generating dual comparison GIF (ID Reference vs True Table Center Failure)...")
    comp_frames = []
    min_len = max(len(frames_id), len(frames_tc1))
    
    # Pad shorter frames with its last frame
    id_padded = frames_id + [frames_id[-1]] * (min_len - len(frames_id))
    tc_padded = frames_tc1 + [frames_tc1[-1]] * (min_len - len(frames_tc1))

    for i in range(0, min_len, 3): # every 3rd frame
        f_id = id_padded[i]
        f_tc = tc_padded[i]
        
        # Resize both to 480w
        h, w = f_id.shape[:2]
        new_w = 540
        new_h = int(h * (new_w / w))
        r_id = cv2.resize(f_id, (new_w, new_h), interpolation=cv2.INTER_AREA)
        r_tc = cv2.resize(f_tc, (new_w, new_h), interpolation=cv2.INTER_AREA)
        stacked = np.vstack([r_id, r_tc])
        comp_frames.append(stacked)

    gif_comp = out_dir / "comparison_id_vs_true_table_center.gif"
    imageio.mimsave(str(gif_comp), comp_frames, fps=10, loop=0)
    print(f"Saved: {gif_comp} ({gif_comp.stat().st_size / 1024 / 1024:.1f} MB)")

    # Copy all to brain dir
    if brain_dir.exists():
        import shutil
        shutil.copy(mp4_tc, brain_dir / "true_table_center_failure.mp4")
        shutil.copy(mp4_id, brain_dir / "id_baseline_success.mp4")
        shutil.copy(gif_tc, brain_dir / "true_table_center_failure.gif")
        shutil.copy(gif_comp, brain_dir / "comparison_id_vs_true_table_center.gif")
        print(f"Copied all media files to brain artifact directory: {brain_dir}")

if __name__ == "__main__":
    main()
