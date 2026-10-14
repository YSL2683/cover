#!/usr/bin/env python3
"""
Generate 2 side-by-side policy comparison videos for In-Distribution (ID) and Position OOD:
Compares:
1) base policy (BC diffusion policy, step 66000)
2) residual rl (ResFiT without visual similarity reward)
3) Ours (Decoupled E2C visual similarity PBRS)

Scenario requirement: ONLY Ours succeeds, while Base Policy and Residual RL fail.

Layout per policy row: [ Main Camera (agentview) | Wrist Camera (robot0_eye_in_hand) ]
Stacked vertically:
  Top   : base policy
  Middle: residual rl
  Bottom: Ours
"""

import os
import sys

# Cap threads to prevent CPU contention with ongoing background training
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path
import cv2
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy, _make_noise_scheduler
from resfit.lerobot.utils.load_policy import load_policy
from resfit.rl_finetuning.config.residual_td3 import ResidualTD3DexmgConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.wrappers.residual_env_wrapper import BasePolicyVecEnvWrapper


def load_base_diffusion_policy(policy_path: str | Path, device: torch.device, ddim_steps: int = 20) -> DiffusionPolicy:
    print(f"Loading base policy from: {policy_path}")
    base_policy: DiffusionPolicy = load_policy(Path(policy_path))
    base_policy.to(device)
    base_policy.eval()

    if ddim_steps is not None and hasattr(base_policy, "noise_scheduler"):
        print(f"Patching base policy with DDIM ({ddim_steps} steps)...")
        base_policy.config.noise_scheduler_type = "DDIM"
        base_policy.config.num_inference_steps = ddim_steps
        base_policy.noise_scheduler = _make_noise_scheduler(
            "DDIM",
            num_train_timesteps=base_policy.config.num_train_timesteps,
            beta_schedule=base_policy.config.beta_schedule,
            beta_start=base_policy.config.beta_start,
            beta_end=base_policy.config.beta_end,
            clip_sample=base_policy.config.clip_sample,
        )
    return base_policy


def load_rl_agent(
    ckpt_path: str | Path,
    device: torch.device,
    action_scale: float = 0.1,
    prop_dim: int = 9,
    action_dim: int = 7,
    image_keys: list[str] = ["observation.images.agentview", "observation.images.robot0_eye_in_hand"],
) -> QAgent:
    cfg = OmegaConf.structured(ResidualTD3DexmgConfig)
    cfg.agent.actor.action_scale = action_scale

    agent = QAgent(
        obs_shape=(3, 128, 128),
        prop_shape=(prop_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
    )
    print(f"Loading QAgent checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    agent.load_state_dict(state_dict)
    agent.to(device)
    agent.eval()
    return agent


def create_env(
    task: str,
    base_policy: DiffusionPolicy,
    action_scaler: ActionScaler,
    state_standardizer: StateStandardizer,
    device: torch.device,
    env_modifier_config=None,
) -> BasePolicyVecEnvWrapper:
    vec_env = create_vectorized_env(
        env_name=task,
        num_envs=1,
        device=str(device),
        video_key="observation.images.agentview",
        debug=True,  # SyncVectorEnv for 1 env (lightweight, zero subprocess overhead)
        camera_size=128,
        env_modifier_config=env_modifier_config,
    )
    return BasePolicyVecEnvWrapper(
        vec_env=vec_env,
        base_policy=base_policy,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
    )


def rollout_episode(
    env: BasePolicyVecEnvWrapper,
    agent: QAgent | None,
    seed: int,
    device: torch.device,
    max_steps: int = 250,
    record_frames: bool = False,
    camera_target_size: tuple[int, int] = (256, 256),
) -> tuple[bool, list[np.ndarray], int]:
    """Run an episode with deterministic seeding.
    If agent is None -> base policy only (residual action = 0).
    Returns (success, frames, step_count).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs, _ = env.reset(seed=seed)
    done = False
    is_success = False
    step_count = 0
    frames = []

    def _extract_frame():
        main_img = obs["observation.images.agentview"][0].detach().cpu().numpy().transpose(1, 2, 0)
        wrist_img = obs["observation.images.robot0_eye_in_hand"][0].detach().cpu().numpy().transpose(1, 2, 0)
        main_u8 = (main_img * 255.0).clip(0, 255).astype(np.uint8)
        wrist_u8 = (wrist_img * 255.0).clip(0, 255).astype(np.uint8)
        if (main_u8.shape[1], main_u8.shape[0]) != camera_target_size:
            main_u8 = cv2.resize(main_u8, camera_target_size, interpolation=cv2.INTER_LINEAR)
            wrist_u8 = cv2.resize(wrist_u8, camera_target_size, interpolation=cv2.INTER_LINEAR)
        return np.concatenate([main_u8, wrist_u8], axis=1)  # (H, 2W, 3)

    if record_frames:
        frames.append(_extract_frame())

    action_dim = env.action_dim
    while not done and step_count < max_steps:
        step_count += 1
        with torch.no_grad():
            if agent is None:
                res_action = torch.zeros((1, action_dim), device=device, dtype=torch.float32)
            else:
                res_action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

        next_obs, reward, terminated, truncated, info = env.step(res_action)
        obs = next_obs

        # Check success
        succ = False
        if "final_info" in info and info["final_info"] is not None:
            fin = info["final_info"]
            if isinstance(fin, (list, tuple)) and len(fin) > 0 and fin[0] is not None:
                succ = bool(fin[0].get("success", False))
        if not succ and "success" in info:
            s_val = info["success"]
            if isinstance(s_val, torch.Tensor):
                succ = bool(s_val[0].item())
            elif isinstance(s_val, (list, tuple, np.ndarray)):
                succ = bool(s_val[0])
            else:
                succ = bool(s_val)
        if not succ and reward[0].item() >= 1.0:
            succ = True

        if succ:
            is_success = True

        done = bool(terminated[0].item() or truncated[0].item() or is_success)

        if record_frames:
            frames.append(_extract_frame())

    return is_success, frames, step_count

def get_nut_displacement_from_id(env: BasePolicyVecEnvWrapper, seed: int):
    """Calculate the displacement of the square nut from the ID distribution box."""
    obs, _ = env.reset(seed=seed)
    sub_env = env.vec_env.vec_env.envs[0]
    body_id = sub_env.env.sim.model.body_name2id("SquareNut_main")
    pos = sub_env.env.sim.data.body_xpos[body_id]
    tx, ty = sub_env.env.table_offset[0], sub_env.env.table_offset[1]
    rel_x = pos[0] - tx
    rel_y = pos[1] - ty

    # ID reference bounds: x in [-0.115, -0.110], y in [0.110, 0.225]
    dist_x = max(0.0, -0.115 - rel_x, rel_x - (-0.110))
    dist_y = max(0.0, 0.110 - rel_y, rel_y - 0.225)
    return rel_x, rel_y, dist_x, dist_y


def search_differentiating_seed(
    env: BasePolicyVecEnvWrapper,
    base_policy: DiffusionPolicy,
    resfit_agent: QAgent,
    ours_agent: QAgent,
    device: torch.device,
    start_seed: int = 100,
    max_search_seeds: int = 150,
    min_ood_displacement: float = 0.0,
) -> int:
    """Find a seed where ONLY Ours succeeds, and both Base Policy and ResFiT fail.
    If min_ood_displacement > 0, enforces that the initial nut position is strictly and visibly OOD.
    """
    disp_str = f", min_disp={min_ood_displacement*100:.1f}cm" if min_ood_displacement > 0 else ""
    print(f"\n--- Searching for seed where ONLY Ours succeeds (seeds {start_seed}..{start_seed + max_search_seeds}{disp_str}) ---")
    
    for candidate_seed in range(start_seed, start_seed + max_search_seeds):
        # 0. Check initial nut placement displacement if min_ood_displacement > 0
        if min_ood_displacement > 0.0:
            rel_x, rel_y, dist_x, dist_y = get_nut_displacement_from_id(env, candidate_seed)
            total_dist = max(dist_x, dist_y)
            if total_dist < min_ood_displacement:
                print(f". [Seed {candidate_seed}: too close to ID (dist={total_dist*100:.2f}cm < {min_ood_displacement*100:.1f}cm)]", flush=True)
                continue
            else:
                print(f"[Seed {candidate_seed} is true OOD: rel_x={rel_x:.4f}, rel_y={rel_y:.4f}, dist_from_id={total_dist*100:.2f}cm]", flush=True)

        # 1. Quick test: Check Ours first (fastest prune since we need Ours to succeed)
        ours_succ, _, ours_steps = rollout_episode(env, ours_agent, candidate_seed, device, record_frames=False)
        if not ours_succ:
            print(f". [Seed {candidate_seed}: Ours=FAIL]", flush=True)
            continue
        
        # 2. Check Base policy (must FAIL)
        base_succ, _, base_steps = rollout_episode(env, None, candidate_seed, device, record_frames=False)
        if base_succ:
            print(f". [Seed {candidate_seed}: Ours=SUCC, Base=SUCC]", flush=True)
            continue
            
        # 3. Check ResFiT policy (must FAIL)
        resfit_succ, _, resfit_steps = rollout_episode(env, resfit_agent, candidate_seed, device, record_frames=False)
        if resfit_succ:
            print(f". [Seed {candidate_seed}: Ours=SUCC, Base=FAIL, ResFiT=SUCC]", flush=True)
            continue
            
        print(f"\n>>> FOUND PERFECT SEED: {candidate_seed} <<<")
        print(f"  Base Policy : FAIL (steps={base_steps})")
        print(f"  Residual RL : FAIL (steps={resfit_steps})")
        print(f"  Ours        : SUCCESS (steps={ours_steps})")
        return candidate_seed

    raise RuntimeError(f"Could not find a differentiating seed in range [{start_seed}, {start_seed + max_search_seeds})!")


def annotate_row(
    frame: np.ndarray,
    label: str,
    is_success: bool,
    is_finished: bool,
    step: int,
    max_steps: int,
) -> np.ndarray:
    """frame is (H, 2W, 3) where [0..W] is Main camera, [W..2W] is Wrist camera.
    Adds label to top-left of Main camera, and clean badges.
    """
    canvas = frame.copy()
    h, w_total = canvas.shape[:2]
    w_cam = w_total // 2

    # Background banner for label for high contrast
    cv2.rectangle(canvas, (6, 6), (180, 36), (0, 0, 0), -1)
    
    # Label styling
    if label == "Ours":
        color = (50, 220, 50)  # Bright green
    elif label == "residual rl":
        color = (240, 160, 40) # Bright orange/cyan
    else:
        color = (220, 220, 220) # Clean white/gray

    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)

    # Status badge on top right of Wrist camera
    if is_finished:
        if is_success:
            status_text = "SUCCESS"
            status_bg = (30, 160, 30)
            status_fg = (255, 255, 255)
        else:
            status_text = "FAIL"
            status_bg = (180, 30, 30)
            status_fg = (255, 255, 255)
    else:
        status_text = f"STEP {step}"
        status_bg = (40, 40, 40)
        status_fg = (200, 200, 200)

    badge_w = 110
    cv2.rectangle(canvas, (w_total - badge_w - 10, 6), (w_total - 10, 34), status_bg, -1)
    cv2.putText(
        canvas,
        status_text,
        (w_total - badge_w - 4, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_fg,
        2,
        cv2.LINE_AA,
    )

    # Sub-camera labels at bottom
    cv2.putText(canvas, "MAIN CAM", (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "MAIN CAM", (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "WRIST CAM", (w_cam + 12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "WRIST CAM", (w_cam + 12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas


def compose_comparison_video(
    base_frames: list[np.ndarray],
    base_succ: bool,
    resfit_frames: list[np.ndarray],
    resfit_succ: bool,
    ours_frames: list[np.ndarray],
    ours_succ: bool,
    output_path: Path,
    fps: int = 20,
    freeze_tail_seconds: float = 2.0,
):
    """Combine the 3 policy rollouts into a vertically stacked comparison video."""
    len_base = len(base_frames)
    len_resfit = len(resfit_frames)
    len_ours = len(ours_frames)

    max_len = max(len_base, len_resfit, len_ours)
    freeze_frames = int(round(freeze_tail_seconds * fps))
    total_video_frames = max_len + freeze_frames

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=fps)

    print(f"Writing comparison video to: {output_path} (total {total_video_frames} frames)...")

    for t in range(total_video_frames):
        # Base policy frame
        t_base = min(t, len_base - 1)
        base_finished = t >= (len_base - 1)
        row_base = annotate_row(
            base_frames[t_base],
            label="base policy",
            is_success=base_succ,
            is_finished=base_finished,
            step=t_base + 1,
            max_steps=len_base,
        )

        # Residual RL frame
        t_resfit = min(t, len_resfit - 1)
        resfit_finished = t >= (len_resfit - 1)
        row_resfit = annotate_row(
            resfit_frames[t_resfit],
            label="residual rl",
            is_success=resfit_succ,
            is_finished=resfit_finished,
            step=t_resfit + 1,
            max_steps=len_resfit,
        )

        # Ours frame
        t_ours = min(t, len_ours - 1)
        ours_finished = t >= (len_ours - 1)
        row_ours = annotate_row(
            ours_frames[t_ours],
            label="Ours",
            is_success=ours_succ,
            is_finished=ours_finished,
            step=t_ours + 1,
            max_steps=len_ours,
        )

        # Separator line between rows (2px dark line)
        sep = np.zeros((3, row_base.shape[1], 3), dtype=np.uint8) + 80
        stacked = np.concatenate([row_base, sep, row_resfit, sep, row_ours], axis=0)

        writer.append_data(stacked)

    writer.close()
    print(f"Video saved successfully: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ood_only", action="store_true", help="Generate only Position OOD video")
    parser.add_argument("--min_disp", type=float, default=0.012, help="Minimum nut displacement from ID region (default: 0.012 = 1.2cm)")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths configuration
    base_policy_path = REPO_ROOT / "resfit/my_lerobot_data/bc_run_2026-08-29_14-38-11_robomimic_square_v15_50_diffusion/policy_step_66000/policy"
    dataset_name = "ysl2683/robomimic_square_v15_50"
    dataset_dir = REPO_ROOT / "resfit/my_lerobot_data" / dataset_name

    # Checkpoint paths
    id_resfit_ckpt = REPO_ROOT / "resfit/outputs/2026-09-11_13-58-20_Square_ID_resfit_seed62/models/agent_best.pt"
    id_decoupled_ckpt = REPO_ROOT / "resfit/outputs/2026-09-18_17-44-12_Square_reward_pbrs_no_mask_nstep_weighted_time_beta1.0_scale0.1_1000k_seed42/models/agent_best.pt"

    ood_resfit_ckpt = REPO_ROOT / "resfit/outputs/2026-09-24_18-54-01_Square_position_ood_4.875x15.875_resfit_1000k_seed42/models/agent_best.pt"
    ood_decoupled_ckpt = REPO_ROOT / "resfit/outputs/2026-10-03_14-19-09_Square_position_ood_4.875x15.875_reward_pbrs_no_mask_nstep_weighted_time_decoupled_z8_beta1.0_scale0.1_1000k_seed42/models/agent_best.pt"

    output_dir = REPO_ROOT / "resfit/outputs/comparison_videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Base Policy & Normalization
    base_policy = load_base_diffusion_policy(base_policy_path, device, ddim_steps=20)
    dataset = LeRobotDataset(dataset_name, root=dataset_dir)
    action_scaler = ActionScaler.from_dataset_stats(dataset.meta.stats["action"], action_scale=0.1, min_range_per_dim=0.1, device=device)
    state_standardizer = StateStandardizer.from_dataset_stats(dataset.meta.stats["observation.state"], min_std=0.1, device=device)

    id_video_path = None
    if not args.ood_only:
        # =========================================================================
        # PART 1: In-Distribution (ID) Comparison
        # =========================================================================
        print("\n=======================================================")
        print("Generating In-Distribution (ID) Policy Comparison Video")
        print("=======================================================")
        env_id = create_env("Square", base_policy, action_scaler, state_standardizer, device, env_modifier_config=None)
        id_resfit_agent = load_rl_agent(id_resfit_ckpt, device, action_scale=0.1)
        id_ours_agent = load_rl_agent(id_decoupled_ckpt, device, action_scale=0.1)

        id_seed = search_differentiating_seed(env_id, base_policy, id_resfit_agent, id_ours_agent, device, start_seed=100)

        print(f"\nRecording ID episodes with seed {id_seed}...")
        b_succ, b_frames, _ = rollout_episode(env_id, None, id_seed, device, record_frames=True)
        r_succ, r_frames, _ = rollout_episode(env_id, id_resfit_agent, id_seed, device, record_frames=True)
        o_succ, o_frames, _ = rollout_episode(env_id, id_ours_agent, id_seed, device, record_frames=True)

        id_video_path = output_dir / f"Square_ID_policy_comparison_seed{id_seed}.mp4"
        compose_comparison_video(b_frames, b_succ, r_frames, r_succ, o_frames, o_succ, id_video_path)

        env_id.vec_env.close()
        del id_resfit_agent, id_ours_agent, env_id
        torch.cuda.empty_cache()

    # =========================================================================
    # PART 2: Position OOD Comparison
    # =========================================================================
    print("\n=======================================================")
    print("Generating Position OOD Policy Comparison Video (Distinct Initial Nut Position)")
    print("=======================================================")
    ood_modifier_config = OmegaConf.create({
        "mode": "ood_position",
        "ood_position": {
            "x_bounds": [-0.136875, -0.088125],
            "y_bounds": [0.088125, 0.246875],
        },
        "disturbance": None,
    })
    env_ood = create_env("Square", base_policy, action_scaler, state_standardizer, device, env_modifier_config=ood_modifier_config)
    ood_resfit_agent = load_rl_agent(ood_resfit_ckpt, device, action_scale=0.1)
    ood_ours_agent = load_rl_agent(ood_decoupled_ckpt, device, action_scale=0.1)

    ood_seed = search_differentiating_seed(
        env_ood,
        base_policy,
        ood_resfit_agent,
        ood_ours_agent,
        device,
        start_seed=100,
        min_ood_displacement=args.min_disp,
    )

    # Report displacement
    rel_x, rel_y, dist_x, dist_y = get_nut_displacement_from_id(env_ood, ood_seed)
    print(f"\nTarget OOD Nut placement: rel_x={rel_x:.4f}, rel_y={rel_y:.4f} (displacement from ID box: {max(dist_x, dist_y)*100:.2f} cm)")

    print(f"\nRecording OOD episodes with seed {ood_seed}...")
    b_succ_ood, b_frames_ood, _ = rollout_episode(env_ood, None, ood_seed, device, record_frames=True)
    r_succ_ood, r_frames_ood, _ = rollout_episode(env_ood, ood_resfit_agent, ood_seed, device, record_frames=True)
    o_succ_ood, o_frames_ood, _ = rollout_episode(env_ood, ood_ours_agent, ood_seed, device, record_frames=True)

    ood_video_path = output_dir / f"Square_OOD_policy_comparison_seed{ood_seed}.mp4"
    compose_comparison_video(b_frames_ood, b_succ_ood, r_frames_ood, r_succ_ood, o_frames_ood, o_succ_ood, ood_video_path)

    env_ood.vec_env.close()
    del ood_resfit_agent, ood_ours_agent, env_ood
    torch.cuda.empty_cache()

    print("\n=======================================================")
    print("Videos generation finished!")
    if id_video_path:
        print(f"ID Video : {id_video_path}")
    print(f"OOD Video: {ood_video_path}")
    print("=======================================================")


if __name__ == "__main__":
    main()
