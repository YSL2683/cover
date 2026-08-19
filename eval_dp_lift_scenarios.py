"""
Evaluate the base Diffusion Policy across 3 OOD scenarios and save agentview videos.
Based on eval_dp_lift.py (direct robosuite.make), with OOD modifications applied via
_load_model hook (same as dexmg.py's _wrap_load_model pattern).

Scenarios:
  1. black_table : visual_ood, table_color=black, 5x5cm cube init
  2. green_cube  : visual_ood, cube_color=green,  5x5cm cube init
  3. init_noise  : robot_pose_ood qpos_noise_range=0.2, 5x5cm cube init
"""

import argparse
import sys
import os
import numpy as np
import torch
import imageio
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import robosuite as suite
from robosuite.controllers import load_controller_config
from resfit.lerobot.utils.load_policy import load_policy

CUBE_HALF   = 0.025    # ±2.5cm → 5x5cm init area (matches all 3 shell scripts)
MAX_STEPS   = 150
SUCCESS_HOLD = 5
FPS         = 20
CAMERA_SIZE = 256

# ──────────────────────────────────────────────────────────────────────────────
# OOD modifier hooks  (mirror dexmg.py _apply_env_modifiers logic)
# ──────────────────────────────────────────────────────────────────────────────

def _hook_load_model(env, scenario: str):
    """
    Wrap env._load_model so that every time robosuite rebuilds the scene
    (i.e. on every env.reset()), our visual/positional modifiers are re-applied.
    Mirrors dexmg.py's _wrap_load_model() + _apply_env_modifiers().
    """
    original_load_model = env._load_model

    def custom_load_model(*args, **kwargs):
        original_load_model(*args, **kwargs)
        _apply_modifiers(env, scenario)

    env._load_model = custom_load_model


def _apply_modifiers(env, scenario: str):
    """Apply XML-level modifications after _load_model rebuilds the scene."""
    cube_half = 0.1 if scenario == "wide_range" else CUBE_HALF

    # ── OOD position: dynamic bounds ─────────────────────────────────────────
    try:
        sampler = env.placement_initializer
        if hasattr(sampler, "samplers"):
            samplers = sampler.samplers
            sampler = samplers.get("ObjectSampler", list(samplers.values())[0])
        if sampler is not None:
            sampler.x_range = [-cube_half, cube_half]
            sampler.y_range = [-cube_half, cube_half]
    except Exception as e:
        print(f"[OOD position] warning: {e}")

    # ── Black table ─────────────────────────────────────────────────────────
    if scenario == "black_table":
        try:
            for geom in env.model.mujoco_arena.worldbody.findall(".//geom"):
                if geom.get("name") == "table_visual":
                    geom.attrib.pop("material", None)
                    geom.set("rgba", "0.05 0.05 0.05 1")
        except Exception as e:
            print(f"[black_table] warning: {e}")

    # ── Green cube ──────────────────────────────────────────────────────────
    elif scenario == "green_cube":
        try:
            for geom in env.model.root.findall(".//geom"):
                name = geom.get("name", "")
                if "cube" in name and "vis" in name:
                    geom.attrib.pop("material", None)
                    geom.set("rgba", "0 1 0 1")
        except Exception as e:
            print(f"[green_cube] warning: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Environment factory
# ──────────────────────────────────────────────────────────────────────────────

def make_env(scenario: str):
    config = load_controller_config(default_controller="OSC_POSE")

    extra = {}
    if scenario == "init_noise":
        extra["initialization_noise"] = {"magnitude": 0.2, "type": "uniform"}

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=config,
        has_renderer=False,
        has_offscreen_renderer=True,
        control_freq=10,
        horizon=MAX_STEPS,
        use_object_obs=True,
        use_camera_obs=True,
        camera_names=["frontview", "robot0_eye_in_hand", "agentview"],
        camera_heights=[128, 128, CAMERA_SIZE],
        camera_widths=[128, 128, CAMERA_SIZE],
        reward_shaping=True,
        **extra,
    )

    # Hook _load_model so modifiers are applied on every reset()
    _hook_load_model(env, scenario)

    return env


# ──────────────────────────────────────────────────────────────────────────────
# Inference helper  (matches eval_dp_lift.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

def obs_to_policy_input(obs: dict, device: str) -> dict:
    """Convert raw robosuite obs dict to LeRobot policy input.
    Mirrors eval_dp_lift.py: [::-1] flip + (C,H,W) + /255.
    """
    front_img = obs["frontview_image"][::-1].copy()
    wrist_img = obs["robot0_eye_in_hand_image"][::-1].copy()

    front_t = torch.from_numpy(front_img.transpose(2, 0, 1)).float() / 255.0
    wrist_t = torch.from_numpy(wrist_img.transpose(2, 0, 1)).float() / 255.0

    state = np.concatenate([
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
        obs["robot0_gripper_qpos"],
    ])
    state_t = torch.from_numpy(state).float()

    return {
        "observation.images.frontview":          front_t.unsqueeze(0).to(device),
        "observation.images.robot0_eye_in_hand": wrist_t.unsqueeze(0).to(device),
        "observation.state":                      state_t.unsqueeze(0).to(device),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Scenario runner
# ──────────────────────────────────────────────────────────────────────────────

def run_scenario(scenario: str, policy, n_episodes: int, video_path: str, device: str) -> float:
    print(f"\n{'='*60}")
    print(f"  Scenario : {scenario}")
    print(f"  Episodes : {n_episodes}")
    print(f"  Output   : {video_path}")
    print(f"{'='*60}")

    env = make_env(scenario)

    successes  = 0
    all_frames = []

    for ep in range(n_episodes):
        obs = env.reset()
        env.sim.forward()

        # -- force cube init (belt-and-suspenders on top of sampler patch) --
        cube_half = 0.1 if scenario == "wide_range" else CUBE_HALF
        cube_joint = env.cube.joints[0]
        qpos = env.sim.data.get_joint_qpos(cube_joint).copy()
        qpos[0] = np.random.uniform(-cube_half, cube_half)
        qpos[1] = np.random.uniform(-cube_half, cube_half)
        env.sim.data.set_joint_qpos(cube_joint, qpos)
        env.sim.forward()
        obs = env._get_observations(force_update=True)

        policy.reset()
        ep_success   = False
        success_hold = 0
        ep_frames    = []

        for _step in range(MAX_STEPS):
            # agentview render (uint8, H×W×3, right-side up)
            ep_frames.append(obs["agentview_image"][::-1].copy())

            obs_dict = obs_to_policy_input(obs, device)
            with torch.no_grad():
                action = policy.select_action(obs_dict).squeeze(0).cpu().numpy()

            obs, _r, done, _info = env.step(action)

            if env._check_success():
                success_hold += 1
                if success_hold >= SUCCESS_HOLD:
                    ep_success = True
                    break
            else:
                success_hold = 0

            if done:
                break

        tag = "SUCCESS" if ep_success else "FAILURE"
        print(f"  [{ep+1:>2}/{n_episodes}] {tag}")
        if ep_success:
            successes += 1

        all_frames.extend(ep_frames)

    env.close()

    sr = successes / n_episodes
    print(f"\n  Success rate : {sr*100:.1f}%  ({successes}/{n_episodes})")

    if all_frames:
        os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
        imageio.mimsave(video_path, all_frames, fps=FPS)
        print(f"  Video saved  : {video_path}")

    return sr


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

SCENARIOS = ["black_table", "green_cube", "init_noise", "wide_range"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy_path",
        default="resfit/my_lerobot_data/bc_run_2026-07-30_16-20-35_lane_lift_id_20_aligned_diffusion/policy_step_5000/policy",
    )
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--out_dir",    default="eval_videos_base_policy")
    parser.add_argument("--scenarios",  nargs="+", default=SCENARIOS, choices=SCENARIOS)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading policy : {args.policy_path}")
    policy = load_policy(Path(args.policy_path))
    policy.eval().to(device)

    results = {}
    for scenario in args.scenarios:
        video_path = os.path.join(args.out_dir, f"base_policy_{scenario}.mp4")
        sr = run_scenario(scenario, policy, args.n_episodes, video_path, device)
        results[scenario] = sr

    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    for sc, sr in results.items():
        print(f"  {sc:<20} : {sr*100:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()
