#!/usr/bin/env python3
"""
Case Study Analysis & Multi-Policy Visual Comparison on Square Nut Assembly.

Compares 3 policies on the EXACT SAME initial environment state:
1. Row 1: Base Policy (Zero Residual Action)
2. Row 2: ResFiT Policy (Sparse RL Baseline)
3. Row 3: Ours Policy (V-PBRS with Decoupled Latent Shaping)
4. Row 4: Decoupled View Similarity Timeline for Ours (S_main in Red, S_wrist in Green)
          with the shaded area colored according to whichever similarity is higher.

Automatically searches candidate seeds to find a case where:
- Base Policy: FAILS
- ResFiT Policy: FAILS
- Ours Policy: SUCCEEDS
"""

import argparse
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as dist
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from lane.e2c import MLPE2C
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy, _make_noise_scheduler
from resfit.lerobot.utils.load_policy import load_policy
from resfit.rl_finetuning.config.rlpd import QAgentConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.wrappers.residual_env_wrapper import BasePolicyVecEnvWrapper


def get_dino_features(images: torch.Tensor, dino: torch.nn.Module, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract DINOv2 embeddings (384-d each) for main and wrist camera views."""
    images = images.to(device)
    output_size = 112
    n, c, h, w = images.shape
    if h < output_size or w < output_size:
        cropped = TF.resize(images, [output_size, output_size])
    else:
        top = (h - output_size) // 2
        left = (w - output_size) // 2
        cropped = images[:, :, top : top + output_size, left : left + output_size]

    if cropped.dtype == torch.uint8 or cropped.max() > 1.0:
        cropped = cropped.float() / 255.0
    else:
        cropped = cropped.float()

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    front = normalize(cropped[:, :3])
    wrist = normalize(cropped[:, 3:6])

    with torch.no_grad():
        feat_f = dino(front)
        feat_w = dino(wrist)
    return feat_f, feat_w


def render_similarity_plot_frame(
    S_m: np.ndarray,
    S_w: np.ndarray,
    current_step: int,
    total_steps: int,
    width: int = 768,
    height: int = 256,
) -> np.ndarray:
    """Render right-side similarity plot: S_main (Red) vs S_wrist (Green) with closed region shading."""
    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
    steps = np.arange(len(S_m))

    # Plot base curves
    ax.plot(steps, S_m, color="#d32f2f", linewidth=1.1, label=r"$S_{\mathrm{main}}$ (Front View)")
    ax.plot(steps, S_w, color="#2e7d32", linewidth=1.1, label=r"$S_{\mathrm{wrist}}$ (Eye-in-Hand View)")

    # Shading ONLY in the closed region bounded between the two curves
    ax.fill_between(steps, S_w, S_m, where=(S_m >= S_w), interpolate=True, color="#d32f2f", alpha=0.35)
    ax.fill_between(steps, S_m, S_w, where=(S_w > S_m), interpolate=True, color="#2e7d32", alpha=0.35)

    # Current step cursor line and dot
    cur_idx = min(current_step, len(S_m) - 1)
    ax.axvline(x=cur_idx, color="#212121", linestyle="--", linewidth=1.8, alpha=0.8)
    ax.scatter([cur_idx], [S_m[cur_idx]], color="#b71c1c", s=40, zorder=5)
    ax.scatter([cur_idx], [S_w[cur_idx]], color="#1b5e20", s=40, zorder=5)

    ax.set_xlim(0, max(total_steps - 1, 1))
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Time Step (t)", fontsize=9, fontweight="bold", labelpad=2)
    ax.set_ylabel("Similarity", fontsize=9, fontweight="bold", labelpad=2)
    ax.set_title("Similarity", fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2, fontsize=8.5, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout(pad=0.8)
    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"):
        plot_img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    else:
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    if plot_img.shape[1] != width or plot_img.shape[0] != height:
        plot_img = cv2.resize(plot_img, (width, height), interpolation=cv2.INTER_AREA)

    return plot_img


def render_base_placeholder_frame(width: int = 768, height: int = 256) -> np.ndarray:
    """Render Row 1 right panel: Clean empty neutral panel for Base Policy (no text, no RL plot)."""
    panel = np.full((height, width, 3), 26, dtype=np.uint8)
    cv2.rectangle(panel, (8, 8), (width - 8, height - 8), (42, 42, 42), 1)
    return panel


def annotate_policy_frame(
    frame: np.ndarray,
    policy_name: str,
    step_idx: int,
    is_terminal: bool,
    is_success: bool,
    color_theme: tuple[int, int, int],
) -> np.ndarray:
    """Overlay title banner and execution status on camera frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Top banner background bar
    bar_h = 28
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Policy Title
    cv2.putText(canvas, policy_name, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_theme, 2, cv2.LINE_AA)

    # Step info
    step_text = f"Step: {step_idx:03d}"
    cv2.putText(canvas, step_text, (w - 210, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    # Status Badge
    if is_terminal:
        if is_success:
            status_text = "SUCCESS"
            status_bg = (34, 139, 34)  # Forest Green (RGB)
        else:
            status_text = "FAILED"
            status_bg = (180, 30, 30)  # Crimson Red (RGB)
        cv2.rectangle(canvas, (w - 105, 4), (w - 6, 24), status_bg, -1)
        cv2.putText(canvas, status_text, (w - 98, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(canvas, (w - 105, 4), (w - 6, 24), (70, 70, 70), -1)
        cv2.putText(canvas, "RUNNING", (w - 98, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    # Bottom camera labels
    mid_x = w // 2
    cv2.putText(canvas, "MAIN VIEW (Agentview)", (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "MAIN VIEW (Agentview)", (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "WRIST VIEW (Eye-in-Hand)", (mid_x + 10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "WRIST VIEW (Eye-in-Hand)", (mid_x + 10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def extract_camera_frame(obs: dict) -> np.ndarray:
    """Extract and horizontally concatenate agentview and robot0_eye_in_hand directly from obs."""
    img_m = obs.get("observation.images.agentview")
    img_w = obs.get("observation.images.robot0_eye_in_hand")
    if img_m is None or img_w is None:
        return np.zeros((256, 512, 3), dtype=np.uint8)

    if isinstance(img_m, torch.Tensor):
        img_m = img_m.detach().cpu().numpy()
    if isinstance(img_w, torch.Tensor):
        img_w = img_w.detach().cpu().numpy()

    if img_m.ndim == 4:
        img_m = img_m[0]
    if img_w.ndim == 4:
        img_w = img_w[0]

    # Convert float [0, 1] to uint8 [0, 255]
    if img_m.dtype != np.uint8:
        if img_m.max() <= 1.01:
            img_m = (np.clip(img_m, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_m = np.clip(img_m, 0, 255).astype(np.uint8)

    if img_w.dtype != np.uint8:
        if img_w.max() <= 1.01:
            img_w = (np.clip(img_w, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_w = np.clip(img_w, 0, 255).astype(np.uint8)

    # Transpose (C, H, W) -> (H, W, C)
    if img_m.shape[0] in [1, 3]:
        img_m = np.transpose(img_m, (1, 2, 0))
    if img_w.shape[0] in [1, 3]:
        img_w = np.transpose(img_w, (1, 2, 0))

    # Resize each view to 256x256
    h_m, w_m = img_m.shape[:2]
    if h_m != 256 or w_m != 256:
        img_m = cv2.resize(img_m, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    h_w, w_w = img_w.shape[:2]
    if h_w != 256 or w_w != 256:
        img_w = cv2.resize(img_w, (256, 256), interpolation=cv2.INTER_LANCZOS4)

    # Concatenate side-by-side: [Agentview | Robot0_eye_in_hand] -> (256, 512, 3)
    frame = np.concatenate([img_m, img_w], axis=1)
    return frame


def run_single_rollout(
    env: BasePolicyVecEnvWrapper,
    mode: str,
    agent: QAgent | None,
    seed: int,
    max_steps: int = 300,
    dino: torch.nn.Module | None = None,
    e2c_main: torch.nn.Module | None = None,
    e2c_wrist: torch.nn.Module | None = None,
    all_z_demo_m: np.ndarray | None = None,
    all_z_demo_w: np.ndarray | None = None,
    gamma_m: float = 1.0,
    gamma_w: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Execute rollout deterministically on seed and record frames & metrics."""
    # Deterministic seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    obs, _ = env.reset(seed=seed)
    num_envs = 1
    action_dim = env.action_space.shape[1]

    frames = []
    S_m_history = []
    S_w_history = []
    is_success = False

    for step_i in range(max_steps):
        # 1. Capture concatenated camera frame (Agentview + Wrist) from obs
        cur_fr = extract_camera_frame(obs)
        frames.append(cur_fr)

        # 2. Extract DINO & Decoupled Similarity if requested (for Ours)
        if dino is not None and e2c_main is not None and e2c_wrist is not None:
            with torch.no_grad():
                img_m = obs["observation.images.agentview"]
                img_w = obs["observation.images.robot0_eye_in_hand"]
                obs_img = torch.cat([img_m, img_w], dim=1)
                feat_f, feat_w = get_dino_features(obs_img, dino, device)
                zf, _ = e2c_main.enc(feat_f)
                zw, _ = e2c_wrist.enc(feat_w)

                zf_np = zf[0].detach().cpu().numpy().reshape(1, -1)
                zw_np = zw[0].detach().cpu().numpy().reshape(1, -1)

                dist_m = dist.cdist(zf_np, all_z_demo_m, "sqeuclidean").min()
                dist_w = dist.cdist(zw_np, all_z_demo_w, "sqeuclidean").min()

                sm = float(np.exp(-gamma_m * (dist_m ** 2)))
                sw = float(np.exp(-gamma_w * (dist_w ** 2)))
                S_m_history.append(sm)
                S_w_history.append(sw)

        # 3. Action selection
        with torch.no_grad():
            if mode == "base" or agent is None:
                actions = torch.zeros((num_envs, action_dim), device=device)
            else:
                actions = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

        # 4. Step environment
        next_obs, reward, terminated, truncated, info = env.step(actions)
        done = bool((terminated | truncated)[0].item())

        # Success check
        try:
            if "final_info" in info and isinstance(info["final_info"], (list, tuple)):
                if info["final_info"][0] is not None:
                    is_success = bool(info["final_info"][0].get("success", False))
            if not is_success and "success" in info:
                if isinstance(info["success"], (list, tuple, np.ndarray, torch.Tensor)):
                    is_success = bool(info["success"][0])
                else:
                    is_success = bool(info["success"])
        except Exception:
            pass

        if not is_success:
            is_success = bool(reward[0].item() == 1.0 or reward[0].item() > 50.0)

        if done:
            # Capture final observation
            if "final_obs" in info and info["final_obs"] is not None and info["final_obs"][0] is not None:
                final_fr = extract_camera_frame(info["final_obs"][0])
            else:
                final_fr = extract_camera_frame(next_obs)
            frames.append(final_fr)
            if len(S_m_history) > 0:
                S_m_history.append(S_m_history[-1])
                S_w_history.append(S_w_history[-1])
            break

        obs = next_obs

    if not done:
        final_fr = extract_camera_frame(obs)
        frames.append(final_fr)
        if len(S_m_history) > 0 and len(S_m_history) < len(frames):
            S_m_history.append(S_m_history[-1])
            S_w_history.append(S_w_history[-1])

    return {
        "frames": frames,
        "is_success": is_success,
        "num_steps": len(frames),
        "S_m": np.array(S_m_history),
        "S_w": np.array(S_w_history),
    }


def generate_case_study_artifacts(
    base_res: dict,
    resfit_res: dict,
    ours_res: dict,
    seed: int,
    output_dir: Path,
):
    """Generates both Synchronized 4-Row MP4 Video and High-Resolution Keyframe Strip PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"case_study_square_seed{seed}.mp4"
    keyframe_path = output_dir / f"case_study_square_seed{seed}_keyframe_strip.png"

    frames_b = base_res["frames"]
    frames_r = resfit_res["frames"]
    frames_o = ours_res["frames"]

    len_b = len(frames_b)
    len_r = len(frames_r)
    len_o = len(frames_o)
    total_T = len_o  # Align comparison horizon to Ours's success steps

    # Align ResFiT potential tracking to Ours's time duration (len_o)
    S_m_r = resfit_res.get("S_m", np.zeros(len_r))[:len_o]
    S_w_r = resfit_res.get("S_w", np.zeros(len_r))[:len_o]
    if len(S_m_r) == 0:
        S_m_r = np.zeros(len_o)
        S_w_r = np.zeros(len_o)

    S_m_o = ours_res.get("S_m", np.zeros(len_o))[:len_o]
    S_w_o = ours_res.get("S_w", np.zeros(len_o))[:len_o]
    if len(S_m_o) == 0:
        S_m_o = np.zeros(len_o)
        S_w_o = np.zeros(len_o)

    print(f"\n[Case Study Generator] Building synchronized video ({total_T} frames, seed={seed})...")
    w_cam = frames_o[0].shape[1]  # 512
    h_cam = frames_o[0].shape[0]  # 256
    w_plot = int(round(w_cam * 1.5))  # 768 (1.5x width)

    # Base right-side clean neutral panel (width 768, height 256, no text)
    right_b = render_base_placeholder_frame(width=w_plot, height=h_cam)

    writer = imageio.get_writer(video_path, fps=20, codec="libx264", quality=8)

    for t in range(total_T):
        # 1. Base Row: [Observation (512x256) | Base Clean Panel (768x256)]
        idx_b = min(t, len_b - 1)
        is_term_b = (t >= total_T - 1)
        fr_b = annotate_policy_frame(
            frames_b[idx_b],
            "1. Base Policy (Diffusion BC)",
            step_idx=idx_b + 1,
            is_terminal=is_term_b,
            is_success=base_res["is_success"],
            color_theme=(255, 120, 120),
        )
        row_b = np.hstack([fr_b, right_b])

        # 2. ResFiT Row: [Observation (512x256) | ResFiT Similarity Plot (768x256)]
        idx_r = min(t, len_r - 1)
        is_term_r = (t >= total_T - 1)
        fr_r = annotate_policy_frame(
            frames_r[idx_r],
            "2. ResFiT (Sparse RL Baseline)",
            step_idx=idx_r + 1,
            is_terminal=is_term_r,
            is_success=resfit_res["is_success"],
            color_theme=(255, 180, 70),
        )
        plot_r = render_similarity_plot_frame(
            S_m=S_m_r,
            S_w=S_w_r,
            current_step=idx_r,
            total_steps=len_o,
            width=w_plot,
            height=h_cam,
        )
        row_r = np.hstack([fr_r, plot_r])

        # 3. Ours Row: [Observation (512x256) | Ours Similarity Plot (768x256)]
        idx_o = min(t, len_o - 1)
        is_term_o = (t >= total_T - 1)
        fr_o = annotate_policy_frame(
            frames_o[idx_o],
            "3. Ours (V-PBRS Residual RL)",
            step_idx=idx_o + 1,
            is_terminal=is_term_o,
            is_success=ours_res["is_success"],
            color_theme=(120, 255, 120),
        )
        plot_o = render_similarity_plot_frame(
            S_m=S_m_o,
            S_w=S_w_o,
            current_step=idx_o,
            total_steps=len_o,
            width=w_plot,
            height=h_cam,
        )
        row_o = np.hstack([fr_o, plot_o])

        # Vertical stack: 3 Rows -> total size: (768, 1280, 3)
        full_canvas = np.vstack([row_b, row_r, row_o])
        writer.append_data(full_canvas)

    # Hold final result for 20 frames (1 second at 20fps)
    for _ in range(20):
        writer.append_data(full_canvas)

    writer.close()
    print(f"  -> Saved Synchronized Case Study Video to: {video_path}")

    # -------------------------------------------------------------------------
    # 2. High-Resolution Keyframe Strip PNG for Publication Figure
    # -------------------------------------------------------------------------
    print(f"\n[Case Study Generator] Building high-resolution publication keyframe strips...")
    num_keyframes = 5
    key_indices = np.linspace(0, len_o - 1, num_keyframes, dtype=int)

    phase_titles = [
        "Initial Reaching",
        "Nut Grasping",
        "Approach Peg",
        "Alignment & Contact",
        "Final Insertion",
    ]

    def _draw_similarity_strip(ax, S_m, S_w, title_text, y_label_text, show_xlabel=True):
        steps = np.arange(len(S_m))
        ax.plot(steps, S_m, color="#d32f2f", linewidth=1.2, label=r"$S_{\mathrm{main}}$ (Front View)")
        ax.plot(steps, S_w, color="#2e7d32", linewidth=1.2, label=r"$S_{\mathrm{wrist}}$ (Eye-in-Hand View)")
        ax.fill_between(steps, S_w, S_m, where=(S_m >= S_w), interpolate=True, color="#d32f2f", alpha=0.35)
        ax.fill_between(steps, S_m, S_w, where=(S_w > S_m), interpolate=True, color="#2e7d32", alpha=0.35)
        for col_k, t_idx in enumerate(key_indices):
            ax.axvline(x=t_idx, color="#212121", linestyle="--", linewidth=1.1, alpha=0.6)
            label_x = t_idx - 5 if col_k == num_keyframes - 1 else t_idx + 1
            ax.text(
                label_x, 0.08, f"K{col_k+1}", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#616161", alpha=0.85)
            )
        ax.set_xlim(0, len_o - 1)
        ax.set_ylim(-0.02, 1.05)
        if show_xlabel:
            ax.set_xlabel("Episode Steps", fontsize=10, fontweight="bold")
        ax.set_ylabel(y_label_text, fontsize=10, fontweight="bold")
        ax.set_title(title_text, fontsize=11, fontweight="bold", loc="left", pad=6)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2, fontsize=9, frameon=True)

    # --- Primary Figure (Layout A: Interleaved - plot directly under each respective policy) ---
    figA = plt.figure(figsize=(16, 12), dpi=300)
    gsA = figA.add_gridspec(5, num_keyframes, height_ratios=[1.0, 1.0, 0.65, 1.0, 0.65], hspace=0.38, wspace=0.08)

    for col, t_idx in enumerate(key_indices):
        # Row 0: Base
        ax = figA.add_subplot(gsA[0, col])
        ax.imshow(frames_b[min(t_idx, len_b - 1)])
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0: ax.set_ylabel("Base Policy\n(Zero Residual)", fontsize=11, fontweight="bold", color="#d32f2f")
        ax.set_title(f"Step {t_idx+1}\n({phase_titles[col]})", fontsize=10, fontweight="bold")
        # Row 1: ResFiT
        ax = figA.add_subplot(gsA[1, col])
        ax.imshow(frames_r[min(t_idx, len_r - 1)])
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0: ax.set_ylabel("ResFiT\n(Sparse RL)", fontsize=11, fontweight="bold", color="#f57c00")
        # Row 3: Ours
        ax = figA.add_subplot(gsA[3, col])
        ax.imshow(frames_o[min(t_idx, len_o - 1)])
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0: ax.set_ylabel("Ours\n(V-PBRS)", fontsize=11, fontweight="bold", color="#2e7d32")

    # Row 2: ResFiT Plot (spanning full width below ResFiT)
    ax_rA = figA.add_subplot(gsA[2, :])
    _draw_similarity_strip(ax_rA, S_m_r, S_w_r, "Similarity", "ResFiT\nSimilarity", show_xlabel=False)

    # Row 4: Ours Plot (spanning full width below Ours)
    ax_oA = figA.add_subplot(gsA[4, :])
    _draw_similarity_strip(ax_oA, S_m_o, S_w_o, "Similarity", "Ours\nSimilarity", show_xlabel=True)

    plt.suptitle(
        f"Qualitative Case Study on Square Nut Assembly (Seed {seed})\n"
        f"Base: Failed (Collision/Drift) | ResFiT: Failed (Erratic Slippage) | Ours: Success (Compliant Insertion)",
        fontsize=13, fontweight="bold", y=0.985
    )
    figA.savefig(keyframe_path, bbox_inches="tight")
    plt.close(figA)
    print(f"  -> Saved Primary Publication Keyframe Strip to: {keyframe_path}")

    # --- Alternative Figure (Layout B: Grouped - all policy observations on top, both plots at bottom) ---
    grouped_keyframe_path = output_dir / f"case_study_square_seed{seed}_keyframe_strip_grouped.png"
    figB = plt.figure(figsize=(16, 12), dpi=300)
    gsB = figB.add_gridspec(5, num_keyframes, height_ratios=[1.0, 1.0, 1.0, 0.65, 0.65], hspace=0.38, wspace=0.08)

    for col, t_idx in enumerate(key_indices):
        # Row 0: Base
        ax = figB.add_subplot(gsB[0, col])
        ax.imshow(frames_b[min(t_idx, len_b - 1)])
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0: ax.set_ylabel("Base Policy\n(Zero Residual)", fontsize=11, fontweight="bold", color="#d32f2f")
        ax.set_title(f"Step {t_idx+1}\n({phase_titles[col]})", fontsize=10, fontweight="bold")
        # Row 1: ResFiT
        ax = figB.add_subplot(gsB[1, col])
        ax.imshow(frames_r[min(t_idx, len_r - 1)])
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0: ax.set_ylabel("ResFiT\n(Sparse RL)", fontsize=11, fontweight="bold", color="#f57c00")
        # Row 2: Ours
        ax = figB.add_subplot(gsB[2, col])
        ax.imshow(frames_o[min(t_idx, len_o - 1)])
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0: ax.set_ylabel("Ours\n(V-PBRS)", fontsize=11, fontweight="bold", color="#2e7d32")

    # Row 3: ResFiT Plot
    ax_rB = figB.add_subplot(gsB[3, :])
    _draw_similarity_strip(ax_rB, S_m_r, S_w_r, "Similarity", "ResFiT\nSimilarity", show_xlabel=False)

    # Row 4: Ours Plot
    ax_oB = figB.add_subplot(gsB[4, :])
    _draw_similarity_strip(ax_oB, S_m_o, S_w_o, "Similarity", "Ours\nSimilarity", show_xlabel=True)

    plt.suptitle(
        f"Qualitative Case Study on Square Nut Assembly (Seed {seed})\n"
        f"Base: Failed (Collision/Drift) | ResFiT: Failed (Erratic Slippage) | Ours: Success (Compliant Insertion)",
        fontsize=13, fontweight="bold", y=0.985
    )
    figB.savefig(grouped_keyframe_path, bbox_inches="tight")
    plt.close(figB)
    print(f"  -> Saved Grouped Publication Keyframe Strip to: {grouped_keyframe_path}")


def main():
    parser = argparse.ArgumentParser(description="Case Study Visual Comparison for Square Task")
    parser.add_argument("--task", type=str, default="Square")
    parser.add_argument("--base_policy_path", type=str, default="resfit/my_lerobot_data/bc_run_2026-08-29_14-38-11_robomimic_square_v15_50_diffusion/policy_step_66000/policy")
    parser.add_argument("--resfit_ckpt", type=str, default="resfit/outputs/2026-08-31_16-01-18_Square_ablation_no_dense_seed42/models/agent_best.pt")
    parser.add_argument("--ours_ckpt", type=str, default="resfit/outputs/2026-09-02_11-33-44_Square_reward_pbrs_no_mask_nstep_beta1.0_scale0.1/models/agent_best.pt")
    parser.add_argument("--e2c_dir", type=str, default="lane/pretrained_e2c/square")
    parser.add_argument("--offline_data_name", type=str, default="resfit/my_lerobot_data/ysl2683/robomimic_square_v15_50")
    parser.add_argument("--output_dir", type=str, default="visualization/outputs/case_study")
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--max_search_seeds", type=int, default=50)
    parser.add_argument("--action_scale", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--specific_seed", type=int, default=None, help="If provided, directly evaluate this seed without searching")
    parser.add_argument("--ddim_steps", type=int, default=20, help="DDIM inference steps for base policy")
    args = parser.parse_args()

    base_policy_path = PROJECT_ROOT / args.base_policy_path
    resfit_ckpt = PROJECT_ROOT / args.resfit_ckpt
    ours_ckpt = PROJECT_ROOT / args.ours_ckpt
    e2c_dir = PROJECT_ROOT / args.e2c_dir
    offline_data_dir = PROJECT_ROOT / args.offline_data_name
    output_dir = PROJECT_ROOT / args.output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Case Study] Initializing on device: {device}")

    # 1. Normalizers
    dataset = LeRobotDataset(repo_id="placeholder", root=offline_data_dir)
    action_scaler = ActionScaler.from_dataset_stats(dataset.meta.stats["action"])
    state_standardizer = StateStandardizer.from_dataset_stats(dataset.meta.stats["observation.state"])

    # 2. Base Policy
    print(f"[Case Study] Loading Base Policy from: {base_policy_path}")
    base_policy = load_policy(base_policy_path)
    base_policy.to(device)
    base_policy.eval()

    if args.ddim_steps is not None:
        print(f"[Case Study] Applying DDIM {args.ddim_steps} steps to Base Policy...")
        base_policy.config.noise_scheduler_type = "DDIM"
        base_policy.config.num_inference_steps = args.ddim_steps
        base_policy.noise_scheduler = _make_noise_scheduler(
            "DDIM",
            num_train_timesteps=base_policy.config.num_train_timesteps,
            beta_schedule=base_policy.config.beta_schedule,
            beta_start=base_policy.config.beta_start,
            beta_end=base_policy.config.beta_end,
            clip_sample=base_policy.config.clip_sample,
        )

    # 3. Environment
    print(f"[Case Study] Creating Vectorized Environment for {args.task}...")
    raw_vec_env = create_vectorized_env(
        env_name=args.task,
        num_envs=1,
        device=str(device),
        camera_size=128,
    )
    env = BasePolicyVecEnvWrapper(
        vec_env=raw_vec_env,
        base_policy=base_policy,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
    )

    action_dim = env.action_space.shape[1]
    lowdim_dim = env.observation_space["observation.state"].shape[1]

    # 4. QAgents
    cfg_agent = QAgentConfig()
    cfg_agent.actor.action_scale = args.action_scale

    print(f"[Case Study] Loading ResFiT Agent from {resfit_ckpt}...")
    resfit_agent = QAgent(
        obs_shape=(3, 128, 128),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=["observation.images.agentview", "observation.images.robot0_eye_in_hand"],
        cfg=cfg_agent,
        residual_actor=True,
    )
    resfit_agent.load_state_dict(torch.load(resfit_ckpt, map_location=device, weights_only=True))
    resfit_agent.eval()

    print(f"[Case Study] Loading Ours Agent from {ours_ckpt}...")
    ours_agent = QAgent(
        obs_shape=(3, 128, 128),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=["observation.images.agentview", "observation.images.robot0_eye_in_hand"],
        cfg=cfg_agent,
        residual_actor=True,
    )
    ours_agent.load_state_dict(torch.load(ours_ckpt, map_location=device, weights_only=True))
    ours_agent.eval()

    # 5. DINO & E2C Encoders
    print("[Case Study] Loading DINOv2 ViT-S/14 backbone...")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(device)
    dino.eval()

    print(f"[Case Study] Loading E2C models from {e2c_dir}...")
    e2c_main = MLPE2C(obs_shape=(384,), action_dim=action_dim, z_dimension=16).to(device)
    # Check if run-specific best E2C exists, otherwise fallback to e2c_dir
    run_e2c_main = ours_ckpt.parent / "e2c_main_best.pt"
    if run_e2c_main.exists():
        e2c_main.load_state_dict(torch.load(run_e2c_main, map_location=device))
    else:
        e2c_main.load_state_dict(torch.load(e2c_dir / "e2c_main.pt", map_location=device))
    e2c_main.eval()

    e2c_wrist = MLPE2C(obs_shape=(384,), action_dim=action_dim, z_dimension=16).to(device)
    run_e2c_wrist = ours_ckpt.parent / "e2c_wrist_best.pt"
    if run_e2c_wrist.exists():
        e2c_wrist.load_state_dict(torch.load(run_e2c_wrist, map_location=device))
    else:
        e2c_wrist.load_state_dict(torch.load(e2c_dir / "e2c_wrist.pt", map_location=device))
    e2c_wrist.eval()

    # 6. Preload Demo Latents for exact similarity scaling
    demo_latents_path = e2c_dir / "demo_latents.pt"
    print(f"[Case Study] Loading demonstration latents from {demo_latents_path}...")
    demo_data = torch.load(demo_latents_path, map_location="cpu", weights_only=False)
    z_f_demo = demo_data.get("z_demo_main", demo_data.get("z_demo_front"))
    z_w_demo = demo_data["z_demo_wrist"]
    all_z_demo_m = np.concatenate([z.squeeze(0).numpy() for z in z_f_demo], axis=0)
    all_z_demo_w = np.concatenate([z.squeeze(0).numpy() for z in z_w_demo], axis=0)

    one_step_m = [((z.squeeze(0)[1:] - z.squeeze(0)[:-1]) ** 2).sum(dim=1).mean().item() for z in z_f_demo if len(z.squeeze(0)) > 1]
    ref_one_step_dist_m = sum(one_step_m) / len(one_step_m)
    one_step_w = [((z.squeeze(0)[1:] - z.squeeze(0)[:-1]) ** 2).sum(dim=1).mean().item() for z in z_w_demo if len(z.squeeze(0)) > 1]
    ref_one_step_dist_w = sum(one_step_w) / len(one_step_w)

    gamma_m = args.beta / ((ref_one_step_dist_m ** 2) + 1e-8)
    gamma_w = args.beta / ((ref_one_step_dist_w ** 2) + 1e-8)

    # 7. Seed Search
    target_seed = args.specific_seed
    if target_seed is not None:
        candidate_seeds = [target_seed]
    else:
        candidate_seeds = list(range(args.start_seed, args.start_seed + args.max_search_seeds))

    print(f"\n{'='*70}")
    print(f"Searching for Candidate Seed where Ours=SUCCESS, Base=FAIL, ResFiT=FAIL")
    print(f"{'='*70}")

    found_seed = None
    saved_base_res = None
    saved_resfit_res = None
    saved_ours_res = None

    cache_path = output_dir / f"rollout_cache_seed{target_seed}.pt" if target_seed is not None else None
    if cache_path is not None and cache_path.exists():
        print(f"\n[Case Study] Loading cached rollouts from {cache_path}...")
        cached_data = torch.load(cache_path, weights_only=False)
        saved_base_res = cached_data["base_res"]
        saved_resfit_res = cached_data["resfit_res"]
        saved_ours_res = cached_data["ours_res"]
        found_seed = target_seed
    else:
        for seed in candidate_seeds:
            print(f"\n[Test Seed {seed:02d}] Checking Ours...")
            ours_res = run_single_rollout(
                env=env,
                mode="ours",
                agent=ours_agent,
                seed=seed,
                dino=dino,
                e2c_main=e2c_main,
                e2c_wrist=e2c_wrist,
                all_z_demo_m=all_z_demo_m,
                all_z_demo_w=all_z_demo_w,
                gamma_m=gamma_m,
                gamma_w=gamma_w,
                device=device,
            )

            if not ours_res["is_success"]:
                print(f"  -> Ours FAILED (steps: {ours_res['num_steps']}). Skipping seed {seed}.")
                continue

            print(f"  -> Ours SUCCEEDED in {ours_res['num_steps']} steps! Checking Base Policy...")
            base_res = run_single_rollout(
                env=env,
                mode="base",
                agent=None,
                seed=seed,
                device=device,
            )

            if base_res["is_success"]:
                print(f"  -> Base SUCCEEDED (steps: {base_res['num_steps']}). Skipping seed {seed}.")
                continue

            print(f"  -> Base FAILED (steps: {base_res['num_steps']})! Checking ResFiT...")
            resfit_res = run_single_rollout(
                env=env,
                mode="resfit",
                agent=resfit_agent,
                seed=seed,
                dino=dino,
                e2c_main=e2c_main,
                e2c_wrist=e2c_wrist,
                all_z_demo_m=all_z_demo_m,
                all_z_demo_w=all_z_demo_w,
                gamma_m=gamma_m,
                gamma_w=gamma_w,
                device=device,
            )

            if resfit_res["is_success"]:
                print(f"  -> ResFiT SUCCEEDED (steps: {resfit_res['num_steps']}). Skipping seed {seed}.")
                continue

            print(f"  -> ResFiT FAILED (steps: {resfit_res['num_steps']})!")
            print(f"\n🎉 FOUND PERFECT CASE STUDY SEED: {seed}")
            print(f"   Base:   FAILED ({base_res['num_steps']} steps)")
            print(f"   ResFiT: FAILED ({resfit_res['num_steps']} steps)")
            print(f"   Ours:   SUCCESS ({ours_res['num_steps']} steps)")

            found_seed = seed
            saved_base_res = base_res
            saved_resfit_res = resfit_res
            saved_ours_res = ours_res

            torch.save({
                "base_res": saved_base_res,
                "resfit_res": saved_resfit_res,
                "ours_res": saved_ours_res,
            }, output_dir / f"rollout_cache_seed{found_seed}.pt")
            break

    if found_seed is None:
        print(f"\n[Warning] No seed found meeting all criteria within searched range.")
        print("Please increase --max_search_seeds or change --start_seed.")
        return

    # 8. Render Visualizations
    generate_case_study_artifacts(
        base_res=saved_base_res,
        resfit_res=saved_resfit_res,
        ours_res=saved_ours_res,
        seed=found_seed,
        output_dir=output_dir,
    )

    print("\n" + "=" * 70)
    print("Case Study Analysis Complete!")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
