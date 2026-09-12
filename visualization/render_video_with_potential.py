#!/usr/bin/env python3
"""
Render Synchronized Case Study Video with Decoupled Similarity, Potential, and 1-Step PBRS Reward.

Visualizes:
- Row 1: Base Policy (Diffusion BC)
- Row 2: ResFiT (Sparse RL) with Visual Similarity (Top) + Potential & 1-Step PBRS Reward (Bottom)
- Row 3: Ours (V-PBRS) with Visual Similarity (Top) + Potential & 1-Step PBRS Reward (Bottom)
"""

import os
import sys
from pathlib import Path
import cv2
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualization.case_study_analysis import annotate_policy_frame, render_base_placeholder_frame


def render_stacked_plot_frame(
    S_m: np.ndarray,
    S_w: np.ndarray,
    Phi: np.ndarray,
    F_t: np.ndarray,
    current_step: int,
    total_steps: int,
    width: int = 768,
    height: int = 256,
    policy_name: str = "ResFiT",
) -> np.ndarray:
    """Renders 2-row stacked plot: Top=Similarity (S_m, S_w), Bottom=Potential (Phi) & 1-Step Shaping Reward (F_t)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width / 100.0, height / 100.0), dpi=100, sharex=True)
    fig.patch.set_facecolor("#ffffff")
    steps = np.arange(len(S_m))
    cur_idx = min(current_step, len(S_m) - 1)

    # -------------------------------------------------------------
    # 1. Top Subplot: Visual Similarity (S_m in Red, S_w in Green)
    # -------------------------------------------------------------
    ax1.plot(steps, S_m, color="#d32f2f", linewidth=1.1, label=r"$S_{\mathrm{main}}$ (Front)")
    ax1.plot(steps, S_w, color="#2e7d32", linewidth=1.1, label=r"$S_{\mathrm{wrist}}$ (Wrist)")
    ax1.fill_between(steps, S_w, S_m, where=(S_m >= S_w), interpolate=True, color="#d32f2f", alpha=0.30)
    ax1.fill_between(steps, S_m, S_w, where=(S_w > S_m), interpolate=True, color="#2e7d32", alpha=0.30)

    # Cursor & marker dots
    ax1.axvline(x=cur_idx, color="#212121", linestyle="--", linewidth=1.5, alpha=0.85)
    ax1.scatter([cur_idx], [S_m[cur_idx]], color="#b71c1c", s=25, zorder=5)
    ax1.scatter([cur_idx], [S_w[cur_idx]], color="#1b5e20", s=25, zorder=5)

    ax1.set_ylim(-0.02, 1.05)
    ax1.set_ylabel("Similarity", fontsize=8, fontweight="bold", labelpad=2)
    ax1.set_title(f"{policy_name}: Visual Similarity (Front $S_m$ vs Wrist $S_w$)", fontsize=8.5, fontweight="bold", loc="left", pad=3)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02), ncol=2, fontsize=7.0, frameon=True, framealpha=0.85)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # -------------------------------------------------------------
    # 2. Bottom Subplot: Potential (Phi) & 1-Step Shaping Reward (F_t)
    # -------------------------------------------------------------
    colors = ["#1976d2" if v >= 0 else "#e53935" for v in F_t]
    ax2.plot(steps, Phi, color="#6a1b9a", linewidth=1.2, label=r"Potential $\Phi$")
    ax2.bar(
        steps[: cur_idx + 1],
        F_t[: cur_idx + 1],
        width=1.0,
        color=colors[: cur_idx + 1],
        alpha=0.60,
        label=r"1-step PBRS $F_t$",
    )

    ax2.axhline(0, color="#616161", linestyle="-", linewidth=0.8, alpha=0.7)
    ax2.axvline(x=cur_idx, color="#212121", linestyle="--", linewidth=1.5, alpha=0.85)
    ax2.scatter([cur_idx], [Phi[cur_idx]], color="#4a148c", s=25, zorder=5)

    ax2.set_xlim(0, max(total_steps - 1, 1))
    ax2.set_ylim(-0.65, 1.05)
    ax2.set_ylabel(r"$\Phi$ / $F_t$", fontsize=8, fontweight="bold", labelpad=2)
    ax2.set_xlabel("Time Step (t)", fontsize=8, fontweight="bold", labelpad=2)
    ax2.set_title(r"Comprehensive Potential $\Phi$ & 1-Step Shaping Reward $F_t = \gamma\Phi^\prime - \Phi$", fontsize=8.5, fontweight="bold", loc="left", pad=3)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02), ncol=2, fontsize=7.0, frameon=True, framealpha=0.85)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout(pad=0.5, h_pad=0.35)
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


def main():
    cache_path = PROJECT_ROOT / "visualization/outputs/case_study/rollout_cache_seed7.pt"
    if not cache_path.exists():
        print(f"Error: Cache path {cache_path} does not exist.")
        sys.exit(1)

    print(f"Loading cached rollout from {cache_path}...")
    data = torch.load(cache_path, map_location="cpu", weights_only=False)

    base_res = data["base_res"]
    resfit_res = data["resfit_res"]
    ours_res = data["ours_res"]

    frames_b = base_res["frames"]
    frames_r = resfit_res["frames"]
    frames_o = ours_res["frames"]

    len_b = len(frames_b)
    len_r = len(frames_r)
    len_o = len(frames_o)
    total_T = len_o  # 147 steps

    # Extract similarities aligned to total_T
    sm_r = resfit_res["S_m"][:total_T]
    sw_r = resfit_res["S_w"][:total_T]
    sm_o = ours_res["S_m"][:total_T]
    sw_o = ours_res["S_w"][:total_T]

    # Algorithm hyperparameters
    w_m, w_w = 0.3, 0.7
    gamma = 0.99

    # 1. ResFiT potential and 1-step PBRS reward
    phi_r = w_m * sm_r + w_w * sw_r
    f_r = np.zeros(len(phi_r), dtype=np.float32)
    f_r[:-1] = gamma * phi_r[1:] - phi_r[:-1]

    # 2. Ours potential and 1-step PBRS reward
    phi_o = w_m * sm_o + w_w * sw_o
    f_o = np.zeros(len(phi_o), dtype=np.float32)
    f_o[:-1] = gamma * phi_o[1:] - phi_o[:-1]

    output_dir = PROJECT_ROOT / "visualization/outputs/case_study"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "case_study_square_seed7.mp4"

    w_cam = frames_o[0].shape[1]  # 512
    h_cam = frames_o[0].shape[0]  # 256
    w_plot = int(round(w_cam * 1.5))  # 768

    right_b = render_base_placeholder_frame(width=w_plot, height=h_cam)

    print(f"Rendering synchronized video ({total_T} frames, 1280x768 at 20fps)...")
    writer = imageio.get_writer(video_path, fps=20, codec="libx264", quality=8)

    sample_steps = [25, 65, 107, total_T - 1]
    sample_frames = {}

    for t in range(total_T):
        # 1. Base Row
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

        # 2. ResFiT Row
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
        plot_r = render_stacked_plot_frame(
            S_m=sm_r,
            S_w=sw_r,
            Phi=phi_r,
            F_t=f_r,
            current_step=t,
            total_steps=total_T,
            width=w_plot,
            height=h_cam,
            policy_name="ResFiT",
        )
        row_r = np.hstack([fr_r, plot_r])

        # 3. Ours Row
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
        plot_o = render_stacked_plot_frame(
            S_m=sm_o,
            S_w=sw_o,
            Phi=phi_o,
            F_t=f_o,
            current_step=t,
            total_steps=total_T,
            width=w_plot,
            height=h_cam,
            policy_name="Ours",
        )
        row_o = np.hstack([fr_o, plot_o])

        # Vertical stack: 3 Rows -> total size: (768, 1280, 3)
        full_canvas = np.vstack([row_b, row_r, row_o])
        writer.append_data(full_canvas)

        if t in sample_steps:
            sample_frames[t] = full_canvas.copy()

        if (t + 1) % 25 == 0 or t == total_T - 1:
            print(f"  Frame {t+1:3d}/{total_T} rendered.")

    # Freeze final frame for 20 frames (1.0s at 20fps)
    for _ in range(20):
        writer.append_data(full_canvas)

    writer.close()
    print(f"Successfully saved synchronized video to: {video_path}")

    # Copy to artifact directory
    artifact_dir = Path("/home/moai/.gemini/antigravity-cli/brain/f279ab21-5659-4869-bcd5-c8600b0f2725")
    artifact_video = artifact_dir / "case_study_square_seed7.mp4"
    import shutil
    shutil.copy2(video_path, artifact_video)
    print(f"Copied to artifact dir: {artifact_video}")

    # Save sample frames as artifacts
    for st, fr in sample_frames.items():
        sample_path = artifact_dir / f"scratch/video_potential_step_{st+1}.png"
        cv2.imwrite(str(sample_path), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        print(f"Saved sample frame step {st+1} to {sample_path}")


if __name__ == "__main__":
    main()
