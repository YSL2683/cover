# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from pathlib import Path

import cv2
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

import wandb
from resfit.dexmg.environments.dexmg import VectorizedEnvWrapper
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent


def run_dexmg_evaluation(
    *,
    env: VectorizedEnvWrapper,
    agent: QAgent,
    num_episodes: int = 20,
    device: torch.device | str = "cpu",
    global_step: int | None = None,
    save_video: bool = False,
    save_q_plots: bool = False,
    run_name: str | None = None,
    output_dir: str | Path | None = "outputs",
    lane_shaper = None,
    e2c_dir: str | None = None,
) -> tuple[dict[str, float], float]:
    """Extended evaluation to match the richer functionality available in
    the *residual_td3_dexmg* evaluator.  In particular, this version:

    1. Annotates every rendered frame with useful metadata (env index,
       episode counter, step counter, predicted Q-value and SUCCESS/FAIL).
    2. Caches frames per-episode and flushes them into a single video file
       at the end of the evaluation.
    3. Keeps the original simple success-rate / return metrics so existing
       training code continues to work unchanged.
    """

    # ------------------------------------------------------------------
    # Helper functions (local to avoid polluting module namespace)
    # ------------------------------------------------------------------
    def _annotate_frame(
        frame: np.ndarray,
        *,
        env_idx: int,
        episode_num: int,
        total_episodes: int,
        step_idx: int,
        is_success: bool,
        q_value: float,
    ) -> np.ndarray:
        """Overlay crisp evaluation metadata onto *frame* (H, W, C)."""
        import cv2
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        font_scale = max(0.35, h / 500.0)
        thickness = 1 if h < 200 else 2
        
        status_text = "SUCCESS" if is_success else "FAIL"
        status_color = (40, 220, 40) if is_success else (220, 50, 50)
        
        lines = [
            (f"Step {step_idx}", (255, 255, 255)),
            (f"Q: {q_value:.2f}", (255, 255, 255)),
            (status_text, status_color),
            (f"Ep {episode_num}/{total_episodes}", (200, 200, 200)),
        ]
        y = int(22 * (h / 256.0))
        dy = int(24 * (h / 256.0))
        x = int(14 * (h / 256.0))
        
        for text, color in lines:
            cv2.putText(canvas, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            y += dy
            
        # Camera identifier tags if 2 cameras are present
        if w >= int(2.5 * h):
            cam_w = h
            cv2.putText(canvas, 'MAIN CAM', (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, 'MAIN CAM', (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 220, 80), 1, cv2.LINE_AA)
            cv2.putText(canvas, 'WRIST CAM', (cam_w + 15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, 'WRIST CAM', (cam_w + 15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 140, 255), 1, cv2.LINE_AA)
            
        return canvas

    def _render_2d_action_panel(base_a, res_a, tot_a, res_history=None, current_step=0, max_steps=120, target_h=256) -> np.ndarray:
        """Render a crisp, fast (<0.3ms) 2D Action HUD with upper XY/Z indicators and lower residual sparkline."""
        import cv2
        w = target_h
        canvas = np.full((target_h, w, 3), 250, dtype=np.uint8)
        
        # 1. Header & Color Legend
        cv2.putText(canvas, 'ACTION HUD', (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(canvas, 'B', (145, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 80, 220), 2, cv2.LINE_AA) # Blue
        cv2.putText(canvas, 'R', (175, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 50, 50), 2, cv2.LINE_AA) # Red
        cv2.putText(canvas, 'T', (205, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 160, 40), 2, cv2.LINE_AA) # Green
        
        # 2. Upper Half: XY Radar & Z Gauge
        cx, cy = int(w * 0.28), int(target_h * 0.33)
        radius = int(target_h * 0.18)
        cv2.circle(canvas, (cx, cy), radius, (215, 215, 215), 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), radius // 2, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.line(canvas, (cx - radius, cy), (cx + radius, cy), (225, 225, 225), 1, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy - radius), (cx, cy + radius), (225, 225, 225), 1)
        cv2.putText(canvas, 'XY', (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)
        
        scale = radius * 0.95
        # Base Arrow (Blue)
        bx = int(cx + np.clip(base_a[0], -1, 1) * scale)
        by = int(cy - np.clip(base_a[1], -1, 1) * scale)
        cv2.arrowedLine(canvas, (cx, cy), (bx, by), (50, 80, 220), 2, tipLength=0.22)
        
        # Residual Arrow (Red) starting from Base arrow tip
        rx = int(bx + np.clip(res_a[0], -1, 1) * scale)
        ry = int(by - np.clip(res_a[1], -1, 1) * scale)
        cv2.arrowedLine(canvas, (bx, by), (rx, ry), (220, 50, 50), 2, tipLength=0.22)
        
        # Total Arrow (Green) from origin
        tx = int(cx + np.clip(tot_a[0], -1, 1) * scale)
        ty = int(cy - np.clip(tot_a[1], -1, 1) * scale)
        cv2.arrowedLine(canvas, (cx, cy), (tx, ty), (40, 160, 40), 2, tipLength=0.22)
        
        # Z Gauge
        zx = int(w * 0.80)
        zy_center = int(target_h * 0.33)
        zh = int(target_h * 0.17)
        cv2.line(canvas, (zx, zy_center - zh), (zx, zy_center + zh), (210, 210, 210), 1, cv2.LINE_AA)
        cv2.line(canvas, (zx - 10, zy_center), (zx + 10, zy_center), (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'Z', (zx - 5, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)
        
        b_z = int(np.clip(base_a[2], -1, 1) * zh)
        r_z = int(np.clip(res_a[2], -1, 1) * zh)
        t_z = int(np.clip(tot_a[2], -1, 1) * zh)
        
        cv2.line(canvas, (zx - 6, zy_center), (zx - 6, zy_center - b_z), (50, 80, 220), 3, cv2.LINE_AA)
        cv2.line(canvas, (zx, zy_center), (zx, zy_center - r_z), (220, 50, 50), 3, cv2.LINE_AA)
        cv2.line(canvas, (zx + 6, zy_center), (zx + 6, zy_center - t_z), (40, 160, 40), 3, cv2.LINE_AA)
        
        # Divider Line
        div_y = int(target_h * 0.55)
        cv2.line(canvas, (12, div_y), (w - 12, div_y), (220, 220, 220), 1, cv2.LINE_AA)
        
        # 3. Lower Half: Continuous Residual Action Tracking Plot
        cur_res = float(np.linalg.norm(res_a[:3]))
        cv2.putText(canvas, f'|res|: {cur_res:.3f}', (14, div_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 40, 160), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'History', (w - 60, div_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)
        
        px0, py0 = 14, div_y + 30
        pw, ph = w - 28, int(target_h * 0.28)
        cv2.rectangle(canvas, (px0, py0), (px0 + pw, py0 + ph), (242, 242, 242), -1)
        cv2.rectangle(canvas, (px0, py0), (px0 + pw, py0 + ph), (210, 210, 210), 1, cv2.LINE_AA)
        cv2.line(canvas, (px0, py0 + ph // 2), (px0 + pw, py0 + ph // 2), (230, 230, 230), 1, cv2.LINE_AA)
        cv2.line(canvas, (px0, py0 + ph // 4), (px0 + pw, py0 + ph // 4), (235, 235, 235), 1, cv2.LINE_AA)
        cv2.line(canvas, (px0, py0 + 3 * ph // 4), (px0 + pw, py0 + 3 * ph // 4), (235, 235, 235), 1, cv2.LINE_AA)
        
        if res_history is not None and len(res_history) > 0:
            max_val = max(0.2, float(np.max(res_history)) * 1.1)
            points = []
            for step_i in range(min(current_step + 1, len(res_history))):
                val = res_history[step_i]
                x = px0 + int((step_i / max(1, max_steps - 1)) * pw)
                y = py0 + ph - int((val / max_val) * (ph - 6)) - 3
                points.append((x, int(np.clip(y, py0 + 2, py0 + ph - 2))))
                
            pts = np.array(points, np.int32)
            if len(points) >= 2:
                cv2.polylines(canvas, [pts], isClosed=False, color=(140, 60, 180), thickness=2, lineType=cv2.LINE_AA)
            if len(points) >= 1:
                cv2.circle(canvas, points[-1], 4, (120, 30, 160), -1, cv2.LINE_AA)
            cv2.putText(canvas, f'{max_val:.2f}', (px0 + 3, py0 + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
            cv2.putText(canvas, '0', (px0 + 3, py0 + ph - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
            
        return canvas

    _render_3d_action_vectors = _render_2d_action_panel

    def _create_q_trajectory_plots(
        trajectories: list[list[float]],
        episode_lengths: list[int],
        successes: list[bool],
        output_path: Path,
        global_step: int | None = None,
    ) -> None:
        """Create Q-value trajectory plots for all episodes."""
        if not trajectories:
            return

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: All Q-trajectories over time
        # Separate successful and failed episodes
        successful_trajs = [traj for i, traj in enumerate(trajectories) if successes[i]]
        failed_trajs = [traj for i, traj in enumerate(trajectories) if not successes[i]]

        # Plot all trajectories with different colors for success/failure
        for i, traj in enumerate(successful_trajs):
            steps = list(range(len(traj)))
            ax1.plot(steps, traj, "g-", alpha=0.6, linewidth=1, label="Success" if i == 0 else "")

        for i, traj in enumerate(failed_trajs):
            steps = list(range(len(traj)))
            ax1.plot(steps, traj, "r-", alpha=0.6, linewidth=1, label="Failure" if i == 0 else "")

        ax1.set_xlabel("Episode Step")
        ax1.set_ylabel("Q-Value")
        ax1.set_title(f"Q-Value Trajectories Over Time (Step {global_step or 'N/A'})")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Q-value distribution at different episode progress points
        progress_points = [0.25, 0.5, 0.75, 1.0]  # 25%, 50%, 75%, 100% of episode
        q_values_at_progress = {f"{int(p * 100)}%": [] for p in progress_points}

        for traj in trajectories:
            traj_len = len(traj)
            for p in progress_points:
                step_idx = min(int(p * traj_len), traj_len - 1)
                if step_idx < len(traj):
                    q_values_at_progress[f"{int(p * 100)}%"].append(traj[step_idx])

        # Create box plot
        box_data = [q_values_at_progress[f"{int(p * 100)}%"] for p in progress_points]
        box_labels = [f"{int(p * 100)}%" for p in progress_points]

        ax2.boxplot(box_data, labels=box_labels)
        ax2.set_xlabel("Episode Progress")
        ax2.set_ylabel("Q-Value")
        ax2.set_title("Q-Value Distribution at Different Episode Progress Points")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved Q-trajectory plots to: {output_path}")

    # ------------------------------------------------------------------
    # Initial setup -----------------------------------------------------
    # ------------------------------------------------------------------
    device = torch.device(device)
    agent.eval()

    num_envs: int = env.num_envs if hasattr(env, "num_envs") else 1

    # Per-environment episode buffers ----------------------------------
    ep_rewards: list[list[float]] = [[] for _ in range(num_envs)]
    ep_q_preds: list[list[float]] = [[] for _ in range(num_envs)]

    successes: list[bool] = []  # episode-level success flags
    returns: list[float] = []  # episode-level undiscounted returns

    # Q-trajectory data for plotting ------------------------------------
    all_q_trajectories: list[list[float]] = []  # Store Q-trajectories for all episodes
    all_episode_lengths: list[int] = []  # Store episode lengths for plotting

    # Video buffers -----------------------------------------------------
    frame_buffer: list[list[np.ndarray]] | None = [[] for _ in range(num_envs)] if save_video else None
    base_act_buffer: list[list[np.ndarray]] | None = [[] for _ in range(num_envs)] if save_video else None
    res_act_buffer: list[list[np.ndarray]] | None = [[] for _ in range(num_envs)] if save_video else None
    tot_act_buffer: list[list[np.ndarray]] | None = [[] for _ in range(num_envs)] if save_video else None
    all_frames: list[np.ndarray] | None = [] if save_video else None

    # Latent buffers
    ep_z_f = [[] for _ in range(num_envs)] if lane_shaper is not None else None
    ep_z_w = [[] for _ in range(num_envs)] if lane_shaper is not None else None
    all_success_z_f = [] if lane_shaper is not None else None
    all_success_z_w = [] if lane_shaper is not None else None
    saved_success_video = False

    done_episodes = 0
    obs, _ = env.reset()

    # Initialize progress display with dots
    progress_dots = ["."] * num_episodes
    print(f"Evaluating {num_episodes} episodes: {''.join(progress_dots)}", end="", flush=True)

    while done_episodes < num_episodes:
        # --------------------------------------------------------------
        # 1. Policy inference + Q-value prediction ---------------------
        # --------------------------------------------------------------
        with torch.no_grad():
            actions = q_actions = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            # Build features on-the-fly to obtain Q-predictions --------
            obs_q = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
            obs_q["feat"] = agent._encode(obs_q, augment=False)

            # For Q-value computation, use combined action and clamp to [-1, 1] (consistent with training)
            if agent.residual_actor:
                q_actions = torch.clamp(obs["observation.base_action"] + actions, -1.0, 1.0)

            q_pred = (
                agent.critic.q_value(obs_q["feat"], obs_q["observation.state"], q_actions).detach().cpu().squeeze(-1)
            )

        # --------------------------------------------------------------
        # 2. Environment step ------------------------------------------
        # --------------------------------------------------------------
        next_obs, reward, terminated, truncated, info = env.step(actions)
        done_flags = terminated | truncated

        # Capture frames ------------------------------------------------
        if save_video and frame_buffer is not None:
            frame = env.render()
            for env_idx in range(num_envs):
                frame_buffer[env_idx].append(frame[env_idx])
                base_act_buffer[env_idx].append(obs["observation.base_action"][env_idx].detach().cpu().numpy())
                res_act_buffer[env_idx].append(actions[env_idx].detach().cpu().numpy())
                tot_act_buffer[env_idx].append(q_actions[env_idx].detach().cpu().numpy())

        # Extract latents -----------------------------------------------
        if lane_shaper is not None:
            import sys
            import os
            project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            if project_dir not in sys.path:
                sys.path.append(project_dir)
            from visualize_rl_latents import get_dino_features
            
            obs_img = torch.cat([obs[lane_shaper.main_cam_key], obs["observation.images.robot0_eye_in_hand"]], dim=1)
            feat_f, feat_w = get_dino_features(obs_img, lane_shaper.dino, device)
            with torch.no_grad():
                if hasattr(lane_shaper, 'e2c_unified') and lane_shaper.e2c_unified is not None:
                    zf, _ = lane_shaper.e2c_unified.enc(feat_f)
                    zw, _ = lane_shaper.e2c_unified.enc(feat_w)
                else:
                    zf, _ = lane_shaper.e2c_main.enc(feat_f)
                    zw, _ = lane_shaper.e2c_wrist.enc(feat_w)
            for env_idx in range(num_envs):
                ep_z_f[env_idx].append(zf[env_idx].unsqueeze(0).cpu())
                ep_z_w[env_idx].append(zw[env_idx].unsqueeze(0).cpu())

        # --------------------------------------------------------------
        # 3. Per-environment bookkeeping -------------------------------
        # --------------------------------------------------------------
        for env_idx in range(num_envs):
            ep_rewards[env_idx].append(reward[env_idx].item())
            ep_q_preds[env_idx].append(q_pred[env_idx].item())

            if done_flags[env_idx]:
                # Episode finished -- aggregate results ----------------
                ep_return = float(sum(ep_rewards[env_idx]))
                
                # Retrieve success flag directly from environment info (Gymnasium VectorEnv format)
                is_success = False
                try:
                    if "final_info" in info:
                        if isinstance(info["final_info"], (list, tuple)):
                            if info["final_info"][env_idx] is not None:
                                is_success = bool(info["final_info"][env_idx].get("success", False))
                    if not is_success and "success" in info:
                        if isinstance(info["success"], (list, tuple, np.ndarray)):
                            is_success = bool(info["success"][env_idx])
                        elif isinstance(info["success"], torch.Tensor):
                            is_success = bool(info["success"][env_idx].item())
                        else:
                            is_success = bool(info["success"])
                except Exception:
                    pass
                
                # Ultimate fallback to reward logic just in case info is missing
                if not is_success:
                    is_success = bool(reward[env_idx].item() == 1.0 or reward[env_idx].item() > 50.0)

                # Update progress display
                progress_dots[done_episodes] = "✓" if is_success else "✗"
                print(f"\rEvaluating {num_episodes} episodes: {''.join(progress_dots)}", end="", flush=True)

                successes.append(is_success)
                returns.append(ep_return)

                # Store Q-trajectory data for plotting ------------------
                if save_q_plots:
                    all_q_trajectories.append(ep_q_preds[env_idx].copy())

                if is_success and lane_shaper is not None:
                    all_success_z_f.append(ep_z_f[env_idx].copy())
                    all_success_z_w.append(ep_z_w[env_idx].copy())

                # Always track episode length for successful episodes logging
                all_episode_lengths.append(len(ep_q_preds[env_idx]))

                # Annotate and flush frames ---------------------------
                if save_video and frame_buffer is not None and all_frames is not None:
                    episode_frames = frame_buffer[env_idx]
                    episode_qs = ep_q_preds[env_idx]

                    episode_global_idx = done_episodes + 1  # 1-based

                    # Precompute residual magnitude history for the 2D HUD sparkline
                    ep_res_acts = res_act_buffer[env_idx]
                    ep_res_norms = [float(np.linalg.norm(a[:3])) for a in ep_res_acts] if len(ep_res_acts) > 0 else []
                    max_ep_steps = len(episode_frames)

                    # Save only 1 episode video (prefer first success, or fallback to episode 1) to minimize overhead
                    should_save_video = False
                    if not saved_success_video:
                        if is_success:
                            all_frames.clear()
                            should_save_video = True
                            saved_success_video = True
                        elif episode_global_idx == 1:
                            should_save_video = True

                    if should_save_video:
                        for step_idx, fr in enumerate(episode_frames):
                            # Upscale camera frame preserving aspect ratio (supports 1, 2, or more cameras)
                            h_orig, w_orig = fr.shape[:2]
                            if h_orig < 256:
                                scale = 256.0 / h_orig
                                target_w = int(round(w_orig * scale))
                                target_h = 256
                                fr_up = cv2.resize(fr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                            else:
                                fr_up = fr

                            hud_img = _render_2d_action_panel(
                                base_act_buffer[env_idx][step_idx],
                                res_act_buffer[env_idx][step_idx],
                                tot_act_buffer[env_idx][step_idx],
                                res_history=ep_res_norms,
                                current_step=step_idx,
                                max_steps=max_ep_steps,
                                target_h=fr_up.shape[0]
                            )
                            combined_fr = np.concatenate([fr_up, hud_img], axis=1)

                            annotated_fr = _annotate_frame(
                                combined_fr,
                                env_idx=env_idx,
                                episode_num=episode_global_idx,
                                total_episodes=num_episodes,
                                step_idx=step_idx + 1,
                                is_success=is_success,
                                q_value=episode_qs[step_idx],
                            )
                            all_frames.append(annotated_fr)

                    # Clear per-episode frame buffers
                    frame_buffer[env_idx].clear()
                    base_act_buffer[env_idx].clear()
                    res_act_buffer[env_idx].clear()
                    tot_act_buffer[env_idx].clear()

                # Reset per-env caches --------------------------------
                ep_rewards[env_idx].clear()
                ep_q_preds[env_idx].clear()
                if lane_shaper is not None:
                    ep_z_f[env_idx].clear()
                    ep_z_w[env_idx].clear()

                done_episodes += 1

                if done_episodes == num_episodes:
                    break

        # Prepare for next loop ----------------------------------------
        obs = next_obs

    print("Done")

    # ------------------------------------------------------------------
    # 4. Aggregate metrics ---------------------------------------------
    # ------------------------------------------------------------------
    # Sanity check: episode lengths must align 1:1 with successes
    if len(all_episode_lengths) != len(successes):
        raise RuntimeError(
            f"Episode length/success misalignment: lengths={len(all_episode_lengths)} successes={len(successes)}"
        )

    success_rate: float = float(np.mean(successes)) if successes else 0.0
    mean_return: float = float(np.mean(returns)) if returns else 0.0

    # Calculate mean episode length among successful episodes
    successful_episode_lengths = [length for length, is_success in zip(all_episode_lengths, successes) if is_success]
    mean_successful_episode_length: float = (
        float(np.mean(successful_episode_lengths)) if successful_episode_lengths else 0.0
    )

    metrics: dict = {
        "eval/success_rate": success_rate,
        "eval/mean_return": mean_return,
        "eval/mean_successful_episode_length": mean_successful_episode_length,
    }

    if successful_episode_lengths:
        metrics["eval/successful_episode_length_hist"] = wandb.Histogram(successful_episode_lengths)

    if wandb.run is not None:
        wandb.log(metrics, step=global_step)

    # ------------------------------------------------------------------
    # 5. Q-trajectory plots --------------------------------------------
    # ------------------------------------------------------------------
    if save_q_plots and all_q_trajectories and run_name is not None:
        parent = Path(str(output_dir or "outputs")) / run_name.split("__")[0]
        parent.mkdir(parents=True, exist_ok=True)

        plot_name = f"eval_q_trajectories_{run_name}_step_{global_step if global_step is not None else 'NA'}.png"
        plot_path = parent / plot_name

        _create_q_trajectory_plots(
            trajectories=all_q_trajectories,
            episode_lengths=all_episode_lengths,
            successes=successes,
            output_path=plot_path,
            global_step=global_step,
        )

        # Log to W&B if available
        if wandb.run is not None:
            wandb.log({"value/q_trajectories": wandb.Image(str(plot_path))}, step=global_step)

    # ------------------------------------------------------------------
    # 6. Video dump + W&B logging --------------------------------------
    # ------------------------------------------------------------------
    if save_video and all_frames is not None and run_name is not None:
        parent = Path(str(output_dir or "outputs")) / run_name.split("__")[0]
        parent.mkdir(parents=True, exist_ok=True)

        vid_name = f"eval_{run_name}_step_{global_step if global_step is not None else 'NA'}.mp4"
        video_path = parent / vid_name

        fps_val = getattr(env, "fps", 20)

        writer = imageio.get_writer(video_path, fps=fps_val)
        for fr in all_frames:
            writer.append_data(fr)
        writer.close()

        if wandb.run is not None:
            wandb.log({"eval/video": wandb.Video(str(video_path), format="mp4")}, step=global_step)

    # ------------------------------------------------------------------
    # 7. Latent PCA plotting -------------------------------------------
    # ------------------------------------------------------------------
    if lane_shaper is not None and all_success_z_f and run_name is not None:
        import os
        import sys, importlib
        if "visualize_rl_latents" in sys.modules:
            importlib.reload(sys.modules["visualize_rl_latents"])
        from visualize_rl_latents import plot_eval_latents
        
        parent = Path(str(output_dir or "outputs")) / run_name.split("__")[0]
        latent_dir = parent / "latent"
        latent_dir.mkdir(parents=True, exist_ok=True)
        
        plot_name = f"eval_pca_{run_name}_step_{global_step if global_step is not None else 'NA'}.png"
        plot_path = latent_dir / plot_name
        
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # Original Time-Gradient PCA
        try:
            plot_eval_latents(all_success_z_f, all_success_z_w, project_dir, str(plot_path), step=global_step, e2c_dir=e2c_dir)
            if wandb.run is not None:
                wandb.log({"eval/latent_pca": wandb.Image(str(plot_path))}, step=global_step)
        except Exception as e:
            print(f"[Warning] Failed to plot eval latents: {e}")

    # Restore training mode --------------------------------------------
    agent.train(True)

    return metrics
