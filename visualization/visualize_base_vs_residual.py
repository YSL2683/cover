#!/usr/bin/env python3
"""
Post-Hoc Latent Space Visualization: Base Policy vs. Residual RL Policy

Projects and compares:
1. Background: 2D Gaussian KDE Density Contour of Base Policy successful trajectories (Orange).
2. Foreground: Scatter points of Residual RL Policy successful trajectories, colored by Cross-View Similarity.
   - Main Camera PCA panel is colored by Wrist View Similarity (S_wrist).
   - Wrist Camera PCA panel is colored by Main View Similarity (S_main).
   - Uses a single contrasting deep blue colormap (light sky blue -> midnight navy).
   - Start and Goal markers are completely removed as requested.

All paths are resolved relative to PROJECT_ROOT.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

# Setup project root and system path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as dist
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from lane.e2c import MLPE2C
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from resfit.lerobot.utils.load_policy import load_policy
from resfit.rl_finetuning.config.rlpd import QAgentConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.wrappers.residual_env_wrapper import BasePolicyVecEnvWrapper


# -----------------------------------------------------------------------------
# DINO Feature Extraction
# -----------------------------------------------------------------------------
def get_dino_features(images: torch.Tensor, dino: torch.nn.Module, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract DINOv2 embeddings (384-d each) for main and wrist camera views.
    
    Args:
        images: Float or uint8 tensor of shape [B, 6, H, W] containing [main_cam, wrist_cam].
    """
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


def _to_np(z) -> np.ndarray:
    """Helper to convert latent tensors to numpy 2D array (T, latent_dim)."""
    if z is None:
        return np.array([])
    if isinstance(z, list):
        if len(z) == 0:
            return np.array([])
        arrs = []
        for x in z:
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            elif hasattr(x, "cpu"):
                x = x.cpu().numpy()
            else:
                x = np.asarray(x)
            arrs.append(x.reshape(-1))
        return np.stack(arrs, axis=0)
    else:
        if hasattr(z, "detach"):
            z = z.detach().cpu().numpy()
        elif hasattr(z, "cpu"):
            z = z.cpu().numpy()
        else:
            z = np.asarray(z)
        if z.ndim == 3:
            if z.shape[0] == 1:
                z = z.squeeze(0)
            elif z.shape[1] == 1:
                z = z.squeeze(1)
            else:
                z = z.reshape(-1, z.shape[-1])
        elif z.ndim == 1:
            z = np.expand_dims(z, 0)
        return z


# -----------------------------------------------------------------------------
# Rollout Collection
# -----------------------------------------------------------------------------
def collect_trajectories(
    env: BasePolicyVecEnvWrapper,
    agent: QAgent | None,
    dino: torch.nn.Module,
    e2c_main: torch.nn.Module,
    e2c_wrist: torch.nn.Module,
    mode: str,
    target_episodes: int = 50,
    main_cam_key: str = "observation.images.agentview",
    wrist_cam_key: str = "observation.images.robot0_eye_in_hand",
    device: torch.device = torch.device("cpu"),
    seed: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Runs rollouts until `target_episodes` successful trajectories are collected.
    
    Args:
        mode: 'base' (zero residual action) or 'residual' (agent.act).
        seed: Initial seed for environment reset.
    """
    print(f"\n[Rollout] Collecting {target_episodes} successful episodes for mode: '{mode.upper()}' (seed={seed})...")
    num_envs = env.num_envs if hasattr(env, "num_envs") else 1
    action_dim = env.action_space.shape[1]

    successful_trajs_f: list[np.ndarray] = []
    successful_trajs_w: list[np.ndarray] = []

    ep_z_f = [[] for _ in range(num_envs)]
    ep_z_w = [[] for _ in range(num_envs)]

    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    completed_episodes = 0
    total_rollouts = 0

    while len(successful_trajs_f) < target_episodes:
        with torch.no_grad():
            if mode == "base" or agent is None:
                # Zero residual action: BasePolicyVecEnvWrapper executes purely base policy
                actions = torch.zeros((num_envs, action_dim), device=device)
            else:
                actions = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            # Extract DINO and E2C latent state for current frame
            img_m = obs[main_cam_key]
            img_w = obs[wrist_cam_key]
            obs_img = torch.cat([img_m, img_w], dim=1)
            feat_f, feat_w = get_dino_features(obs_img, dino, device)

            zf, _ = e2c_main.enc(feat_f)
            zw, _ = e2c_wrist.enc(feat_w)

            for env_idx in range(num_envs):
                ep_z_f[env_idx].append(zf[env_idx].detach().cpu().numpy())
                ep_z_w[env_idx].append(zw[env_idx].detach().cpu().numpy())

        # Step simulator
        next_obs, reward, terminated, truncated, info = env.step(actions)
        done_flags = terminated | truncated

        for env_idx in range(num_envs):
            if done_flags[env_idx]:
                total_rollouts += 1
                is_success = False
                try:
                    if "final_info" in info and isinstance(info["final_info"], (list, tuple)):
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

                if not is_success:
                    is_success = bool(reward[env_idx].item() == 1.0 or reward[env_idx].item() > 50.0)

                if is_success and len(successful_trajs_f) < target_episodes:
                    traj_f = np.stack(ep_z_f[env_idx], axis=0)  # (T, 16)
                    traj_w = np.stack(ep_z_w[env_idx], axis=0)  # (T, 16)
                    successful_trajs_f.append(traj_f)
                    successful_trajs_w.append(traj_w)
                    completed_episodes += 1

                # Live progress update for every attempt
                print(
                    f"\r  -> [{mode.upper()}] Successes: {len(successful_trajs_f)}/{target_episodes} "
                    f"(Total Attempts: {total_rollouts}, SR: {len(successful_trajs_f)/total_rollouts:.1%})",
                    end="",
                    flush=True,
                )

                ep_z_f[env_idx].clear()
                ep_z_w[env_idx].clear()

        obs = next_obs

    print(f"\n[Rollout] Finished {mode.upper()}: {len(successful_trajs_f)} successful trajectories collected.")
    return successful_trajs_f, successful_trajs_w


# -----------------------------------------------------------------------------
# Custom Contrasting Deep Blue Colormap
# -----------------------------------------------------------------------------
def get_contrasting_blue_colormap() -> mcolors.LinearSegmentedColormap:
    """Returns a smooth sequential colormap contrasting sharply with Orange.
    
    Transitions from light cyan/sky blue (#80d8ff) at S=0 to deep midnight navy (#081b4b) at S=1.
    Higher cross-view similarity produces a noticeably darker, richer shade.
    """
    colors = [
        (0.00, "#80d8ff"),  # Light Sky Cyan
        (0.35, "#3d84a8"),  # Steel Ocean Blue
        (0.70, "#19398a"),  # Deep Cobalt Blue
        (1.00, "#081b4b"),  # Midnight Navy
    ]
    return mcolors.LinearSegmentedColormap.from_list("ContrastingBlueNavy", [c[1] for c in colors])


# -----------------------------------------------------------------------------
# Plotting & Analysis
# -----------------------------------------------------------------------------
def plot_base_vs_residual_latent_space(
    base_trajs_f: list[np.ndarray],
    base_trajs_w: list[np.ndarray],
    rl_trajs_f: list[np.ndarray],
    rl_trajs_w: list[np.ndarray],
    demo_latents_path: Path | str,
    beta: float = 1.0,
    point_alpha: float = 0.7,
    point_size: float = 8.0,
    bg_style: str = "heatmap",
    save_path: Path | str = "outputs/base_vs_residual_pca.png",
    title_suffix: str = "",
):
    """Generates a 2-panel PCA plot:
    
    - Background: Base Policy successful rollouts represented as continuous density glow (or contour).
    - Foreground: Points of Residual RL successful rollouts colored by Cross-View Similarity with alpha blending.
    - Start and Goal markers are completely removed.
    """
    print("\n[Plotting] Fitting PCA on Base Policy successful trajectories...")
    all_z_base_f = np.concatenate([_to_np(z) for z in base_trajs_f], axis=0)
    all_z_base_w = np.concatenate([_to_np(z) for z in base_trajs_w], axis=0)

    pca_f = PCA(n_components=2).fit(all_z_base_f)
    pca_w = PCA(n_components=2).fit(all_z_base_w)

    f_base_2d = pca_f.transform(all_z_base_f)
    w_base_2d = pca_w.transform(all_z_base_w)

    rl_trajs_f_clean = [_to_np(z) for z in rl_trajs_f if len(_to_np(z)) > 0]
    rl_trajs_w_clean = [_to_np(z) for z in rl_trajs_w if len(_to_np(z)) > 0]

    all_z_rl_f = np.vstack(rl_trajs_f_clean)
    all_z_rl_w = np.vstack(rl_trajs_w_clean)

    # Load demonstration latents for similarity normalization
    print(f"[Plotting] Loading demonstration latents from {demo_latents_path}...")
    demo_data = torch.load(demo_latents_path, map_location="cpu", weights_only=False)
    z_f_demo = demo_data.get("z_demo_main", demo_data.get("z_demo_front"))
    z_w_demo = demo_data["z_demo_wrist"]

    all_z_demo_f = np.concatenate([z.squeeze(0).numpy() for z in z_f_demo], axis=0)
    all_z_demo_w = np.concatenate([z.squeeze(0).numpy() for z in z_w_demo], axis=0)

    # Compute 1-step reference distances (exact match with lane_reward_shaper.py)
    one_step_f = [
        ((z.squeeze(0)[1:] - z.squeeze(0)[:-1]) ** 2).sum(dim=1).mean().item()
        for z in z_f_demo
        if len(z.squeeze(0)) > 1
    ]
    ref_one_step_dist_f = sum(one_step_f) / len(one_step_f)

    one_step_w = [
        ((z.squeeze(0)[1:] - z.squeeze(0)[:-1]) ** 2).sum(dim=1).mean().item()
        for z in z_w_demo
        if len(z.squeeze(0)) > 1
    ]
    ref_one_step_dist_w = sum(one_step_w) / len(one_step_w)

    gamma_f = beta / ((ref_one_step_dist_f ** 2) + 1e-8)
    gamma_w = beta / ((ref_one_step_dist_w ** 2) + 1e-8)

    # Compute minimal squared distances to demonstration manifold
    dist_f = dist.cdist(all_z_rl_f, all_z_demo_f, "sqeuclidean").min(axis=1)
    dist_w = dist.cdist(all_z_rl_w, all_z_demo_w, "sqeuclidean").min(axis=1)

    # 4th-power exponential kernel (reward_pbrs_no_mask_nstep formulation)
    S_f_all = np.exp(-gamma_f * (dist_f ** 2))
    S_w_all = np.exp(-gamma_w * (dist_w ** 2))

    cmap_blue = get_contrasting_blue_colormap()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), dpi=200)

    def _render_single_panel(ax, base_2d, rl_traj_list, pca_model, S_cross_all, view_name, cross_view_label):
        all_x = [base_2d[:, 0]] + [traj[:, 0] for traj in rl_traj_list]
        all_y = [base_2d[:, 1]] + [traj[:, 1] for traj in rl_traj_list]
        x_all = np.concatenate(all_x)
        y_all = np.concatenate(all_y)
        
        xmin, xmax = x_all.min() - 0.5, x_all.max() + 0.5
        ymin, ymax = y_all.min() - 0.5, y_all.max() + 0.5

        X, Y = np.mgrid[xmin:xmax:140j, ymin:ymax:140j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        kernel = gaussian_kde(np.vstack([base_2d[:, 0], base_2d[:, 1]]))
        Z = np.reshape(kernel(positions).T, X.shape)

        # 1. Base Policy Density Representation
        if bg_style == "heatmap":
            # Continuous Soft Density Heatmap / Glow via imshow
            Z_2d = Z.T
            Z_norm = (Z_2d - Z_2d.min()) / (Z_2d.max() - Z_2d.min() + 1e-8)
            # Non-linear color mapping: boost mid/low density into richer, deeper orange tones
            Z_color = np.clip(Z_norm ** 0.58, 0.0, 1.0)
            cmap_orange = plt.cm.Oranges
            rgba = cmap_orange(Z_color)
            # Richer, deeper alpha fade for vivid background density presence
            rgba[..., 3] = np.clip(Z_norm ** 0.70, 0.0, 1.0) * 0.88
            ax.imshow(
                rgba,
                extent=[xmin, xmax, ymin, ymax],
                origin="lower",
                interpolation="bicubic",
                zorder=1,
                aspect="auto",
            )
        else:
            # Discrete contour bands
            cf = ax.contourf(X, Y, Z, levels=16, cmap="Oranges", alpha=0.55, zorder=1)

        # 2. Residual Policy Trajectories (Colored by Cross-View Similarity with alpha blending)
        # Note: Start and Goal markers are intentionally omitted
        idx = 0
        sc = None
        for traj in rl_traj_list:
            n_pts = len(traj)
            traj_2d = pca_model.transform(traj)
            s_cross = S_cross_all[idx : idx + n_pts]
            idx += n_pts

            sc = ax.scatter(
                traj_2d[:, 0],
                traj_2d[:, 1],
                c=s_cross,
                cmap=cmap_blue,
                s=point_size,
                alpha=point_alpha,
                vmin=0.0,
                vmax=1.0,
                edgecolors="none",
                zorder=5,
            )


        ax.set_title(f"{view_name} Latent Space\n(Background: Base Policy | Color: {cross_view_label})", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Principal Component 1", fontsize=11)
        ax.set_ylabel("Principal Component 2", fontsize=11)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.25, linestyle="--")

        # Colorbar for Cross-View Similarity
        if sc is not None:
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"Cross-View Similarity: {cross_view_label} [0, 1]", rotation=270, labelpad=18, fontsize=11)
            cbar.ax.tick_params(labelsize=9)

    # Left: Main Camera PCA colored by S_wrist
    _render_single_panel(
        axes[0],
        base_2d=f_base_2d,
        rl_traj_list=rl_trajs_f_clean,
        pca_model=pca_f,
        S_cross_all=S_w_all,
        view_name="Main Camera (Agentview)",
        cross_view_label="Wrist Guidance (S_wrist)",
    )

    # Right: Wrist Camera PCA colored by S_main
    _render_single_panel(
        axes[1],
        base_2d=w_base_2d,
        rl_traj_list=rl_trajs_w_clean,
        pca_model=pca_w,
        S_cross_all=S_f_all,
        view_name="Wrist Camera (Eye-in-hand)",
        cross_view_label="Main Guidance (S_main)",
    )

    bg_label = "Orange Density Glow" if bg_style == "heatmap" else "Orange Contour"
    suptitle = f"Latent Space Comparison: Base Policy ({bg_label}) vs. Residual RL (Deep Blue Points)"
    if title_suffix:
        suptitle += f"\n[{title_suffix}]"
    plt.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[Plotting] Successfully saved latent visualization to: {save_path}")


# -----------------------------------------------------------------------------
# Main CLI Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Post-hoc Latent Space Visualization: Base vs. Residual Policy")
    parser.add_argument("--task", type=str, default="Square", help="Environment task name (e.g. Square, Can, Lift)")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="resfit/outputs/2026-09-02_11-33-44_Square_reward_pbrs_no_mask_nstep_beta1.0_scale0.1/models/agent_best.pt",
        help="Relative path to residual RL agent checkpoint",
    )
    parser.add_argument(
        "--base_policy_path",
        type=str,
        default="resfit/my_lerobot_data/bc_run_2026-08-29_14-38-11_robomimic_square_v15_50_diffusion/policy_step_66000/policy",
        help="Relative path to base policy checkpoint",
    )
    parser.add_argument(
        "--e2c_dir",
        type=str,
        default="lane/pretrained_e2c/square",
        help="Relative path to pretrained E2C weights and demo latents",
    )
    parser.add_argument(
        "--offline_data_name",
        type=str,
        default="resfit/my_lerobot_data/ysl2683/robomimic_square_v15_50",
        help="Relative path to dataset directory for normalizer initialization",
    )
    parser.add_argument("--num_episodes", type=int, default=50, help="Target successful episodes to collect per policy")
    parser.add_argument("--num_envs", type=int, default=5, help="Number of parallel environments for rollout")
    parser.add_argument("--action_scale", type=float, default=0.1, help="Residual action scale factor (matches training)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility and environment reset")
    parser.add_argument("--beta", type=float, default=1.0, help="Kernel scale parameter beta")
    parser.add_argument("--point_alpha", type=float, default=0.7, help="Alpha transparency for residual scatter points (default: 0.7)")
    parser.add_argument("--point_size", type=float, default=8.0, help="Marker size for residual scatter points (default: 8.0)")
    parser.add_argument("--bg_style", type=str, default="heatmap", choices=["heatmap", "contour"], help="Background representation style: 'heatmap' (continuous soft density glow) or 'contour' (discrete contour bands)")
    parser.add_argument("--output_path", type=str, default="visualization/outputs/base_vs_residual_square.png", help="Output PNG path")
    parser.add_argument("--cache_dir", type=str, default="visualization/cache", help="Cache directory for collected latents")
    parser.add_argument("--use_cache", action="store_true", help="Load cached trajectories if available to skip rollouts")
    parser.add_argument("--dry_run", action="store_true", help="Validate components and generate a synthetic dry-run figure without rollouts")
    args = parser.parse_args()

    # Set random seeds for strict reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Resolve paths relative to PROJECT_ROOT
    ckpt_path = PROJECT_ROOT / args.checkpoint_path
    base_path = PROJECT_ROOT / args.base_policy_path
    e2c_dir = PROJECT_ROOT / args.e2c_dir
    # Resolve offline dataset path under resfit/my_lerobot_data
    offline_arg = str(args.offline_data_name).strip("/")
    if "my_lerobot_data/" in offline_arg:
        repo_id = offline_arg.split("my_lerobot_data/")[-1]
    else:
        repo_id = offline_arg
    data_path = PROJECT_ROOT / "resfit/my_lerobot_data" / repo_id
    output_path = PROJECT_ROOT / args.output_path
    cache_dir = PROJECT_ROOT / args.cache_dir
    demo_latents_path = e2c_dir / "demo_latents.pt"

    print("=" * 70)
    print("Post-Hoc Latent Space Visualization (Base vs. Residual Policy)")
    print("=" * 70)
    print(f"Task               : {args.task}")
    print(f"Seed               : {args.seed}")
    print(f"Action Scale       : {args.action_scale}")
    print(f"Checkpoint Path    : {ckpt_path}")
    print(f"Base Policy Path   : {base_path}")
    print(f"E2C Directory      : {e2c_dir}")
    print(f"Demo Latents Path  : {demo_latents_path}")
    print(f"Offline Data Path  : {data_path}")
    print(f"Target Episodes    : {args.num_episodes} successes per policy")
    print(f"Output Figure Path : {output_path}")
    print(f"Dry Run Mode       : {args.dry_run}")
    print("=" * 70)

    # Validate essential files exist
    assert ckpt_path.exists(), f"Checkpoint file not found: {ckpt_path}"
    assert base_path.exists(), f"Base policy directory not found: {base_path}"
    assert e2c_dir.exists(), f"E2C directory not found: {e2c_dir}"
    assert demo_latents_path.exists(), f"Demo latents file not found: {demo_latents_path}"

    device = torch.device("cuda" if torch.cuda.is_available() and not args.dry_run else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Dry-Run Mode: Verify models & generate synthetic preview plot without GPU load
    # -------------------------------------------------------------------------
    if args.dry_run:
        print("\n[Dry-Run] Initializing models on CPU to verify architecture...")
        # 1. Verify E2C
        e2c_main = MLPE2C(obs_shape=(384,), action_dim=7, z_dimension=16).to("cpu")
        e2c_main.load_state_dict(torch.load(e2c_dir / "e2c_main.pt", map_location="cpu"))
        e2c_wrist = MLPE2C(obs_shape=(384,), action_dim=7, z_dimension=16).to("cpu")
        e2c_wrist.load_state_dict(torch.load(e2c_dir / "e2c_wrist.pt", map_location="cpu"))
        print("  -> Pretrained E2C models loaded successfully.")

        # 2. Verify Agent Weights
        cfg_agent = QAgentConfig()
        cfg_agent.actor.action_scale = args.action_scale
        agent = QAgent(
            obs_shape=(3, 128, 128),
            prop_shape=(9,),
            action_dim=7,
            rl_cameras=["observation.images.agentview", "observation.images.robot0_eye_in_hand"],
            cfg=cfg_agent,
            residual_actor=True,
        )
        agent.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        print(f"  -> Residual QAgent checkpoint loaded successfully (action_scale={cfg_agent.actor.action_scale}).")

        # 3. Create synthetic preview trajectory data
        print("\n[Dry-Run] Generating synthetic trajectories to test PCA/KDE/Coloring pipeline...")
        np.random.seed(args.seed)
        syn_base_f, syn_base_w = [], []
        syn_rl_f, syn_rl_w = [], []

        for _ in range(args.num_episodes):
            T_len = np.random.randint(40, 80)
            t_axis = np.linspace(0, 1, T_len)[:, None]
            # Base trajectory: smooth random walk around origin
            noise_base = np.cumsum(np.random.randn(T_len, 16) * 0.1, axis=0) + t_axis * 0.5
            syn_base_f.append(noise_base)
            syn_base_w.append(noise_base + 0.1)

            # Residual trajectory: more goal-directed / tighter
            noise_rl = np.cumsum(np.random.randn(T_len, 16) * 0.08, axis=0) + t_axis * 0.55
            syn_rl_f.append(noise_rl)
            syn_rl_w.append(noise_rl + 0.1)

        dry_run_out = output_path.parent / f"dry_run_{output_path.name}"
        plot_base_vs_residual_latent_space(
            base_trajs_f=syn_base_f,
            base_trajs_w=syn_base_w,
            rl_trajs_f=syn_rl_f,
            rl_trajs_w=syn_rl_w,
            demo_latents_path=demo_latents_path,
            beta=args.beta,
            point_alpha=args.point_alpha,
            point_size=args.point_size,
            bg_style=args.bg_style,
            save_path=dry_run_out,
            title_suffix="Dry-Run Verification Test",
        )
        print(f"\n[Dry-Run Complete] Synthetic preview figure verified at: {dry_run_out}")
        return

    # -------------------------------------------------------------------------
    # Full Execution Mode
    # -------------------------------------------------------------------------
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{args.task}_base_vs_residual_{args.num_episodes}ep.pt"

    if args.use_cache and cache_file.exists():
        print(f"\n[Cache] Loading cached trajectories from {cache_file}...")
        cached_data = torch.load(cache_file, map_location="cpu", weights_only=False)
        base_trajs_f = cached_data["base_trajs_f"]
        base_trajs_w = cached_data["base_trajs_w"]
        rl_trajs_f = cached_data["rl_trajs_f"]
        rl_trajs_w = cached_data["rl_trajs_w"]
    else:
        # 1. Load Normalizers & Dataset
        print(f"Loading dataset metadata from local dataset: {data_path}...")
        dataset = LeRobotDataset(repo_id, root=data_path)
        action_scaler = ActionScaler.from_dataset_stats(
            action_stats=dataset.meta.stats["action"],
            action_scale=args.action_scale,
            device=device,
        )
        state_standardizer = StateStandardizer.from_dataset_stats(
            state_stats=dataset.meta.stats["observation.state"],
            device=device,
        )

        # 2. Load Base Diffusion Policy
        print(f"Loading Base Policy from {base_path}...")
        base_policy: DiffusionPolicy = load_policy(base_path)
        base_policy.to(device)
        base_policy.eval()

        # 3. Create Vectorized Environment wrapped with Base Policy
        print(f"Creating {args.num_envs} vectorized environment(s) for task: {args.task}...")
        raw_vec_env = create_vectorized_env(
            env_name=args.task,
            num_envs=args.num_envs,
            device=str(device),
            camera_size=128,
        )
        env = BasePolicyVecEnvWrapper(
            vec_env=raw_vec_env,
            base_policy=base_policy,
            action_scaler=action_scaler,
            state_standardizer=state_standardizer,
        )

        # 4. Load DINOv2 and E2C Encoders
        print("Loading DINOv2 ViT-S/14 backbone...")
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(device)
        dino.eval()

        print(f"Loading pretrained E2C encoders from {e2c_dir}...")
        action_dim = env.action_space.shape[1]
        e2c_main = MLPE2C(obs_shape=(384,), action_dim=action_dim, z_dimension=16).to(device)
        e2c_main.load_state_dict(torch.load(e2c_dir / "e2c_main.pt", map_location=device))
        e2c_main.eval()

        e2c_wrist = MLPE2C(obs_shape=(384,), action_dim=action_dim, z_dimension=16).to(device)
        e2c_wrist.load_state_dict(torch.load(e2c_dir / "e2c_wrist.pt", map_location=device))
        e2c_wrist.eval()

        # 5. Load Residual QAgent with explicit action_scale matching training
        print(f"Loading Residual QAgent from {ckpt_path} (action_scale={args.action_scale})...")
        cfg_agent = QAgentConfig()
        cfg_agent.actor.action_scale = args.action_scale
        lowdim_dim = env.observation_space["observation.state"].shape[1]
        agent = QAgent(
            obs_shape=(3, 128, 128),
            prop_shape=(lowdim_dim,),
            action_dim=action_dim,
            rl_cameras=["observation.images.agentview", "observation.images.robot0_eye_in_hand"],
            cfg=cfg_agent,
            residual_actor=True,
        )
        agent.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        agent.eval()

        # 6. Collect Base Policy Successes (starting with seed)
        base_trajs_f, base_trajs_w = collect_trajectories(
            env=env,
            agent=None,
            dino=dino,
            e2c_main=e2c_main,
            e2c_wrist=e2c_wrist,
            mode="base",
            target_episodes=args.num_episodes,
            device=device,
            seed=args.seed,
        )

        # 7. Collect Residual Policy Successes (starting with exact same seed)
        rl_trajs_f, rl_trajs_w = collect_trajectories(
            env=env,
            agent=agent,
            dino=dino,
            e2c_main=e2c_main,
            e2c_wrist=e2c_wrist,
            mode="residual",
            target_episodes=args.num_episodes,
            device=device,
            seed=args.seed,
        )

        # Save to Cache
        print(f"\n[Cache] Saving collected trajectories to {cache_file}...")
        torch.save(
            {
                "base_trajs_f": base_trajs_f,
                "base_trajs_w": base_trajs_w,
                "rl_trajs_f": rl_trajs_f,
                "rl_trajs_w": rl_trajs_w,
            },
            cache_file,
        )

    # 8. Render Composite Visualization
    plot_base_vs_residual_latent_space(
        base_trajs_f=base_trajs_f,
        base_trajs_w=base_trajs_w,
        rl_trajs_f=rl_trajs_f,
        rl_trajs_w=rl_trajs_w,
        demo_latents_path=demo_latents_path,
        beta=args.beta,
        point_alpha=args.point_alpha,
        point_size=args.point_size,
        bg_style=args.bg_style,
        save_path=output_path,
        title_suffix=f"Task: {args.task} | {len(base_trajs_f)} Successful Trajectories per Policy",
    )


if __name__ == "__main__":
    main()
