import argparse
import os
import torch
import numpy as np
from pathlib import Path
from hydra import compose, initialize
from omegaconf import OmegaConf
import torchvision.transforms as T
import sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Ensure resfit is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from resfit.lerobot.utils.load_policy import load_policy
from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.rl_finetuning.wrappers.residual_env_wrapper import BasePolicyVecEnvWrapper
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lane.e2c import MLPE2C
from resfit.rl_finetuning.config.residual_td3 import ResidualTD3DexmgConfig

def get_dino_features(images, dino, device):
    images = images.to(device)
    output_size = 112
    n, c, h, w = images.shape
    top = (h - output_size) // 2
    left = (w - output_size) // 2
    cropped = images[:, :, top : top + output_size, left : left + output_size]
    # If images are [0, 255], divide by 255. BasePolicyVecEnvWrapper provides [0, 1] float
    if cropped.max() > 1.0:
        cropped = cropped.float() / 255.0
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    front = normalize(cropped[:, :3])
    wrist = normalize(cropped[:, 3:6])
    with torch.no_grad():
        feat_f = dino(front)
        feat_w = dino(wrist)
    return feat_f, feat_w

def visualize_analysis(rl_latent_path, script_dir):
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import cdist
    
    demo_latent_path = os.path.join(script_dir, "lane/pretrained_e2c/lift/demo_latents.pt")
    
    if not os.path.exists(demo_latent_path):
        print("Demo latent file not found!")
        return
        
    demo_data = torch.load(demo_latent_path, map_location="cpu", weights_only=False)
    z_f_demo = demo_data["z_demo_main"]
    z_w_demo = demo_data["z_demo_wrist"]
    num_demos = len(z_f_demo)
    
    # Load RL latents
    has_rl = rl_latent_path is not None and os.path.exists(rl_latent_path)
    if has_rl:
        rl_data = torch.load(rl_latent_path, map_location="cpu", weights_only=False)
        z_f_rl = rl_data["z_rl_main"]
        z_w_rl = rl_data["z_rl_wrist"]
        num_rl = len(z_f_rl)
        print(f"Loaded {num_rl} RL successful trajectories from {rl_latent_path}.")
    else:
        num_rl = 0
        z_f_rl = []
        z_w_rl = []
        if rl_latent_path:
            print(f"RL latents not found at {rl_latent_path}. Only plotting demo latents.")
        else:
            print("No RL latents provided. Only plotting demo latents.")
    
    # Collect points
    all_z_f_demo = np.concatenate([z.squeeze(0).numpy() for z in z_f_demo], axis=0)
    all_z_w_demo = np.concatenate([z.squeeze(0).numpy() for z in z_w_demo], axis=0)
    
    if num_rl > 0:
        all_z_f_rl = np.concatenate([z.squeeze(0).numpy() for z in z_f_rl], axis=0)
        all_z_w_rl = np.concatenate([z.squeeze(0).numpy() for z in z_w_rl], axis=0)
    else:
        all_z_f_rl = np.array([])
        all_z_w_rl = np.array([])
    
    print("Fitting PCA on ID Demos...")
    pca_f = PCA(n_components=2).fit(all_z_f_demo)
    pca_w = PCA(n_components=2).fit(all_z_w_demo)
    
    f_demo_2d = pca_f.transform(all_z_f_demo)
    w_demo_2d = pca_w.transform(all_z_w_demo)
    
    if num_rl > 0:
        f_rl_2d = pca_f.transform(all_z_f_rl)
        w_rl_2d = pca_w.transform(all_z_w_rl)
    
    # --- Extract Goal States ---
    goal_f_demo = np.stack([z.squeeze(0).numpy()[-1] for z in z_f_demo], axis=0)
    goal_w_demo = np.stack([z.squeeze(0).numpy()[-1] for z in z_w_demo], axis=0)
    goal_f_demo_2d = pca_f.transform(goal_f_demo)
    goal_w_demo_2d = pca_w.transform(goal_w_demo)
    
    if num_rl > 0:
        goal_f_rl = np.stack([z.squeeze(0).numpy()[-1] for z in z_f_rl], axis=0)
        goal_w_rl = np.stack([z.squeeze(0).numpy()[-1] for z in z_w_rl], axis=0)
        goal_f_rl_2d = pca_f.transform(goal_f_rl)
        goal_w_rl_2d = pca_w.transform(goal_w_rl)
    
    # --- 1. PCA Scatter Plot ---
    plt.figure(figsize=(14, 7))
    
    from scipy.stats import gaussian_kde
    import matplotlib.cm as cm

    def plot_kde_and_trajectories(ax, demo_2d, demo_traj_2d_list, rl_traj_2d_list, goal_demo_2d, title):
        ax.set_title(title)
        
        # 1. KDE for ID Demo
        x = demo_2d[:, 0]
        y = demo_2d[:, 1]
        
        xmin, xmax = x.min() - 0.5, x.max() + 0.5
        ymin, ymax = y.min() - 0.5, y.max() + 0.5
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        
        # Plot filled contours (keep the empty background for ID demo)
        cf = ax.contourf(X, Y, Z, levels=15, cmap='Oranges', alpha=1.0)
        
        # 2. Demo Trajectories (Start Markers Only)
        if demo_traj_2d_list:
            cmap_demo = cm.Oranges
            start_color_demo = cmap_demo(0.4)
            for i, traj in enumerate(demo_traj_2d_list):
                # Plot start marker only
                if i == 0:
                    ax.scatter(traj[0, 0], traj[0, 1], color=start_color_demo, marker='o', s=80, zorder=6, edgecolors='darkorange', label='ID Start')
                else:
                    ax.scatter(traj[0, 0], traj[0, 1], color=start_color_demo, marker='o', s=80, zorder=6, edgecolors='darkorange')
        
        # Plot Goal for Demo
        ax.scatter(goal_demo_2d[:, 0], goal_demo_2d[:, 1], color='orange', marker='*', s=150, zorder=7, edgecolors='k', label='ID Goal')
        
        # 3. KDE for RL Trajectories
        if rl_traj_2d_list:
            cmap_rl = cm.Blues
            start_color_rl = cmap_rl(0.3)
            
            # Gather all RL points for KDE
            rl_points = np.vstack(rl_traj_2d_list)
            x_rl = rl_points[:, 0]
            y_rl = rl_points[:, 1]
            
            values_rl = np.vstack([x_rl, y_rl])
            kernel_rl = gaussian_kde(values_rl)
            Z_rl = np.reshape(kernel_rl(positions).T, X.shape)
            
            # Plot filled contours for RL (transparent blue to mix with orange, keeping empty background)
            cf_rl = ax.contourf(X, Y, Z_rl, levels=15, cmap='Blues', alpha=0.6)
            
            # Plot start and goal markers for RL
            for i, traj in enumerate(rl_traj_2d_list):
                # Plot start marker
                if i == 0:
                    ax.scatter(traj[0, 0], traj[0, 1], color=start_color_rl, marker='^', s=100, zorder=8, edgecolors='black', label='RL Start')
                else:
                    ax.scatter(traj[0, 0], traj[0, 1], color=start_color_rl, marker='^', s=100, zorder=8, edgecolors='black')
                
                # Plot end goal
                if i == 0:
                    ax.scatter(traj[-1, 0], traj[-1, 1], color='navy', marker='*', s=150, zorder=9, edgecolors='k', label='RL Goal')
                else:
                    ax.scatter(traj[-1, 0], traj[-1, 1], color='navy', marker='*', s=150, zorder=9, edgecolors='k')

        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    demo_traj_f_2d = []
    demo_traj_w_2d = []
    idx = 0
    for z in z_f_demo:
        l = len(z.squeeze(0))
        demo_traj_f_2d.append(f_demo_2d[idx:idx+l])
        idx += l
    idx = 0
    for z in z_w_demo:
        l = len(z.squeeze(0))
        demo_traj_w_2d.append(w_demo_2d[idx:idx+l])
        idx += l

    if num_rl > 0:
        rl_traj_f_2d = []
        rl_traj_w_2d = []
        idx = 0
        for z in z_f_rl:
            l = len(z.squeeze(0))
            rl_traj_f_2d.append(f_rl_2d[idx:idx+l])
            idx += l
        idx = 0
        for z in z_w_rl:
            l = len(z.squeeze(0))
            rl_traj_w_2d.append(w_rl_2d[idx:idx+l])
            idx += l
    else:
        rl_traj_f_2d = []
        rl_traj_w_2d = []

    ax1 = plt.subplot(1, 2, 1)
    plot_kde_and_trajectories(ax1, f_demo_2d, demo_traj_f_2d, rl_traj_f_2d, goal_f_demo_2d, "Main Camera (PCA projected on ID variance)")
    
    ax2 = plt.subplot(1, 2, 2)
    plot_kde_and_trajectories(ax2, w_demo_2d, demo_traj_w_2d, rl_traj_w_2d, goal_w_demo_2d, "Wrist Camera (PCA projected on ID variance)")
    
    plt.tight_layout()
    if rl_latent_path:
        pca_save_path = os.path.join(os.path.dirname(rl_latent_path), "pca_comparison_advanced2.png")
    else:
        pca_save_path = os.path.join(script_dir, "outputs/pca_comparison_advanced2.png")
    
    os.makedirs(os.path.dirname(pca_save_path), exist_ok=True)
    plt.savefig(pca_save_path, dpi=150)
    print(f"PCA Plot saved to {pca_save_path}")
    

def plot_eval_latents(z_f_rl, z_w_rl, script_dir, save_path, step=None, e2c_dir=None):
    from sklearn.decomposition import PCA
    from scipy.stats import gaussian_kde
    import matplotlib.cm as cm
    import os
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if e2c_dir is None:
        print("e2c_dir is not provided. Cannot find demo_latents.pt")
        return
        
    demo_latent_path = os.path.join(e2c_dir, "demo_latents.pt")
    if not os.path.exists(demo_latent_path):
        print(f"Demo latent file not found at {demo_latent_path}")
        return
        
    demo_data = torch.load(demo_latent_path, map_location="cpu", weights_only=False)
    z_f_demo = demo_data["z_demo_main"]
    z_w_demo = demo_data["z_demo_wrist"]
    
    all_z_f_demo = np.concatenate([z.squeeze(0).numpy() for z in z_f_demo], axis=0)
    all_z_w_demo = np.concatenate([z.squeeze(0).numpy() for z in z_w_demo], axis=0)
    
    pca_f = PCA(n_components=2).fit(all_z_f_demo)
    pca_w = PCA(n_components=2).fit(all_z_w_demo)
    
    f_demo_2d = pca_f.transform(all_z_f_demo)
    w_demo_2d = pca_w.transform(all_z_w_demo)
    
    def get_demo_traj_2d(z_demo, pca):
        traj_2d = []
        for z in z_demo:
            arr = z.squeeze(0).numpy()
            traj_2d.append(pca.transform(arr))
        return traj_2d

    demo_traj_f_2d = get_demo_traj_2d(z_f_demo, pca_f)
    demo_traj_w_2d = get_demo_traj_2d(z_w_demo, pca_w)
    
    def plot_single(ax, demo_2d, demo_traj_2d, rl_traj_list, title):
        ax.set_title(title)
        
        # KDE for Demo
        x, y = demo_2d[:, 0], demo_2d[:, 1]
        xmin, xmax = x.min() - 0.5, x.max() + 0.5
        ymin, ymax = y.min() - 0.5, y.max() + 0.5
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        
        ax.contourf(X, Y, Z, levels=15, cmap='Oranges', alpha=0.7)
        
        cmap_demo = cm.Oranges
        start_color_demo = cmap_demo(0.4)
        if demo_traj_2d:
            for i, traj in enumerate(demo_traj_2d):
                if i == 0:
                    ax.scatter(traj[0, 0], traj[0, 1], color=start_color_demo, marker='o', s=80, zorder=6, edgecolors='black', label='ID Demo Start')
                    ax.scatter(traj[-1, 0], traj[-1, 1], color='darkorange', marker='*', s=150, zorder=7, edgecolors='black', label='ID Demo Goal')
                else:
                    ax.scatter(traj[0, 0], traj[0, 1], color=start_color_demo, marker='o', s=80, zorder=6, edgecolors='black')
                    ax.scatter(traj[-1, 0], traj[-1, 1], color='darkorange', marker='*', s=150, zorder=7, edgecolors='black')
        
        # RL Trajectories (Points/Lines)
        if rl_traj_list:
            cmap_rl = cm.Blues
            for i, traj in enumerate(rl_traj_list):
                if isinstance(traj, list):
                    traj_numpy = np.stack([z.detach().cpu().squeeze().numpy() for z in traj], axis=0)
                else:
                    traj_numpy = traj.detach().cpu().squeeze(0).numpy()
                    if traj_numpy.ndim == 1:
                        traj_numpy = np.expand_dims(traj_numpy, 0)
                
                traj_2d = pca_f.transform(traj_numpy) if "Main" in title else pca_w.transform(traj_numpy)
                n_points = len(traj_2d)
                
                if i == 0:
                    ax.scatter(traj_2d[0, 0], traj_2d[0, 1], color=cmap_rl(0.3), marker='^', s=100, zorder=8, edgecolors='black', label='RL Start')
                    ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], color='navy', marker='*', s=150, zorder=9, edgecolors='k', label='RL Goal')
                else:
                    ax.scatter(traj_2d[0, 0], traj_2d[0, 1], color=cmap_rl(0.3), marker='^', s=100, zorder=8, edgecolors='black')
                    ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], color='navy', marker='*', s=150, zorder=9, edgecolors='k')
                
                # Points with gradient
                if n_points > 2:
                    colors = [cmap_rl(0.3 + 0.7 * (j / n_points)) for j in range(1, n_points - 1)]
                    ax.scatter(traj_2d[1:-1, 0], traj_2d[1:-1, 1], color=colors, s=20, zorder=5, alpha=0.8, edgecolors='none')
        
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    plt.figure(figsize=(14, 7))
    ax1 = plt.subplot(1, 2, 1)
    plot_single(ax1, f_demo_2d, demo_traj_f_2d, z_f_rl, "Main Camera (PCA projected on ID variance)")
    
    ax2 = plt.subplot(1, 2, 2)
    plot_single(ax2, w_demo_2d, demo_traj_w_2d, z_w_rl, "Wrist Camera (PCA projected on ID variance)")
    
    if step is not None:
        plt.suptitle(f"Latent PCA at Step {step}", fontsize=14)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_policy_path", type=str, required=False, help="Path to base policy directory")
    parser.add_argument("--residual_policy_path", type=str, required=False, help="Path to residual agent checkpoint (.pt)")
    parser.add_argument("--n_episodes", type=int, default=20, help="Number of successful episodes to collect")
    parser.add_argument("--ood_range", type=float, default=None, help="OOD initialization range in meters (e.g. 0.05 or 0.1). If provided, applies OOD_pos environment setup.")
    parser.add_argument("--only_plot", action="store_true", help="Only generate plot using existing latents")
    parser.add_argument("--latent_path", type=str, default=None, help="Path to existing rl_latents.pt when using --only_plot")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.only_plot:
        if args.latent_path is None:
            print("Please provide --latent_path when using --only_plot")
            return
        print("\n--- Starting Advanced PCA & Distance Visualization (Only Plot) ---")
        visualize_analysis(args.latent_path, script_dir)
        
        # Test eval visualization logic
        print("\n--- Testing Eval Visualization Logic ---")
        try:
            data = torch.load(args.latent_path, map_location="cpu")
            z_f_rl = data["z_rl_main"]
            z_w_rl = data["z_rl_wrist"]
            e2c_dir = os.path.join(script_dir, "lane/pretrained_e2c/lift")
            eval_save_path = os.path.join(os.path.dirname(args.latent_path), "test_eval_plot.png")
            plot_eval_latents(z_f_rl, z_w_rl, script_dir, eval_save_path, step="Offline", e2c_dir=e2c_dir)
            print(f"Eval PCA Plot successfully generated at {eval_save_path}")
        except Exception as e:
            print(f"Failed to test eval visualization logic: {e}")
            
        return

    residual_path_obj = Path(args.residual_policy_path).resolve()
    run_dir = residual_path_obj.parent.parent
    latent_dir = run_dir / "latent"
    out_path = str(latent_dir / "rl_latents.pt")

    if args.ood_range is not None:
        os.environ["EXPERIMENT_MODE"] = "OOD_pos"
        os.environ["OOD_RANGE"] = str(args.ood_range)
        print(f"Set environment variables for OOD: EXPERIMENT_MODE=OOD_pos, OOD_RANGE={args.ood_range}")
    else:
        print("Note: --ood_range not provided. Defaulting to codebase's environment setup (usually ID).")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with initialize(version_base=None, config_path="resfit/rl_finetuning/config"):
        cfg = compose(
            config_name="residual_td3_dexmg_config",
            overrides=[
                "task=Lift",
                "rl_camera=['observation.images.agentview','observation.images.robot0_eye_in_hand']"
            ]
        )

    print(f"Loading base policy from {args.base_policy_path}...")
    base_policy = load_policy(Path(args.base_policy_path)).to(device)
    base_policy.eval()

    print("Loading dataset for normalization stats...")
    dataset = LeRobotDataset(cfg.offline_data.name, root=f"resfit/my_lerobot_data/{cfg.offline_data.name}")
    action_scaler = ActionScaler.from_dataset_stats(
        action_stats=dataset.meta.stats["action"],
        action_scale=cfg.agent.actor.action_scale,
        min_range_per_dim=cfg.offline_data.min_action_range,
        device=device,
    )
    state_standardizer = StateStandardizer.from_dataset_stats(
        state_stats=dataset.meta.stats["observation.state"],
        min_std=cfg.offline_data.min_state_std,
        device=device,
    )

    print("Creating environment...")
    vec_env = create_vectorized_env(
        env_name=cfg.task,
        num_envs=1,
        device=device,
        video_key=cfg.video_key,
        debug=False,
        camera_size=128,
    )
    env = BasePolicyVecEnvWrapper(vec_env, base_policy, action_scaler, state_standardizer)

    if OmegaConf.is_list(cfg.rl_camera) or isinstance(cfg.rl_camera, (list, tuple)):
        image_keys = list(cfg.rl_camera)
    else:
        image_keys = [cfg.rl_camera]
    img_c, img_h, img_w = vec_env.observation_space[image_keys[0]].shape[1:]
    lowdim_dim = vec_env.observation_space["observation.state"].shape[1]
    action_dim = vec_env.action_space.shape[1]

    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
    )
    print(f"Loading residual policy from {args.residual_policy_path}...")
    agent.load_state_dict(torch.load(args.residual_policy_path, map_location=device, weights_only=True))
    agent.eval()

    print("Loading E2C and DINO...")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(device)
    dino.eval()

    e2c_f = MLPE2C(obs_shape=(384,), action_dim=action_dim, z_dimension=16).to(device)
    e2c_w = MLPE2C(obs_shape=(384,), action_dim=action_dim, z_dimension=16).to(device)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    e2c_f.load_state_dict(torch.load(os.path.join(script_dir, "lane/pretrained_e2c/lift/e2c_main.pt"), map_location=device))
    e2c_w.load_state_dict(torch.load(os.path.join(script_dir, "lane/pretrained_e2c/lift/e2c_wrist.pt"), map_location=device))
    e2c_f.eval()
    e2c_w.eval()

    print(f"Collecting {args.n_episodes} successful episodes...")
    z_rl_front = []
    z_rl_wrist = []
    rl_lengths = []

    success_count = 0
    max_steps = 150

    while success_count < args.n_episodes:
        obs, _ = env.reset()
        episode_zf = []
        episode_zw = []
        success = False

        for step in range(max_steps):
            with torch.no_grad():
                action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            front_img = obs["observation.images.agentview"]
            wrist_img = obs["observation.images.robot0_eye_in_hand"]
            obs_img = torch.cat([front_img, wrist_img], dim=1)

            feat_f, feat_w = get_dino_features(obs_img, dino, device)

            with torch.no_grad():
                zf, _ = e2c_f.enc(feat_f)
                zw, _ = e2c_w.enc(feat_w)

            episode_zf.append(zf.cpu())
            episode_zw.append(zw.cpu())

            next_obs, reward, terminated, truncated, info = env.step(action)

            # Check success robustly
            if "final_info" in info and isinstance(info["final_info"], list) and info["final_info"][0] is not None:
                if info["final_info"][0].get("success", False):
                    success = True
            if reward > 0.5:
                success = True

            if terminated.any() or truncated.any():
                break

            obs = next_obs

        if success:
            success_count += 1
            print(f"Episode {success_count}/{args.n_episodes} successful. Length: {len(episode_zf)}")
            z_rl_front.append(torch.cat(episode_zf, dim=0).unsqueeze(0))
            z_rl_wrist.append(torch.cat(episode_zw, dim=0).unsqueeze(0))
            rl_lengths.append(len(episode_zf))
        else:
            print("Episode failed. Retrying...")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "z_rl_main": z_rl_front,
        "z_rl_wrist": z_rl_wrist,
        "rl_lengths": rl_lengths
    }, out_path)
    print(f"Saved RL latents to {out_path}")

    print("\n--- Starting PCA & Distance Visualization ---")
    visualize_analysis(out_path, script_dir)

if __name__ == "__main__":
    main()
