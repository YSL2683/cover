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
    if h < output_size or w < output_size:
        import torchvision.transforms.functional as TF
        cropped = TF.resize(images, [output_size, output_size])
    else:
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

def _to_np(z):
    """Converts a latent trajectory (tensor, numpy array, list of tensors, or list of arrays)
    into a 2D numpy array of shape (T, latent_dim).
    """
    if z is None:
        return np.array([])
    if isinstance(z, list):
        if len(z) == 0:
            return np.array([])
        arrs = []
        for x in z:
            if hasattr(x, 'detach'):
                x = x.detach().cpu().numpy()
            elif hasattr(x, 'cpu'):
                x = x.cpu().numpy()
            else:
                x = np.asarray(x)
            arrs.append(x.reshape(-1))
        return np.stack(arrs, axis=0)
    else:
        if hasattr(z, 'detach'):
            z = z.detach().cpu().numpy()
        elif hasattr(z, 'cpu'):
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
        all_z_f_rl = np.concatenate([_to_np(z) for z in z_f_rl], axis=0)
        all_z_w_rl = np.concatenate([_to_np(z) for z in z_w_rl], axis=0)
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
        goal_f_rl = np.stack([_to_np(z)[-1] for z in z_f_rl], axis=0)
        goal_w_rl = np.stack([_to_np(z)[-1] for z in z_w_rl], axis=0)
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
                    ax.scatter(traj[0, 0], traj[0, 1], color=start_color_demo, marker='^', s=100, zorder=6, edgecolors='darkorange', label='ID Start')
                else:
                    ax.scatter(traj[0, 0], traj[0, 1], color=start_color_demo, marker='^', s=100, zorder=6, edgecolors='darkorange')
        
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
            l = len(_to_np(z))
            rl_traj_f_2d.append(f_rl_2d[idx:idx+l])
            idx += l
        idx = 0
        for z in z_w_rl:
            l = len(_to_np(z))
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
                    ax.scatter(traj[0, 0], traj[0, 1], color=start_color_demo, marker='^', s=100, zorder=6, edgecolors='black', label='ID Demo Start')
                    ax.scatter(traj[-1, 0], traj[-1, 1], color='darkorange', marker='*', s=150, zorder=7, edgecolors='black', label='ID Demo Goal')
                else:
                    ax.scatter(traj[0, 0], traj[0, 1], color=start_color_demo, marker='^', s=100, zorder=6, edgecolors='black')
                    ax.scatter(traj[-1, 0], traj[-1, 1], color='darkorange', marker='*', s=150, zorder=7, edgecolors='black')
        
        # RL Trajectories (Points/Lines)
        if rl_traj_list:
            cmap_rl = cm.Blues
            for i, traj in enumerate(rl_traj_list):
                traj_numpy = _to_np(traj)
                if len(traj_numpy) == 0:
                    continue
                
                traj_2d = pca_f.transform(traj_numpy) if "Main" in title else pca_w.transform(traj_numpy)
                n_points = len(traj_2d)
                if n_points == 0:
                    continue
                
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
    

def plot_crossview_pca(z_f_rl, z_w_rl, script_dir, save_path, step=None, e2c_dir=None, gamma_f=None, gamma_w=None, is_2squared=False):
    import os
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from sklearn.decomposition import PCA
    from scipy.stats import gaussian_kde
    import scipy.spatial.distance as dist
    
    if e2c_dir is None: return
    demo_latent_path = os.path.join(e2c_dir, "demo_latents.pt")
    if not os.path.exists(demo_latent_path): return
        
    demo_data = torch.load(demo_latent_path, map_location="cpu", weights_only=False)
    z_f_demo = demo_data.get("z_demo_main", demo_data.get("z_demo_front"))
    z_w_demo = demo_data["z_demo_wrist"]
    
    all_z_f_demo = np.concatenate([z.squeeze(0).numpy() for z in z_f_demo], axis=0)
    all_z_w_demo = np.concatenate([z.squeeze(0).numpy() for z in z_w_demo], axis=0)
    
    pca_f = PCA(n_components=2).fit(all_z_f_demo)
    pca_w = PCA(n_components=2).fit(all_z_w_demo)
    
    f_demo_2d = pca_f.transform(all_z_f_demo)
    w_demo_2d = pca_w.transform(all_z_w_demo)
    
    rl_traj_f_list = [arr for arr in [_to_np(z) for z in z_f_rl] if len(arr) > 0]
    rl_traj_w_list = [arr for arr in [_to_np(z) for z in z_w_rl] if len(arr) > 0]
    
    if len(rl_traj_f_list) == 0 or len(rl_traj_w_list) == 0:
        return
    
    all_z_f_rl_flat = np.vstack(rl_traj_f_list)
    all_z_w_rl_flat = np.vstack(rl_traj_w_list)
    
    dist_f = dist.cdist(all_z_f_rl_flat, all_z_f_demo, 'sqeuclidean')
    min_dist_f = dist_f.min(axis=1)
    dist_w = dist.cdist(all_z_w_rl_flat, all_z_w_demo, 'sqeuclidean')
    min_dist_w = dist_w.min(axis=1)
    
    if gamma_f is not None and gamma_w is not None:
        # Use the exact reward shaping values
        _gamma_f = gamma_f
        _gamma_w = gamma_w
        if not is_2squared:
            # 4th power kernel uses squared min_dist because min_dist is already squared
            S_f_global = np.exp(-_gamma_f * (min_dist_f ** 2))
            S_w_global = np.exp(-_gamma_w * (min_dist_w ** 2))
        else:
            S_f_global = np.exp(-_gamma_f * min_dist_f)
            S_w_global = np.exp(-_gamma_w * min_dist_w)
    else:
        # Fallback normalization for offline tests
        _gamma_f = 2.0 / (np.mean(min_dist_f) + 1e-8)
        S_f_global = np.exp(-_gamma_f * min_dist_f)
        _gamma_w = 2.0 / (np.mean(min_dist_w) + 1e-8)
        S_w_global = np.exp(-_gamma_w * min_dist_w)
    
    fig = plt.figure(figsize=(16, 7))
    
    def plot_single_crossview(ax, demo_2d, demo_raw_list, rl_traj_list, pca_model, S_other_global, title, cbar_label):
        ax.set_title(title, fontsize=14, pad=15)
        
        x, y = demo_2d[:, 0], demo_2d[:, 1]
        xmin, xmax = x.min() - 0.5, x.max() + 0.5
        ymin, ymax = y.min() - 0.5, y.max() + 0.5
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        
        ax.contourf(X, Y, Z, levels=15, cmap='Oranges', alpha=0.5)
        
        start_color_demo = cm.Oranges(0.4)
        start_color_rl = cm.Blues(0.3)
        
        for i, traj_raw in enumerate(demo_raw_list):
            traj_arr = traj_raw.squeeze(0).numpy() if hasattr(traj_raw, 'numpy') else traj_raw
            traj_2d = pca_model.transform(traj_arr)
            if i == 0:
                ax.scatter(traj_2d[0, 0], traj_2d[0, 1], color=start_color_demo, marker='^', s=100, zorder=6, edgecolors='black', label='ID Demo Start')
                ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], color='darkorange', marker='*', s=150, zorder=7, edgecolors='black', label='ID Demo Goal')
            else:
                ax.scatter(traj_2d[0, 0], traj_2d[0, 1], color=start_color_demo, marker='^', s=100, zorder=6, edgecolors='black')
                ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], color='darkorange', marker='*', s=150, zorder=7, edgecolors='black')
                
        cmap = "viridis"
        idx = 0
        sc = None
        for i, traj in enumerate(rl_traj_list):
            n_points = len(traj)
            traj_2d = pca_model.transform(traj)
            S_traj = S_other_global[idx:idx+n_points]
            idx += n_points
            sc = ax.scatter(traj_2d[:, 0], traj_2d[:, 1], c=S_traj, cmap=cmap, s=25, zorder=5, alpha=0.9, vmin=0, vmax=1)
            
            if i == 0:
                ax.scatter(traj_2d[0, 0], traj_2d[0, 1], color=start_color_rl, marker='^', s=100, zorder=8, edgecolors='black', label='RL Start')
                ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], color=start_color_rl, marker='*', s=150, zorder=9, edgecolors='black', label='RL Goal')
            else:
                ax.scatter(traj_2d[0, 0], traj_2d[0, 1], color=start_color_rl, marker='^', s=100, zorder=8, edgecolors='black')
                ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], color=start_color_rl, marker='*', s=150, zorder=9, edgecolors='black')
            
        ax.set_xlabel("Principal Component 1", fontsize=12)
        ax.set_ylabel("Principal Component 2", fontsize=12)
        ax.grid(True, alpha=0.3)
        if sc is not None:
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=12, fontweight='bold')
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best')

    ax1 = plt.subplot(1, 2, 1)
    plot_single_crossview(ax1, f_demo_2d, z_f_demo, rl_traj_f_list, pca_f, S_w_global, 
                          "Main Camera PCA (Colored by S_wrist)", "Guidance from Wrist Camera (S_wrist)")
    ax2 = plt.subplot(1, 2, 2)
    plot_single_crossview(ax2, w_demo_2d, z_w_demo, rl_traj_w_list, pca_w, S_f_global, 
                          "Wrist Camera PCA (Colored by S_main)", "Guidance from Main Camera (S_main)")

    if step is not None: plt.suptitle(f"Latent Cross-View PCA at Step {step}", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_representative_1d_scores(z_f_rl, z_w_rl, script_dir, save_path, step=None, e2c_dir=None, num_episodes=4, gamma_f=None, gamma_w=None, is_2squared=False):
    import os
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scipy.spatial.distance as dist
    
    if e2c_dir is None: return
    demo_latent_path = os.path.join(e2c_dir, "demo_latents.pt")
    if not os.path.exists(demo_latent_path): return
        
    demo_data = torch.load(demo_latent_path, map_location="cpu", weights_only=False)
    z_f_demo = demo_data.get("z_demo_main", demo_data.get("z_demo_front"))
    z_w_demo = demo_data["z_demo_wrist"]
    
    all_z_f_demo = np.concatenate([z.squeeze(0).numpy() for z in z_f_demo], axis=0)
    all_z_w_demo = np.concatenate([z.squeeze(0).numpy() for z in z_w_demo], axis=0)
    
    rl_traj_f_list = [arr for arr in [_to_np(z) for z in z_f_rl] if len(arr) > 0]
    rl_traj_w_list = [arr for arr in [_to_np(z) for z in z_w_rl] if len(arr) > 0]
    
    if len(rl_traj_f_list) == 0 or len(rl_traj_w_list) == 0:
        return
    
    all_z_f_rl_flat = np.vstack(rl_traj_f_list)
    all_z_w_rl_flat = np.vstack(rl_traj_w_list)
    
    dist_f = dist.cdist(all_z_f_rl_flat, all_z_f_demo, 'sqeuclidean')
    min_dist_f = dist_f.min(axis=1)
    dist_w = dist.cdist(all_z_w_rl_flat, all_z_w_demo, 'sqeuclidean')
    min_dist_w = dist_w.min(axis=1)
    
    if gamma_f is not None and gamma_w is not None:
        # Use the exact reward shaping values
        _gamma_f = gamma_f
        _gamma_w = gamma_w
        if not is_2squared:
            # 4th power kernel uses squared min_dist because min_dist is already squared
            S_f_global = np.exp(-_gamma_f * (min_dist_f ** 2))
            S_w_global = np.exp(-_gamma_w * (min_dist_w ** 2))
        else:
            S_f_global = np.exp(-_gamma_f * min_dist_f)
            S_w_global = np.exp(-_gamma_w * min_dist_w)
    else:
        # Fallback normalization for offline tests
        _gamma_f = 2.0 / (np.mean(min_dist_f) + 1e-8)
        S_f_global = np.exp(-_gamma_f * min_dist_f)
        _gamma_w = 2.0 / (np.mean(min_dist_w) + 1e-8)
        S_w_global = np.exp(-_gamma_w * min_dist_w)
    
    # Calculate OOD variance metric for each trajectory
    idx = 0
    traj_metrics = []
    for i, traj in enumerate(rl_traj_f_list):
        n = len(traj)
        sf = S_f_global[idx:idx+n]
        sw = S_w_global[idx:idx+n]
        idx += n
        
        # Metric: how strongly one view was ID while the other was OOD
        # High sum of absolute differences = strong asymmetric guidance
        metric = np.sum(np.abs(sf - sw))
        traj_metrics.append((metric, sf, sw, i))
        
    # Sort by highest metric and take top N
    traj_metrics.sort(key=lambda x: x[0], reverse=True)
    top_trajs = traj_metrics[:num_episodes]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (metric, sf, sw, orig_idx) in enumerate(top_trajs):
        ax = axes[i]
        timesteps = np.arange(len(sf))
        ax.plot(timesteps, sf, label='S_main (Main Camera)', color='#2ca02c', linewidth=2.5)
        ax.plot(timesteps, sw, label='S_wrist (Wrist Camera)', color='#d62728', linewidth=2.5)
        
        # Dynamic Shading
        ax.fill_between(timesteps, sf, sw, where=(sf >= sw), color='#2ca02c', alpha=0.2, interpolate=True, label='Main Dominant')
        ax.fill_between(timesteps, sf, sw, where=(sw > sf), color='#d62728', alpha=0.2, interpolate=True, label='Wrist Dominant')
        
        ax.set_title(f"Episode {orig_idx+1} (Length: {len(sf)})", fontsize=12)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Similarity Score [0, 1]")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='best')
            
    if step is not None:
        plt.suptitle(f"Top {len(top_trajs)} OOD-Adapted Episodes at Step {step}", fontsize=16)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()



def create_top3_score_video(video_paths, z_f_list, z_w_list, save_path, e2c_dir, gamma_f, gamma_w, is_2squared):
    import os
    import cv2
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scipy.spatial.distance as dist
    import imageio

    if e2c_dir is None: return
    demo_latent_path = os.path.join(e2c_dir, "demo_latents.pt")
    if not os.path.exists(demo_latent_path): return

    demo_data = torch.load(demo_latent_path, map_location="cpu", weights_only=False)
    z_f_demo = demo_data.get("z_demo_main", demo_data.get("z_demo_front"))
    z_w_demo = demo_data["z_demo_wrist"]
    all_z_f_demo = np.concatenate([z.squeeze(0).numpy() for z in z_f_demo], axis=0)
    all_z_w_demo = np.concatenate([z.squeeze(0).numpy() for z in z_w_demo], axis=0)

    # Calculate metrics to find top 3
    traj_metrics = []
    for i, (zf, zw) in enumerate(zip(z_f_list, z_w_list)):
        zf_arr = _to_np(zf)
        zw_arr = _to_np(zw)
        if len(zf_arr) == 0 or len(zw_arr) == 0:
            continue
        
        df = dist.cdist(zf_arr, all_z_f_demo, 'sqeuclidean').min(axis=1)
        dw = dist.cdist(zw_arr, all_z_w_demo, 'sqeuclidean').min(axis=1)
        
        if not is_2squared:
            sf = np.exp(-gamma_f * (df ** 2))
            sw = np.exp(-gamma_w * (dw ** 2))
        else:
            sf = np.exp(-gamma_f * df)
            sw = np.exp(-gamma_w * dw)
            
        metric = np.sum(np.abs(sf - sw))
        traj_metrics.append((metric, sf, sw, i))

    if not traj_metrics:
        return

    # Sort and pick top 3
    traj_metrics.sort(key=lambda x: x[0], reverse=True)
    top_trajs = traj_metrics[:3]

    writer = imageio.get_writer(save_path, fps=20)

    for rank, (metric, sf, sw, orig_idx) in enumerate(top_trajs):
        if orig_idx >= len(video_paths): continue
        frames = np.load(video_paths[orig_idx])
        T = len(sf)
        
        orig_w = frames[0].shape[1]
        dpi = orig_w / 10.0

        fig, ax = plt.subplots(figsize=(10, 3), dpi=dpi)
        timesteps = np.arange(T)
        ax.plot(timesteps, sf, label='S_main', color='#2ca02c', linewidth=2.5)
        ax.plot(timesteps, sw, label='S_wrist', color='#d62728', linewidth=2.5)
        ax.fill_between(timesteps, sf, sw, where=(sf >= sw), color='#2ca02c', alpha=0.2, interpolate=True)
        ax.fill_between(timesteps, sf, sw, where=(sw > sf), color='#d62728', alpha=0.2, interpolate=True)

        ax.set_title(f"Rank {rank+1} OOD Adaptation (Original Ep: {orig_idx+1})", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, max(1, T-1))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        vline = ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
        fig.tight_layout()

        for t in range(min(T, len(frames))):
            vline.set_xdata([t, t])
            fig.canvas.draw()
            if hasattr(fig.canvas, 'buffer_rgba'):
                plot_img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            else:
                plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            if plot_img.shape[1] != frames[t].shape[1]:
                plot_img = cv2.resize(plot_img, (frames[t].shape[1], plot_img.shape[0]))
                
            combined = np.concatenate([frames[t], plot_img], axis=0)
            writer.append_data(combined)

        plt.close(fig)
        
    writer.close()
