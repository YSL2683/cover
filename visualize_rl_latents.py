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
    z_f_demo = demo_data["z_demo_front"]
    z_w_demo = demo_data["z_demo_wrist"]
    num_demos = len(z_f_demo)
    
    # Load RL latents
    has_rl = rl_latent_path is not None and os.path.exists(rl_latent_path)
    if has_rl:
        rl_data = torch.load(rl_latent_path, map_location="cpu", weights_only=False)
        z_f_rl = rl_data["z_rl_front"]
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
    
    # --- 1. PCA Scatter Plot ---
    plt.figure(figsize=(14, 6))
    
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Front Camera (PCA projected on ID variance)")
    ax1.scatter(f_demo_2d[:, 0], f_demo_2d[:, 1], color='darkorange', s=20, alpha=0.6, label='ID Demo')
    if num_rl > 0:
        ax1.scatter(f_rl_2d[:, 0], f_rl_2d[:, 1], color='darkblue', s=20, alpha=0.6, label='RL Success')
    ax1.set_xlabel("Principal Component 1")
    ax1.set_ylabel("Principal Component 2")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Wrist Camera (PCA projected on ID variance)")
    ax2.scatter(w_demo_2d[:, 0], w_demo_2d[:, 1], color='darkorange', s=20, alpha=0.6, label='ID Demo')
    if num_rl > 0:
        ax2.scatter(w_rl_2d[:, 0], w_rl_2d[:, 1], color='darkblue', s=20, alpha=0.6, label='RL Success')
    ax2.set_xlabel("Principal Component 1")
    ax2.set_ylabel("Principal Component 2")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    if rl_latent_path:
        pca_save_path = os.path.join(os.path.dirname(rl_latent_path), "pca_comparison.png")
    else:
        pca_save_path = os.path.join(script_dir, "outputs/pca_comparison.png")
    
    os.makedirs(os.path.dirname(pca_save_path), exist_ok=True)
    plt.savefig(pca_save_path, dpi=150)
    print(f"PCA Plot saved to {pca_save_path}")
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_policy_path", type=str, required=True, help="Path to base policy directory")
    parser.add_argument("--residual_policy_path", type=str, required=True, help="Path to residual agent checkpoint (.pt)")
    parser.add_argument("--n_episodes", type=int, default=20, help="Number of successful episodes to collect")
    parser.add_argument("--ood_range", type=float, default=None, help="OOD initialization range in meters (e.g. 0.05 or 0.1). If provided, applies OOD_pos environment setup.")
    args = parser.parse_args()

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
                "rl_camera=['observation.images.frontview','observation.images.robot0_eye_in_hand']"
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
    e2c_f.load_state_dict(torch.load(os.path.join(script_dir, "lane/pretrained_e2c/lift/e2c_front.pt"), map_location=device))
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

            front_img = obs["observation.images.frontview"]
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
        "z_rl_front": z_rl_front,
        "z_rl_wrist": z_rl_wrist,
        "rl_lengths": rl_lengths
    }, out_path)
    print(f"Saved RL latents to {out_path}")

    print("\n--- Starting PCA & Distance Visualization ---")
    visualize_analysis(out_path, script_dir)

if __name__ == "__main__":
    main()
