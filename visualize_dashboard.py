import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import imageio
from pathlib import Path

# Hydra and environment setup imports
from hydra import compose, initialize
from omegaconf import OmegaConf
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Model imports
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.rl_finetuning.wrappers.residual_env_wrapper import BasePolicyVecEnvWrapper
from resfit.lerobot.utils.load_policy import load_policy

# E2C / DINO imports
from lane.e2c import MLPE2C
from visualize_rl_latents import get_dino_features

class PBRSCalculator:
    def __init__(self, demo_latent_path, beta=1.0, alpha=0.98, gamma=0.99, w_m=0.5, w_w=0.5):
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.w_m = w_m
        self.w_w = w_w
        
        print(f"Loading demo latents from {demo_latent_path}...")
        demo_data = torch.load(demo_latent_path, map_location="cpu", weights_only=False)
        self.z_demo_main_cache = [z.numpy() for z in demo_data["z_demo_front"]]
        self.z_demo_wrist_cache = [z.numpy() for z in demo_data["z_demo_wrist"]]
        
        # Calculate reference distances
        self.ref_one_step_dist_main = self._calc_ref_dist(self.z_demo_main_cache)
        self.ref_one_step_dist_wrist = self._calc_ref_dist(self.z_demo_wrist_cache)
        print(f"Ref dist main: {self.ref_one_step_dist_main:.6f}, wrist: {self.ref_one_step_dist_wrist:.6f}")

    def _calc_ref_dist(self, z_list):
        total_dist = 0
        total_steps = 0
        for z in z_list:
            if z.shape[1] > 1:
                dist = np.sum((z[0, :-1] - z[0, 1:])**2, axis=1).sum()
                total_dist += dist
                total_steps += z.shape[1] - 1
        return total_dist / max(total_steps, 1)

    def compute_potential(self, z_pred_m, z_pred_w):
        # Ensure correct shape (1, 1, 16)
        if len(z_pred_m.shape) == 1:
            z_pred_m = z_pred_m[None, None, :]
            z_pred_w = z_pred_w[None, None, :]
        elif len(z_pred_m.shape) == 2:
            z_pred_m = z_pred_m[:, None, :]
            z_pred_w = z_pred_w[:, None, :]
            
        N = len(z_pred_m)
        min_dist_m = np.ones(N) * 10000
        min_dist_w = np.ones(N) * 10000
        idx_m_best = np.zeros(N)
        idx_w_best = np.zeros(N)
        T_demos_m = np.zeros(N)
        T_demos_w = np.zeros(N)
        
        for i in range(len(self.z_demo_main_cache)):
            z_demo_m = self.z_demo_main_cache[i]
            z_dist_m = ((z_demo_m - z_pred_m) ** 2).sum(axis=2)
            z_dist_min_m = z_dist_m.min(axis=1)
            update_min_m = z_dist_min_m < min_dist_m
            min_dist_m[update_min_m] = z_dist_min_m[update_min_m]
            idx_m_best[update_min_m] = z_dist_m.argmin(axis=1)[update_min_m]
            T_demos_m[update_min_m] = z_dist_m.shape[1]
            
            z_demo_w = self.z_demo_wrist_cache[i]
            z_dist_w = ((z_demo_w - z_pred_w) ** 2).sum(axis=2)
            z_dist_min_w = z_dist_w.min(axis=1)
            update_min_w = z_dist_min_w < min_dist_w
            min_dist_w[update_min_w] = z_dist_min_w[update_min_w]
            idx_w_best[update_min_w] = z_dist_w.argmin(axis=1)[update_min_w]
            T_demos_w[update_min_w] = z_dist_w.shape[1]
            
        gamma_m = self.beta / (self.ref_one_step_dist_main + 1e-8)
        gamma_w = self.beta / (self.ref_one_step_dist_wrist + 1e-8)
        
        S_main = np.exp(-gamma_m * min_dist_m)
        S_wrist = np.exp(-gamma_w * min_dist_w)
        
        rem_t_m = T_demos_m - idx_m_best
        rem_t_w = T_demos_w - idx_w_best
        
        Phi = (self.w_m * np.power(self.alpha, rem_t_m) * S_main) + (self.w_w * np.power(self.alpha, rem_t_w) * S_wrist)
        return Phi[0], S_main[0], S_wrist[0]


def generate_video(episodes_data, output_path):
    print(f"Generating video to {output_path}...")
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(6, 2, figure=fig, width_ratios=[1.2, 2.0], hspace=1.0)
    
    ax_main = fig.add_subplot(gs[0:3, 0])
    ax_wrist = fig.add_subplot(gs[3:6, 0])
    
    ax_sim = fig.add_subplot(gs[0:2, 1])
    ax_f = fig.add_subplot(gs[2:4, 1])
    ax_ares = fig.add_subplot(gs[4:6, 1])
    
    ep_text = fig.text(0.01, 0.98, "", fontsize=20, fontweight='bold', verticalalignment='top')
    
    writer = imageio.get_writer(output_path, fps=10)
    
    for ep_idx, ep_data in enumerate(episodes_data):
        ep_text.set_text(f"Episode {ep_idx+1}/5")
        img_main_list = ep_data["img_main"]
        img_wrist_list = ep_data["img_wrist"]
        s_main_list = ep_data["s_main"]
        s_wrist_list = ep_data["s_wrist"]
        f_reward_list = ep_data["f_reward"]
        a_res_norm_list = ep_data["a_res_norm"]
        
        for ax in [ax_sim, ax_f, ax_ares]:
            ax.clear()
            
        ax_sim.set_title("Latent Similarity (S_main: Blue, S_wrist: Yellow)")
        ax_sim.set_xlim(0, len(s_main_list))
        ax_sim.set_ylim(0, 1.1)
        ax_sim.grid(True, alpha=0.3)
        ax_sim.plot(s_main_list, color='blue', label='S_main')
        ax_sim.plot(s_wrist_list, color='orange', label='S_wrist')
        ax_sim.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        ax_f.set_title("PBRS Dense Reward F(s, a, s')")
        ax_f.set_xlim(0, len(f_reward_list))
        y_min, y_max = min(f_reward_list), max(f_reward_list)
        y_padding = max((y_max - y_min) * 0.1, 0.1)
        ax_f.set_ylim(y_min - y_padding, y_max + y_padding)
        ax_f.grid(True, alpha=0.3)
        ax_f.plot(f_reward_list, color='green')
        
        ax_ares.set_title("Residual Action Magnitude ||a_res||_2")
        ax_ares.set_xlim(0, len(a_res_norm_list))
        ax_ares.set_ylim(0, max(max(a_res_norm_list)*1.1, 0.1))
        ax_ares.grid(True, alpha=0.3)
        ax_ares.plot(a_res_norm_list, color='purple')
        
        fig.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])
        
        for t in range(len(img_main_list)):
            ax_main.clear()
            ax_wrist.clear()
            
            ax_main.imshow(img_main_list[t])
            ax_main.set_title("Main Camera")
            ax_main.axis("off")
            
            ax_wrist.imshow(img_wrist_list[t])
            ax_wrist.set_title("Wrist Camera")
            ax_wrist.axis("off")
            
            # Remove previous tracking lines
            for ax in [ax_sim, ax_f, ax_ares]:
                for line in ax.lines:
                    if line.get_label() == "tracker":
                        line.remove()
                        
            ax_sim.axvline(x=t, color='red', linestyle='--', label="tracker")
            ax_f.axvline(x=t, color='red', linestyle='--', label="tracker")
            ax_ares.axvline(x=t, color='red', linestyle='--', label="tracker")
            
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(image)
            
    writer.close()
    plt.close(fig)
    print("Video generation complete!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_policy_path", type=str, required=True)
    parser.add_argument("--residual_policy_path", type=str, required=True)
    parser.add_argument("--demo_latent_path", type=str, default="/home/moai/ysl_ws/cover/lane/pretrained_e2c/lift/demo_latents.pt")
    parser.add_argument("--ood_range", type=float, default=None)
    parser.add_argument("--reward_scale", type=float, default=100.0)
    args = parser.parse_args()

    residual_path_obj = Path(args.residual_policy_path).resolve()
    run_dir = residual_path_obj.parent.parent
    vis_dir = run_dir / "visualization"
    out_path = str(vis_dir / "dashboard.mp4")

    if args.ood_range is not None:
        os.environ["EXPERIMENT_MODE"] = "OOD_pos"
        os.environ["OOD_RANGE"] = str(args.ood_range)
        print(f"Set environment variables for OOD: OOD_RANGE={args.ood_range}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with initialize(version_base=None, config_path="resfit/rl_finetuning/config"):
        cfg = compose(
            config_name="residual_td3_dexmg_config",
            overrides=[
                "task=Lift",
                "rl_camera=['observation.images.frontview','observation.images.robot0_eye_in_hand']"
            ]
        )

    print("Loading models...")
    base_policy = load_policy(Path(args.base_policy_path)).to(device)
    base_policy.eval()

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

    env_modifier = cfg.get("env_modifier", None)
    if env_modifier is not None:
        env_modifier.disturbance = None
        if args.ood_range is not None:
            env_modifier.ood_position.x_bounds = [-args.ood_range, args.ood_range]
            env_modifier.ood_position.y_bounds = [-args.ood_range, args.ood_range]

    vec_env = create_vectorized_env(
        env_name=cfg.task,
        num_envs=1,
        device=device,
        video_key=cfg.video_key,
        debug=False,
        camera_size=128,
        env_modifier_config=env_modifier,
    )
    env = BasePolicyVecEnvWrapper(vec_env, base_policy, action_scaler, state_standardizer)

    img_c, img_h, img_w = vec_env.observation_space[cfg.rl_camera[0]].shape[1:]
    lowdim_dim = vec_env.observation_space["observation.state"].shape[1]
    action_dim = vec_env.action_space.shape[1]

    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=list(cfg.rl_camera),
        cfg=cfg.agent,
        residual_actor=True,
    )
    agent.load_state_dict(torch.load(args.residual_policy_path, map_location=device, weights_only=True))
    agent.eval()

    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(device)
    dino.eval()
    e2c_f = MLPE2C(obs_shape=(384,), action_dim=action_dim, z_dimension=16).to(device)
    e2c_w = MLPE2C(obs_shape=(384,), action_dim=action_dim, z_dimension=16).to(device)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    e2c_f.load_state_dict(torch.load(os.path.join(script_dir, "lane/pretrained_e2c/lift/e2c_front.pt"), map_location=device))
    e2c_w.load_state_dict(torch.load(os.path.join(script_dir, "lane/pretrained_e2c/lift/e2c_wrist.pt"), map_location=device))
    e2c_f.eval()
    e2c_w.eval()

    pbrs_calc = PBRSCalculator(args.demo_latent_path)

    # Collect 5 successful episodes
    print("Collecting 5 successful episodes...")
    max_steps = 150
    success_count = 0
    
    episodes_data = []
    
    while success_count < 5:
        seed = np.random.randint(0, 1000000)
        obs, _ = env.reset(seed=seed)
        
        ep_img_main, ep_img_wrist = [], []
        ep_s_main, ep_s_wrist = [], []
        ep_f_reward = []
        ep_a_res_norm = []
        
        phi_prev = None
        success = False
        
        for step in range(max_steps):
            front_img = obs["observation.images.frontview"]
            wrist_img = obs["observation.images.robot0_eye_in_hand"]
            obs_img = torch.cat([front_img, wrist_img], dim=1)
            
            # Store images for visualization (C, H, W to H, W, C)
            img_m_np = (front_img.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img_w_np = (wrist_img.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            ep_img_main.append(img_m_np)
            ep_img_wrist.append(img_w_np)

            # Compute features & phi
            feat_f, feat_w = get_dino_features(obs_img, dino, device)
            with torch.no_grad():
                zf, _ = e2c_f.enc(feat_f)
                zw, _ = e2c_w.enc(feat_w)
                
            phi_curr, s_main, s_wrist = pbrs_calc.compute_potential(zf.cpu().numpy(), zw.cpu().numpy())
            ep_s_main.append(s_main)
            ep_s_wrist.append(s_wrist)
            
            if step > 0:
                f_reward = (0.99 * phi_curr - phi_prev) * args.reward_scale
                ep_f_reward.append(f_reward)
            
            phi_prev = phi_curr
            
            with torch.no_grad():
                # Action inference
                action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)
                
            # agent.act returns the residual action directly. 
            a_res_norm = torch.norm(action, p=2).item()
            ep_a_res_norm.append(a_res_norm)
                
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            if "final_info" in info and isinstance(info["final_info"], list) and info["final_info"][0] is not None:
                if info["final_info"][0].get("success", False):
                    success = True
            if reward > 0.5:
                success = True

            if terminated.any() or truncated.any():
                break

            obs = next_obs
            
        # The last step doesn't have an F_reward calculation (we append 0 or duplicate last)
        if len(ep_f_reward) < len(ep_s_main):
            ep_f_reward.append(0.0 if len(ep_f_reward) == 0 else ep_f_reward[-1])

        if success:
            success_count += 1
            print(f"Episode {success_count}/5 successful. Length: {len(ep_s_main)}")
            episodes_data.append({
                "img_main": ep_img_main,
                "img_wrist": ep_img_wrist,
                "s_main": ep_s_main,
                "s_wrist": ep_s_wrist,
                "f_reward": ep_f_reward,
                "a_res_norm": ep_a_res_norm
            })
        else:
            print("Episode failed, retrying...")

    print(f"Collected 5 successful episodes.")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    generate_video(episodes_data, out_path)

if __name__ == "__main__":
    main()
