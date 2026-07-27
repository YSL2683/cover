import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import imageio
from pathlib import Path
from tensordict import TensorDict

def run_visualizations(env, agent, lane_shaper, cfg):
    print("Starting Dashboard and t-SNE Visualizations...")
    device = agent.device if hasattr(agent, "device") else torch.device("cuda")
    agent.eval()
    
    # 1. Extract Expert DINO embeddings
    print("Extracting DINO embeddings from expert buffer...")
    dino_all = lane_shaper.offline_rb._storage._storage.get("dino")
    dino_f_exp = dino_all[:, :384].numpy()
    dino_w_exp = dino_all[:, 384:].numpy()
    
    print("Initializing E2C representations from pretrained weights...")
    if not lane_shaper.initialized:
        e2c_dir = "/home/moai/ysl_ws/cover/lane/pretrained_e2c/lift"
        lane_shaper.e2c_front.load_state_dict(torch.load(f"{e2c_dir}/e2c_front.pt", map_location=device))
        lane_shaper.e2c_wrist.load_state_dict(torch.load(f"{e2c_dir}/e2c_wrist.pt", map_location=device))
        lane_shaper.e2c_front.eval()
        lane_shaper.e2c_wrist.eval()
        lane_shaper.initialize_demos()
    
    # 2. Rollout and Buffer Data
    print("Running Rollout...")
    
    max_attempts = 20
    successful_episode_data = None
    
    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts}...")
        obs, info = env.reset()
        buffered_data = []
        rewards = []
        step = 0
        max_steps = 300
        success = False
        
        while step < max_steps:
            with torch.no_grad():
                action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)
                
            next_obs, env_reward, done, truncated, info = env.step(action)
            
            img_front = obs["observation.images.frontview"].float()
            img_wrist = obs["observation.images.robot0_eye_in_hand"].float()
            
            with torch.no_grad():
                obs_img = torch.cat([img_front, img_wrist], dim=1).to(device)
                dino_emb = lane_shaper.dino_embed(obs_img)
                
                d_f = dino_emb[:, :384].cpu().numpy()[0]
                d_w = dino_emb[:, 384:].cpu().numpy()[0]
                
                dummy_batch = {
                    ("next", "dino"): dino_emb,
                    "nonterminal": torch.tensor([[True]], device=device),
                    ("next", "reward"): torch.tensor([[0.0]], device=device),
                    "action": torch.zeros((1, 7), device=device)
                }
                dummy_td = TensorDict(dummy_batch, batch_size=[1])
                
                lane_stats = lane_shaper.shape_reward(dummy_td, step=1)
                r_val = dummy_td["next", "reward"].item()
                
                # Log to wandb so we can track dense reward in eval
                import wandb
                if wandb.run is not None:
                    wandb.log({f"eval_viz/{k}": v for k, v in lane_stats.items()})
                rewards.append(r_val)
            
            f_im_tensor = obs["observation.images.frontview"][0].permute(1, 2, 0).cpu().numpy()
            w_im_tensor = obs["observation.images.robot0_eye_in_hand"][0].permute(1, 2, 0).cpu().numpy()
            
            f_im = (f_im_tensor * 255.0).astype(np.uint8)
            w_im = (w_im_tensor * 255.0).astype(np.uint8)
            
            buffered_data.append({
                'f_im': f_im,
                'w_im': w_im,
                'd_f': d_f,
                'd_w': d_w,
                'reward': r_val
            })
            
            obs = next_obs
            step += 1
            
            # Use info["success"] or env_reward to check success
            if info.get("success", False) or env_reward > 0.5:
                success = True
                
            if done.any() or truncated.any():
                break
                
        if success:
            print(f"Success found on attempt {attempt + 1}!")
            successful_episode_data = (buffered_data, rewards)
            break
        else:
            print(f"Attempt {attempt + 1} failed. Trying again...")
            
    if successful_episode_data is None:
        print("Warning: Could not find a successful episode! Using the last failed attempt.")
        successful_episode_data = (buffered_data, rewards)
        
    buffered_data, rewards = successful_episode_data
            
    # 3. Compute t-SNE for combined data
    print("Computing t-SNE on Expert + Rollout embeddings...")
    rollout_d_f = np.stack([d['d_f'] for d in buffered_data])
    rollout_d_w = np.stack([d['d_w'] for d in buffered_data])
    
    combined_f = np.concatenate([dino_f_exp, rollout_d_f], axis=0)
    combined_w = np.concatenate([dino_w_exp, rollout_d_w], axis=0)
    
    perplexity = min(30, len(dino_f_exp) // 2)
    tsne_f = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    tsne_w = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    
    z_f_tsne = tsne_f.fit_transform(combined_f)
    z_w_tsne = tsne_w.fit_transform(combined_w)
    
    n_exp = len(dino_f_exp)
    z_f_tsne_exp = z_f_tsne[:n_exp]
    z_f_tsne_rollout = z_f_tsne[n_exp:]
    
    z_w_tsne_exp = z_w_tsne[:n_exp]
    z_w_tsne_rollout = z_w_tsne[n_exp:]
    
    # 4. Generate frames
    print("Generating frames...")
    frames = []
    current_rewards = []
    
    def get_lims(z_exp, z_roll, margin=0.5):
        all_z = np.vstack([z_exp, z_roll])
        z_min = all_z.min(axis=0)
        z_max = all_z.max(axis=0)
        z_range = z_max - z_min
        z_range = np.maximum(z_range, 1.0)
        return (z_min[0] - margin * z_range[0], z_max[0] + margin * z_range[0]), \
               (z_min[1] - margin * z_range[1], z_max[1] + margin * z_range[1])

    xlim_f, ylim_f = get_lims(z_f_tsne_exp, z_f_tsne_rollout, margin=0.3)
    xlim_w, ylim_w = get_lims(z_w_tsne_exp, z_w_tsne_rollout, margin=0.3)
    
    for i, data in enumerate(buffered_data):
        current_rewards.append(data['reward'])
        
        fig = plt.figure(figsize=(15, 10))
        
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(data['f_im'])
        ax1.set_title("Main Camera")
        ax1.axis('off')
        
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.imshow(data['w_im'])
        ax2.set_title("Wrist Camera")
        ax2.axis('off')
        
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(current_rewards, color='r')
        ax3.set_xlim(0, len(buffered_data))
        
        # Set dynamic y-axis limits to clearly show the reward variance
        if max(rewards) > min(rewards):
            y_min = min(0.0, min(rewards))
            y_max = max(1e-4, max(rewards)) * 1.1
            ax3.set_ylim(y_min, y_max)
        else:
            ax3.set_ylim(-0.1, 0.1)
            
        ax3.set_title("Reward Over Time")
        ax3.set_xlabel("Steps")
        
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.scatter(z_f_tsne_exp[:, 0], z_f_tsne_exp[:, 1], c='gray', alpha=0.5, s=20, label="Expert Data")
        ax4.plot(z_f_tsne_rollout[:i+1, 0], z_f_tsne_rollout[:i+1, 1], c='b', linewidth=2, label="Rollout")
        ax4.scatter(z_f_tsne_rollout[i, 0], z_f_tsne_rollout[i, 1], c='r', s=50)
        ax4.set_xlim(xlim_f)
        ax4.set_ylim(ylim_f)
        ax4.set_title("Main Camera - t-SNE Manifold")
        
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.scatter(z_w_tsne_exp[:, 0], z_w_tsne_exp[:, 1], c='gray', alpha=0.5, s=20, label="Expert Data")
        ax5.plot(z_w_tsne_rollout[:i+1, 0], z_w_tsne_rollout[:i+1, 1], c='g', linewidth=2, label="Rollout")
        ax5.scatter(z_w_tsne_rollout[i, 0], z_w_tsne_rollout[i, 1], c='r', s=50)
        ax5.set_xlim(xlim_w)
        ax5.set_ylim(ylim_w)
        ax5.set_title("Wrist Camera - t-SNE Manifold")
        
        plt.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(img)
        plt.close(fig)
        
    out_dir = Path("/home/moai/ysl_ws/cover/resfit/my_lerobot_data/REWARD_2/outputs/REWARD_2")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dashboard_analysis.mp4"
    print(f"Saving video to {out_path} ...")
    imageio.mimsave(out_path, frames, fps=10)
    print("Done! Visualizations complete.")
